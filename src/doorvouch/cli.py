"""Command line interface: ``doorvouch <command>``.

Commands that need the camera import OpenCV and MediaPipe only when they run,
so listing people or reading the event log works without them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from doorvouch import __version__, models
from doorvouch.config import Settings
from doorvouch.core.engine import EngineEvent, VouchingEngine
from doorvouch.errors import DoorVouchError
from doorvouch.storage import SqliteStore

Handler = Callable[[argparse.Namespace, Settings], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doorvouch",
        description="Home entrance assistant: face recognition plus an admin's thumbs-up.",
    )
    parser.add_argument("--version", action="version", version=f"doorvouch {__version__}")
    parser.add_argument(
        "--home", metavar="DIR", help="data folder (default: $DOORVOUCH_HOME, else ./data)"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    doctor = commands.add_parser("doctor", help="check that this computer can run DoorVouch")
    doctor.add_argument("--camera", type=int, default=0, metavar="N", help="camera to test")
    doctor.add_argument("--no-camera", action="store_true", help="skip the camera check")
    doctor.set_defaults(handler=_doctor)

    model = commands.add_parser("models", help="check or download the model files")
    model.add_argument("action", nargs="?", choices=["status", "download"], default="status")
    model.add_argument("--force", action="store_true", help="download again even if present")
    model.set_defaults(handler=_models)

    enroll = commands.add_parser("enroll", help="add a person, or more samples of one")
    enroll.add_argument("--name", required=True, help="the person's name")
    enroll.add_argument("--admin", action="store_true", help="may vouch for visitors")
    enroll.add_argument("--samples", type=int, default=8, help="face samples to capture")
    _camera_options(enroll)
    enroll.set_defaults(handler=_enroll)

    run = commands.add_parser("run", help="watch the camera and apply the thumbs-up rule")
    run.add_argument("--seconds", type=float, help="stop after this many seconds")
    _camera_options(run)
    run.set_defaults(handler=_run)

    people = commands.add_parser("people", help="list or remove enrolled people")
    people.add_argument("action", nargs="?", choices=["list", "remove"], default="list")
    people.add_argument("id", nargs="?", type=int, help="person to remove")
    people.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    people.set_defaults(handler=_people)

    pending = commands.add_parser("pending", help="list, name or discard pending visitors")
    pending.add_argument("action", nargs="?", choices=["list", "name", "discard"], default="list")
    pending.add_argument("id", nargs="?", type=int, help="pending visitor number")
    pending.add_argument("name", nargs="?", help="name to give the visitor")
    pending.add_argument("--admin", action="store_true", help="make the new person an admin")
    pending.set_defaults(handler=_pending)

    events = commands.add_parser("events", help="show the most recent events")
    events.add_argument("--limit", type=int, default=20, help="how many to show")
    events.set_defaults(handler=_events)

    trial = commands.add_parser("trial", help="run the V0 acceptance trial")
    trial.add_argument("--scenario", required=True, choices=["vouch", "impostor"])
    trial.add_argument("--attempts", type=int, default=10)
    trial.add_argument("--seconds", type=float, default=8.0, help="length of each attempt")
    trial.add_argument("--pause", type=float, default=5.0, help="countdown before each attempt")
    trial.add_argument("--output", metavar="FILE", help="append the report to this file")
    _camera_options(trial)
    trial.set_defaults(handler=_trial)
    return parser


def _camera_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=0, metavar="N", help="camera number")
    parser.add_argument("--no-window", action="store_true", help="do not open a preview window")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerate_unprintable_characters()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args, Settings.load(home=args.home))
    except DoorVouchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


def _tolerate_unprintable_characters() -> None:
    """Keep a name the console cannot display from crashing a command."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


# Commands that do not need a camera ---------------------------------------------


def _doctor(args: argparse.Namespace, settings: Settings) -> int:
    from doorvouch.apps.doctor import run_doctor

    healthy = run_doctor(settings, camera_index=None if args.no_camera else args.camera)
    return 0 if healthy else 1


def _models(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "download":
        for spec in models.ALL_MODELS:
            print(f"{spec.filename} ... ", end="", flush=True)
            found = models.download(spec, settings.models_dir, force=args.force)
            print(f"ok ({found.size:,} bytes, sha256 {found.sha256[:16]})")
        print(f"Models are in {settings.models_dir}")
        return 0

    rows = []
    ready = True
    for spec in models.ALL_MODELS:
        found = models.inspect(spec, settings.models_dir)
        ready = ready and found.ok
        rows.append([spec.filename, "ok" if found.ok else str(found.problem), spec.license])
    print(_table(["File", "State", "License"], rows))
    if not ready:
        print("Run `doorvouch models download` to fetch what is missing.")
    return 0 if ready else 1


def _people(args: argparse.Namespace, settings: Settings) -> int:
    with SqliteStore.open(settings.home) as store:
        if args.action == "remove":
            if args.id is None:
                raise DoorVouchError("Say who to remove: `doorvouch people remove ID`.")
            person = store.get_person(args.id)
            if person is None:
                raise DoorVouchError(f"There is no person #{args.id}.")
            question = f"Remove {person.name} with all their face samples and events?"
            if not args.yes and not _confirm(question):
                print("Nothing was removed.")
                return 1
            store.remove_person(person.id)
            print(f"Removed {person.name}.")
            return 0

        summaries = store.list_people()
        if not summaries:
            print("Nobody is enrolled yet. Start with `doorvouch enroll --name YOUR_NAME --admin`.")
            return 0
        rows = [
            [
                str(summary.person.id),
                summary.person.name,
                "admin" if summary.person.is_admin else "member",
                str(summary.samples),
                _local(summary.created_at),
            ]
            for summary in summaries
        ]
        print(_table(["ID", "Name", "Role", "Samples", "Since"], rows))
        return 0


def _pending(args: argparse.Namespace, settings: Settings) -> int:
    with SqliteStore.open(settings.home) as store:
        if args.action == "name":
            if args.id is None or not args.name:
                raise DoorVouchError("Usage: `doorvouch pending name ID NAME`.")
            person = store.name_pending(args.id, args.name, is_admin=args.admin)
            print(f"Pending visitor #{args.id} is now {person.name} (person #{person.id}).")
            return 0
        if args.action == "discard":
            if args.id is None:
                raise DoorVouchError("Usage: `doorvouch pending discard ID`.")
            if not store.discard_pending(args.id):
                raise DoorVouchError(f"There is no pending visitor #{args.id}.")
            print(f"Pending visitor #{args.id} was forgotten.")
            return 0

        waiting = store.list_pending()
        if not waiting:
            print("No pending visitors.")
            return 0
        names = {summary.person.id: summary.person.name for summary in store.list_people()}
        rows = [
            [
                str(pending.id),
                names.get(pending.vouched_by, "-") if pending.vouched_by is not None else "-",
                str(pending.seen_count),
                _local(pending.created_at),
                str(pending.snapshot_path) if pending.snapshot_path is not None else "-",
            ]
            for pending in waiting
        ]
        print(_table(["ID", "Vouched by", "Seen", "First seen", "Snapshot"], rows))
        print("Give one a name with `doorvouch pending name ID NAME`.")
        return 0


def _events(args: argparse.Namespace, settings: Settings) -> int:
    with SqliteStore.open(settings.home) as store:
        events = store.list_events(limit=args.limit)
        if not events:
            print("No events yet.")
            return 0
        names = {summary.person.id: summary.person.name for summary in store.list_people()}
        rows = [
            [
                _local(event.created_at),
                event.kind,
                names.get(event.person_id, "-") if event.person_id is not None else "-",
                f"#{event.pending_id}" if event.pending_id is not None else "-",
                json.dumps(event.detail, sort_keys=True) if event.detail else "",
            ]
            for event in events
        ]
        print(_table(["When", "Event", "Person", "Pending", "Detail"], rows))
        return 0


# Commands that use the camera ---------------------------------------------------


def _enroll(args: argparse.Namespace, settings: Settings) -> int:
    from doorvouch.apps.enroll import MIN_SAMPLES, collect_samples, save_enrollment
    from doorvouch.vision.camera import Camera
    from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer

    name = " ".join(args.name.split())
    if not name:
        raise DoorVouchError("A person needs a name.")
    if args.samples < MIN_SAMPLES:
        raise DoorVouchError(f"Capture at least {MIN_SAMPLES} samples.")
    paths = models.require(settings.models_dir, (models.FACE_DETECTOR, models.FACE_RECOGNIZER))
    analyzer = OpenCvFaceAnalyzer.from_models(paths["face_detector"], paths["face_recognizer"])

    display = _open_display(args, "DoorVouch - enroll")
    try:
        with Camera(args.camera) as camera:
            print(f"Enrolling {name}: look at the camera and turn your head slowly.")
            embeddings = collect_samples(
                camera.frames(), faces=analyzer, samples=args.samples, display=display, name=name
            )
    finally:
        if display is not None:
            display.close()

    with SqliteStore.open(settings.home) as store:
        enrollment = save_enrollment(
            store,
            embeddings,
            name=name,
            is_admin=args.admin,
            match_threshold=settings.engine.match_threshold,
        )
    person = enrollment.person
    role = "admin" if person.is_admin else "member"
    verb = "Enrolled" if enrollment.created else "Added samples to"
    print(
        f"{verb} {person.name} ({role}, person #{person.id}): {enrollment.samples_added} samples."
    )
    return 0


def _run(args: argparse.Namespace, settings: Settings) -> int:
    from doorvouch.apps.live import LiveStats, run_live
    from doorvouch.vision.camera import Camera
    from doorvouch.vision.opencv_faces import face_jpeg

    paths = models.require(settings.models_dir)
    with SqliteStore.open(settings.home) as store:
        if not store.has_admin():
            print(
                "warning: no admin is enrolled, so nobody can vouch for visitors yet. "
                "Run `doorvouch enroll --name YOUR_NAME --admin`.",
                file=sys.stderr,
            )
        faces, gestures = _vision(paths)
        engine = VouchingEngine(store, settings.engine)
        display = _open_display(args, "DoorVouch")
        stats = LiveStats()
        try:
            with Camera(args.camera) as camera:
                width, height = camera.resolution
                stop = "Ctrl+C" if display is None else "q in the window"
                print(f"Camera {args.camera} at {width}x{height}. Press {stop} to stop.")
                stats = run_live(
                    camera.frames(),
                    faces=faces,
                    gestures=gestures,
                    engine=engine,
                    snapshot=face_jpeg,
                    display=display,
                    on_event=_print_event,
                    max_seconds=args.seconds,
                )
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            gestures.close()
            if display is not None:
                display.close()
    if stats.frames:
        print(f"{stats.frames} frames in {stats.seconds:.1f} s: {stats.fps:.1f} FPS.")
    return 0


def _trial(args: argparse.Namespace, settings: Settings) -> int:
    from doorvouch.apps.trial import SCENARIOS, Attempt, run_trial
    from doorvouch.vision.camera import Camera

    def print_attempt(attempt: Attempt) -> None:
        print(
            f"Attempt {attempt.number}: {attempt.verdict} "
            f"(pending created: {attempt.pending_created}, "
            f"thumbs-up ignored: {attempt.gestures_ignored}, {attempt.fps:.1f} FPS)"
        )

    paths = models.require(settings.models_dir)
    with (
        SqliteStore.open(settings.home) as store,
        tempfile.TemporaryDirectory(prefix="doorvouch-trial-") as workdir,
    ):
        if not store.has_admin():
            raise DoorVouchError(
                "Enroll an admin first: `doorvouch enroll --name YOUR_NAME --admin`."
            )
        faces, gestures = _vision(paths)
        display = _open_display(args, "DoorVouch - trial")
        print(SCENARIOS[args.scenario])
        print(f"{args.attempts} attempts of {args.seconds:g} s, {args.pause:g} s apart.")
        try:
            with Camera(args.camera) as camera:
                report = run_trial(
                    camera.frames(),
                    scenario=args.scenario,
                    attempts=args.attempts,
                    seconds=args.seconds,
                    store=store,
                    config=settings.engine,
                    faces=faces,
                    gestures=gestures,
                    workdir=Path(workdir),
                    display=display,
                    pause_seconds=args.pause,
                    on_attempt=print_attempt,
                )
        finally:
            gestures.close()
            if display is not None:
                display.close()

    markdown = report.to_markdown()
    print()
    print(markdown)
    if args.output:
        with Path(args.output).open("a", encoding="utf-8") as handle:
            handle.write(f"\n{markdown}\n\nRun on {datetime.now():%Y-%m-%d %H:%M}.\n")
        print(f"\nReport appended to {args.output}")
    return 0 if report.accepted else 1


# Helpers ------------------------------------------------------------------------


def _vision(paths: dict[str, Path]) -> tuple[Any, Any]:
    from doorvouch.vision.mediapipe_gestures import MediaPipeGestureDetector
    from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer

    faces = OpenCvFaceAnalyzer.from_models(paths["face_detector"], paths["face_recognizer"])
    gestures = MediaPipeGestureDetector.from_model(paths["gesture_recognizer"])
    return faces, gestures


def _open_display(args: argparse.Namespace, title: str) -> Any:
    if args.no_window:
        return None
    from doorvouch.vision.overlay import OpenCvDisplay

    return OpenCvDisplay(title)


def _print_event(event: EngineEvent) -> None:
    print(f"{datetime.now():%H:%M:%S}  {event.message}", flush=True)


def _confirm(question: str) -> bool:
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _local(moment: datetime) -> str:
    return moment.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    return "\n".join(line.rstrip() for line in lines)
