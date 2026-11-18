"""The V0 acceptance trial: repeat a scenario and count how often the rule behaves."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from doorvouch.apps.live import Display, Frames, run_live
from doorvouch.core.engine import EngineConfig, VouchingEngine
from doorvouch.errors import DoorVouchError
from doorvouch.storage import SqliteStore
from doorvouch.vision.base import FaceAnalyzer, Frame, GestureDetector

Verdict = Literal["pass", "fail", "void"]

SCENARIOS = {
    "vouch": (
        "An admin and one visitor are in view and the admin gives a thumbs-up. "
        "An attempt passes when exactly one pending visitor is created."
    ),
    "impostor": (
        "Somebody who is not an admin gives a thumbs-up. An admin may be in view, with "
        "their hands down. An attempt passes when the thumbs-up is seen and ignored, fails "
        "when it is credited to an admin, and is void when no thumbs-up is detected at all."
    ),
}

# Share of the attempts that must pass for the scenario to count as accepted.
REQUIRED_PASS_RATE = {"vouch": 0.8, "impostor": 1.0}

# Events that only happen after the engine confirmed an admin's thumbs-up.
_VOUCH_EVENTS = ("pending_created", "pending_seen", "vouch_without_visitor", "visitor_skipped")


def judge(scenario: str, events: Mapping[str, int]) -> Verdict:
    """Decide one attempt from how often each kind of event happened in it."""
    if scenario == "vouch":
        return "pass" if events.get("pending_created", 0) == 1 else "fail"
    if any(events.get(kind, 0) for kind in _VOUCH_EVENTS):
        return "fail"  # a thumbs-up was credited to an admin who did not give one
    if events.get("gesture_ignored", 0) == 0:
        return "void"  # nobody's thumbs-up was detected, so the rule was not tested
    return "pass"


@dataclass(frozen=True, slots=True)
class Attempt:
    number: int
    verdict: Verdict
    vouch_confirmed: bool
    pending_created: int
    gestures_ignored: int
    frames: int
    fps: float

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass(frozen=True, slots=True)
class TrialReport:
    scenario: str
    attempts: tuple[Attempt, ...]
    requested: int

    @property
    def passed(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.verdict == "pass")

    @property
    def void(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.verdict == "void")

    @property
    def complete(self) -> bool:
        """Whether every requested attempt ran to the end of its window."""
        return 0 < self.requested <= len(self.attempts)

    @property
    def mean_fps(self) -> float:
        rates = [attempt.fps for attempt in self.attempts if attempt.frames > 1]
        return sum(rates) / len(rates) if rates else 0.0

    @property
    def accepted(self) -> bool:
        if not self.complete:
            return False
        return self.passed / len(self.attempts) >= REQUIRED_PASS_RATE[self.scenario]

    def to_markdown(self) -> str:
        lines = [
            f"### Scenario: {self.scenario}",
            "",
            "| Attempt | Vouch confirmed | Pending created | Thumbs-up ignored | Frames | FPS "
            "| Result |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
        for attempt in self.attempts:
            confirmed = "yes" if attempt.vouch_confirmed else "no"
            lines.append(
                f"| {attempt.number} | {confirmed} | {attempt.pending_created} "
                f"| {attempt.gestures_ignored} | {attempt.frames} | {attempt.fps:.1f} "
                f"| {attempt.verdict} |"
            )

        required = REQUIRED_PASS_RATE[self.scenario]
        rule = "every attempt must pass" if required >= 1.0 else f"{required:.0%} must pass"
        outcome = "accepted" if self.accepted else "not accepted"
        lines += [
            "",
            f"Passed {self.passed} of {len(self.attempts)} attempts: {outcome} ({rule}). "
            f"Mean frame rate: {self.mean_fps:.1f} FPS.",
        ]
        if self.void:
            lines += [
                "",
                f"Void attempts: {self.void}. No thumbs-up was detected in them, so they "
                "tested nothing. Repeat the trial with the gesture held in clear view.",
            ]
        if not self.complete:
            lines += [
                "",
                f"The trial is incomplete: {self.requested} attempts were requested and "
                f"{len(self.attempts)} ran to the end.",
            ]
        return "\n".join(lines)


def run_trial(
    frames: Frames,
    *,
    scenario: str,
    attempts: int,
    seconds: float,
    store: SqliteStore,
    config: EngineConfig,
    faces: FaceAnalyzer,
    gestures: GestureDetector,
    workdir: Path,
    display: Display | None = None,
    pause_seconds: float = 5.0,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> TrialReport:
    """Run ``attempts`` timed windows of ``seconds`` each over one camera stream.

    Each attempt works on a scratch copy of the database with the enrolled
    people but no pending visitors, so attempts do not influence each other and
    the real database is never touched.

    An attempt counts only if its window ran to the end. When the frames run
    out or the user stops, the unfinished attempt is dropped and the report is
    marked incomplete.
    """
    if scenario not in SCENARIOS:
        raise DoorVouchError(
            f"Unknown scenario {scenario!r}. Choose one of: {', '.join(SCENARIOS)}."
        )
    if attempts < 1 or seconds <= 0:
        raise DoorVouchError("A trial needs at least one attempt and a positive duration.")

    stream = iter(frames)
    results: list[Attempt] = []
    for number in range(1, attempts + 1):
        caption = f"Attempt {number}/{attempts} starts in"
        if not _countdown(stream, display, pause_seconds, caption):
            break
        if display is not None:
            display.announce(f"Attempt {number}/{attempts}: go")
        scratch = workdir / f"attempt-{number:02d}.db"
        scratch.unlink(missing_ok=True)
        store.backup_to(scratch)
        with SqliteStore(scratch) as copy:
            for pending in copy.list_pending():
                copy.discard_pending(pending.id)
            engine = VouchingEngine(copy, config)
            stats = run_live(
                stream,
                faces=faces,
                gestures=gestures,
                engine=engine,
                display=display,
                max_seconds=seconds,
            )
        if not stats.timed_out:
            break
        attempt = Attempt(
            number=number,
            verdict=judge(scenario, stats.events),
            vouch_confirmed=any(stats.events.get(kind, 0) for kind in _VOUCH_EVENTS),
            pending_created=stats.events.get("pending_created", 0),
            gestures_ignored=stats.events.get("gesture_ignored", 0),
            frames=stats.frames,
            fps=stats.fps,
        )
        results.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
    return TrialReport(scenario, tuple(results), attempts)


def _countdown(
    stream: Iterator[tuple[float, Frame]],
    display: Display | None,
    seconds: float,
    caption: str,
) -> bool:
    """Let people get in position. Returns ``False`` if the stream ends or the user stops."""
    if seconds <= 0:
        return True
    first: float | None = None
    for timestamp, frame in stream:
        if first is None:
            first = timestamp
        remaining = seconds - (timestamp - first)
        if remaining <= 0:
            return True
        if display is not None and not display.show_faces(
            frame, [], [f"{caption} {math.ceil(remaining)} s"]
        ):
            return False
    return False
