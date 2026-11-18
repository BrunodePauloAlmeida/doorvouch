# Roadmap

DoorVouch grows in stages. Each stage has a goal, a list of tasks with an estimate in
hours of focused development, and an exit criterion that can be checked.

| Stage | Goal | Estimate |
| :--- | :--- | ---: |
| [V0](#v0-concept-validation) | One camera, one rule | 8 h |
| [MVP](#mvp-a-door-you-can-leave-alone) | Visits, away mode, one alert channel | 24 h |
| [Phase 1](#phase-1-hard-to-fool-hard-to-silence) | Robustness against tricks and tampering | 30 h |
| [Phase 2](#phase-2-a-product-for-a-household) | Dashboard, several cameras, integrations | 34 h |
| [Phase 3](#phase-3-an-open-platform) | Descriptions, acceleration, custom gestures, plugins | 36 h |
| | **Total** | **132 h** |

V0 is this release. The 124 hours after it are still ahead.

## The idea in one paragraph

A camera watches the main entrance. Face recognition keeps a record of who has come in.
When an admin arrives with somebody new, a thumbs-up to the camera vouches for the
visitor, who is pre-registered and named later. When nobody is home and an unknown
person appears, the system waits a few minutes for an admin's gesture; if none comes, it
tells the people the owner trusts and gives them access to what the camera saw. A hand
gesture is cheap to make, hard to make by accident and visible to the same camera that
sees the face, which makes it a useful second signal.

## V0: concept validation

Goal: prove that a webcam, two small face networks and a gesture classifier are enough
to apply the vouching rule reliably.

| Task | Estimate |
| :--- | ---: |
| Repository: licence, uv, ruff, mypy, pytest, CI | 1 h |
| Webcam capture with a frame-rate counter | 0.5 h |
| Face detection (YuNet), embeddings (SFace) and cosine matching | 2 h |
| `doorvouch enroll` | 1 h |
| Thumbs-up confirmed over consecutive frames (MediaPipe) | 1 h |
| Hand-to-face association and pre-registration in SQLite | 1.5 h |
| Acceptance trial that measures the rule on a real camera | 1 h |

Exit criterion: on a real camera, at least 8 of 10 vouch attempts create exactly one
pending visitor, and 0 of 10 thumbs-ups from somebody who is not an admin are credited
to an admin. The protocol is in [v0-acceptance.md](v0-acceptance.md).

## MVP: a door you can leave alone

Goal: the scenario that motivated the project. An unknown person arrives while the
owner is away, nobody vouches for them, and somebody is told.

| Task | Estimate |
| :--- | ---: |
| Swappable video sources and an event queue between capture and rule | 3 h |
| RTSP cameras with reconnection, and motion gating to save work | 3 h |
| Tracking: group the frames of one appearance into a *visit* | 4 h |
| Database: visits, trusted contacts and a retention period | 3 h |
| Away mode: a timer starts on an unknown person and an admin's gesture cancels it; a person whose face cannot be seen counts as unknown | 4 h |
| Telegram bot: alert with a photo, buttons, naming a visitor by replying | 4 h |
| Recorded clips replayed through the rule in CI | 3 h |

Exit criterion: a known person becomes a named visit within 5 seconds; an unknown
person in away mode triggers an alert with a photo; the admin's gesture cancels it; and
7 days at a real door produce at most one false alarm per day.

## Phase 1: hard to fool, hard to silence

Goal: the system keeps its promise when somebody works against it.

| Task | Estimate |
| :--- | ---: |
| Liveness check, so a photo of an admin cannot vouch | 4 h |
| Gesture password (a short sequence) and a duress gesture that looks like a normal one | 4 h |
| Clips with 10 seconds of pre-roll around each event | 4 h |
| Escalation from the owner to trusted contacts, with acknowledgement and expiring links | 6 h |
| Heartbeat to an external watchdog, and an immediate first photo, for when power or network is cut | 4 h |
| Camera tamper detection: covered, moved or blinded | 4 h |
| Threshold calibration and a false-alarm report | 4 h |

Exit criterion: a printed photo and a replayed video of an admin both fail to vouch;
cutting the power or the network leads to an alert from the watchdog within the
configured interval; covering the camera raises a tamper event.

## Phase 2: a product for a household

Goal: people who do not use a terminal can live with it.

| Task | Estimate |
| :--- | ---: |
| Web dashboard: people, visits, pending visitors, settings | 10 h |
| Several cameras, zones and privacy masks | 5 h |
| Expected visits, and the inverse rule: alert when somebody expected does *not* arrive | 5 h |
| Home Assistant integration over MQTT | 4 h |
| Docker Compose, images for ARM and x86, documentation site | 6 h |
| Retention controls, deleting a person everywhere, encrypted database | 4 h |

Exit criterion: a fresh install with Docker Compose reaches a working dashboard in
under 15 minutes, and every action of the command line is available in the dashboard.

## Phase 3: an open platform

Goal: others can extend DoorVouch, and it runs well on small hardware.

| Task | Estimate |
| :--- | ---: |
| Event descriptions written by a local vision-language model | 6 h |
| Acceleration with OpenVINO and Hailo, with a published benchmark | 8 h |
| Add-on mode for Frigate, reusing its cameras and detections | 6 h |
| Custom gestures trained on the HaGRID dataset | 6 h |
| Evaluation set built from consented clips | 6 h |
| Plugin API for rules and notification channels | 4 h |

Exit criterion: the benchmark shows real-time operation on a Raspberry Pi 5 with an
accelerator, and a third-party plugin can add a notification channel without changes
to the core.

## What is deliberately out of scope

- Unlocking doors. DoorVouch observes and notifies; it does not open anything.
- Identifying strangers against outside databases. A face is only ever compared with
  the people this household enrolled.
- Cloud processing of video.
