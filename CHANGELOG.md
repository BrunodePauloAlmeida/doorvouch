# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-18

First release: V0, the concept validation.

### Added

- The vouching rule: an admin's thumbs-up, held for five consecutive frames and
  attributed to the admin's own face, pre-registers every clearly unknown face in view
  as a pending visitor.
- Face detection with YuNet and face embeddings with SFace, through OpenCV.
- Thumbs-up recognition with MediaPipe's Gesture Recognizer.
- Hand-to-face association that refuses to guess when two people are equally close to
  the hand.
- SQLite storage for people, face samples, pending visitors and an event log.
- The `doorvouch` command: `doctor`, `models`, `enroll`, `run`, `pending`, `people`,
  `events` and `trial`.
- `doorvouch models download`, which verifies size and SHA-256 of the pinned models.
- `doorvouch trial`, the acceptance trial that measures the rule on a real camera.
- Tests that run the real models on public sample photos, on Linux and Windows.
- Documentation: architecture, roadmap, privacy notes, acceptance protocol and three
  decision records.

### Known limitations

- Thresholds are uncalibrated starting points.
- No liveness check, no alerts, no away mode, one local webcam only.
- Not yet measured in front of a real camera: the acceptance trial in
  `docs/v0-acceptance.md` is still to be run.

[0.1.0]: https://github.com/BrunodePauloAlmeida/doorvouch/releases/tag/v0.1.0
