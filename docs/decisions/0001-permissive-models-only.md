# 1. Only models with permissive licences

Status: accepted (V0)

## Context

DoorVouch is published under the MIT licence so that anybody can use it, change it and
build on it, including commercially. That promise is only real if the models it needs
allow the same. A model licence is separate from the licence of the code that runs it,
and some of the most popular choices for this kind of project are more restrictive
than they look:

- **InsightFace.** The library is MIT, but its pretrained face recognition models are
  released for non-commercial research purposes only.
- **Ultralytics YOLO.** The code and models are AGPL-3.0 unless a commercial licence is
  bought, which would put obligations on everything that links to them.

## Decision

DoorVouch uses only models whose licence permits commercial use and redistribution
without copyleft obligations. For V0 these are:

| Purpose | Model | Licence |
| :--- | :--- | :--- |
| Face detection | YuNet, from the OpenCV Zoo | MIT |
| Face embeddings | SFace, from the OpenCV Zoo | Apache-2.0 |
| Hands and gestures | MediaPipe Gesture Recognizer | Published by Google with the Apache-2.0 MediaPipe framework; see its model card |

Model files are never committed. `doorvouch models download` fetches them from their
publishers and checks size and SHA-256 against values pinned in `doorvouch/models.py`.
The MediaPipe bundle is published under a "latest" address, so its hash cannot be
pinned; it is checked for size and file type instead.

## Consequences

- SFace is small and fast, but less accurate than the strongest recognition models.
  That is acceptable for a household-sized gallery and is one reason the engine has an
  `uncertain` zone between "match" and "new person".
- The recognizer sits behind the `FaceAnalyzer` protocol, so a better permissive model
  can replace it without touching the rule. Candidates noted during research: AuraFace
  (Apache-2.0) for recognition, RF-DETR (Apache-2.0) for person detection and the
  MiniFASNet anti-spoofing models (Apache-2.0) for liveness.
- Every model proposed in the future needs its licence checked and recorded in the
  README table before it is added.
