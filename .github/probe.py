"""TEMPORARY: print what the real models see in sample photos. Removed before merging."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe

from doorvouch import models
from doorvouch.config import Settings
from doorvouch.core.association import find_owner
from doorvouch.core.embeddings import cosine_similarity
from doorvouch.core.engine import VouchingEngine
from doorvouch.core.geometry import distance
from doorvouch.core.observations import FrameObservation
from doorvouch.storage import SqliteStore
from doorvouch.vision.mediapipe_gestures import MediaPipeGestureDetector
from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer, face_jpeg

paths = models.require(Settings.load().models_dir)
faces = OpenCvFaceAnalyzer.from_models(paths["face_detector"], paths["face_recognizer"])
print(f"opencv {cv2.__version__} mediapipe {mediapipe.__version__} python {sys.version.split()[0]}")

first_faces = {}
for image_path in sorted(Path(sys.argv[1]).glob("*.jpg")):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"{image_path.name}: could not be read ({image_path.stat().st_size} bytes)")
        continue
    height, width = frame.shape[:2]
    found = faces.analyze(frame)
    gestures = MediaPipeGestureDetector.from_model(paths["gesture_recognizer"])
    try:
        hands = []
        for step in range(3):
            hands = gestures.detect(frame, 1.0 + step * 0.1)
    finally:
        gestures.close()
    print(f"{image_path.name} {width}x{height}: {len(found)} face(s), {len(hands)} hand(s)")
    for index, face in enumerate(found):
        box = face.box
        jpeg = face_jpeg(frame, box)
        print(
            f"  face {index}: x={box.x:.0f} y={box.y:.0f} w={box.w:.0f} h={box.h:.0f} "
            f"score={face.score:.2f} dim={len(face.embedding)} jpeg={len(jpeg or b'')}"
        )
    for index, hand in enumerate(hands):
        owner = find_owner(hand.center, [face.box for face in found])
        reach = [round(distance(hand.center, face.box.center) / face.box.w, 2) for face in found]
        print(
            f"  hand {index}: centre=({hand.center[0]:.0f},{hand.center[1]:.0f}) "
            f"gesture={hand.gesture} score={hand.score:.2f} owner={owner} reach={reach}"
        )
    if found:
        first_faces[image_path.name] = found[0].embedding
    if found and hands:
        with SqliteStore() as store:
            admin = store.add_person("Sample", is_admin=True)
            store.add_face_sample(admin.id, found[0].embedding)
            engine = VouchingEngine(store)
            kinds = []
            gestures = MediaPipeGestureDetector.from_model(paths["gesture_recognizer"])
            try:
                for step in range(8):
                    moment = 10.0 + step * 0.1
                    observation = FrameObservation(
                        moment,
                        tuple(faces.analyze(frame)),
                        tuple(gestures.detect(frame, moment)),
                    )
                    kinds += [event.kind for event in engine.process(observation).events]
            finally:
                gestures.close()
            print(f"  engine, face 0 as admin, 8 frames: {kinds}")

names = sorted(first_faces)
for position, one in enumerate(names):
    for other in names[position + 1 :]:
        similarity = cosine_similarity(first_faces[one], first_faces[other])
        print(f"similarity {one} vs {other}: {similarity:.3f}")
