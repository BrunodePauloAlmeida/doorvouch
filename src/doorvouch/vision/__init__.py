"""Camera-facing code: the only part of DoorVouch that needs OpenCV and MediaPipe.

Those libraries are imported lazily, inside the functions that use them, so the
rest of the program (and its tests) works without them.
"""
