# 3. The rule and the storage use only the standard library

Status: accepted (V0)

## Context

The part of DoorVouch that must be right is the decision logic: who counts as known,
whose gesture counts, when a visitor is stored. The parts around it are heavy and
awkward to test: OpenCV and MediaPipe are large, platform-specific packages, and their
output depends on a camera, on light and on the people in front of it.

If the logic imported those packages, every test would need them installed, a failing
test could come from a model as easily as from the rule, and simple commands such as
listing the enrolled people would not start on a machine where the installation is
incomplete.

## Decision

- `doorvouch.core` and `doorvouch.storage` import nothing outside the Python standard
  library. An embedding is a tuple of floats, cosine similarity is a few lines of
  Python, and the database is `sqlite3`.
- The adapters in `doorvouch.vision` translate OpenCV and MediaPipe results into the
  plain data classes of `core/observations.py`. They import the heavy packages inside
  the functions that use them, never at import time.
- `tests/test_architecture.py` enforces both points: it parses every module for its
  imports, and it imports the whole package in a subprocess where OpenCV, MediaPipe and
  NumPy are blocked.

## Consequences

- The tests for the rule, the storage and the command line run in well under a second,
  on any machine, with synthetic embeddings instead of pictures of people.
- Continuous integration has a job that installs only pytest. It proves the logic does
  not quietly start depending on a heavy package.
- Matching in pure Python is slower than with NumPy. Measured on the development
  machine with 128-dimensional embeddings and three faces in view: about 2 ms per frame
  for 5 people with 8 samples each, 10 ms for 20 people and 50 ms for 100 people. That
  is fine for a household and too slow for a large gallery. When it starts to matter,
  `Gallery` can keep its interface and compute with NumPy or a vector index inside.
- Lazy imports mean that a missing package shows up when a camera command runs, not
  when the program starts. `doorvouch doctor` exists to make that failure clear.
