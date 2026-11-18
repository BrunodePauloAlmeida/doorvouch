# 2. Whose hand is it? Nearest face, measured in face widths, failing closed

Status: accepted (V0), to be revisited with measurements

## Context

A thumbs-up carries authority: it adds a person to the database, and in later versions
it cancels an alert. The gesture classifier reports hands and the face recognizer
reports faces, but nothing connects the two. The engine has to decide which face a hand
belongs to, and a wrong answer in one direction is much worse than in the other:

- Ignoring an admin's real thumbs-up costs a second attempt.
- Crediting somebody else's thumbs-up to an admin lets a stranger vouch for themselves.

## Options considered

1. **Nearest face in pixels.** Simple, but a person standing further back has a smaller
   face and a hand that is fewer pixels away from it, so the comparison between two
   people at different distances is unfair.
2. **Body pose.** A pose model links wrist, elbow, shoulder and head of the same
   person, which answers the question directly. It needs a third network per frame and
   more code, which is too much for a concept validation.
3. **Nearest face in units of face width, with a margin.** Dividing the distance by the
   width of each face makes it roughly independent of how far the person is from the
   camera. A margin turns close calls into "nobody".

## Decision

Option 3, implemented in `core/association.py`:

- The distance from the centre of the hand to the centre of each face is divided by
  the width of that face.
- A hand more than 4 face widths from every face belongs to nobody.
- If two faces are within reach, the nearest one owns the hand only when its distance
  is at most 0.8 times the distance of the runner-up. Otherwise the hand belongs to
  nobody and the gesture is ignored.

Both numbers are settings (`hand_max_reach`, `hand_ambiguity_ratio`).

## Consequences

- Two people standing shoulder to shoulder sometimes get their gesture ignored. The
  admin has to hold the hand in front of their own chest, or the two step apart.
- The rule can be fooled on purpose: a hand held directly in front of an admin's chest
  is attributed to the admin, whoever it belongs to. In V0 this needs the admin to be
  standing there, which is the situation vouching is meant for. Before the gesture is
  allowed to cancel alerts, this has to be closed.
- The `impostor` scenario of the acceptance trial measures how often a hand is
  attributed to the wrong face. If the measured rate is not zero in the shoulder to
  shoulder arrangement, option 2 moves from Phase 1 into the MVP.
