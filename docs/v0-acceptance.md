# V0 acceptance

Unit tests show that the rule is implemented as designed. They cannot show that a real
webcam, real faces and a real thumbs-up produce the observations the rule needs. This
page is the protocol for measuring that, and the place where results are recorded.

## Criteria

| Scenario | Criterion |
| :--- | :--- |
| `vouch` | At least 8 of 10 attempts create exactly one pending visitor. |
| `impostor` | In 10 of 10 attempts the thumbs-up is seen and ignored. |

An `impostor` attempt in which no thumbs-up was detected at all is *void*: it tested
nothing, so it does not count as a pass, and the trial has to be repeated. An attempt
whose time window did not run to the end (the camera stopped, or somebody pressed `q`)
is not counted, and the trial is reported as incomplete.

## Setup

1. `doorvouch doctor` reports no problem. Note the resolution and frame rate it prints.
2. One person is enrolled as an admin: `doorvouch enroll --name NAME --admin`.
3. A second person is available and is **not** enrolled.
4. The camera is at about head height, 1 to 2 metres from where people stand, with
   light on their faces rather than behind them.

## Scenario 1: vouch

```sh
doorvouch trial --scenario vouch --output docs/v0-acceptance.md
```

Before each attempt there is a 5-second countdown. In each 8-second attempt:

- the admin and the visitor stand side by side, both facing the camera;
- the admin holds a thumbs-up in front of their own chest for about two seconds.

Between attempts, both lower their hands.

## Scenario 2: impostor

```sh
doorvouch trial --scenario impostor --output docs/v0-acceptance.md
```

Run it in two arrangements, 10 attempts each:

- **A. Visitor alone.** Only the person who is not enrolled is in view and gives the
  thumbs-up.
- **B. Visitor next to the admin.** Both are in view, shoulder to shoulder. The visitor
  gives the thumbs-up with the hand that is closer to the admin. The admin keeps both
  hands down.

Arrangement B is the hard one: it checks that a hand is not attributed to the wrong
face.

## Reading the report

Each run appends a table like this one to the file given with `--output`:

| Column | Meaning |
| :--- | :--- |
| Vouch confirmed | Whether the engine confirmed an admin's thumbs-up in that attempt. |
| Pending created | How many pending visitors were created. |
| Thumbs-up ignored | How many times a thumbs-up was noticed and ignored. |
| Frames, FPS | How many frames were processed and how fast. |
| Result | `pass`, `fail` or `void`. |

The trial works on a scratch copy of the database. Your enrolled people, pending
visitors and event log are not changed by it.

## If a scenario fails

| Symptom | Likely cause | Try |
| :--- | :--- | :--- |
| `vouch` fails and "Vouch confirmed" is `no` | The thumbs-up was not seen for five frames in a row, or was not attributed to the admin. | Hold the hand at chest height in front of your own body; improve the light; check the frame rate. |
| `vouch` fails with "Vouch confirmed" `yes` and 0 created | The visitor's face was skipped as `uncertain`, too small or unclear. | Move closer; check `doorvouch events` for the `visitor_skipped` reason. |
| `vouch` creates 2 pending visitors | The visitor's face changed enough between two confirmations to look like two people. | Record it here: it is input for threshold calibration. |
| `impostor` fails | A hand was attributed to the admin's face. | Record the arrangement and distances here: it is input for the association rule. |
| Frame rate below 10 FPS | The computer or the camera mode is too slow for five consecutive frames to be quick. | Close other programs using the camera; try another USB port. |

## Results

No results have been recorded yet. Reports appended by `doorvouch trial --output` appear
below this line, together with the date of each run. When recording a run, add the
computer, the camera and the lighting.
