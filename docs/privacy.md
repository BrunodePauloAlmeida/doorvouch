# Privacy

DoorVouch recognises faces. A face embedding is biometric data, which many laws treat
as a sensitive category. This page explains what the software stores, how to delete it,
and what to think about before pointing a camera at people.

This is a description of the software, not legal advice.

## What is stored, and where

Everything is in the data folder on your own computer (`./data` unless you choose
another with `--home` or `DOORVOUCH_HOME`).

| Data | Where | Created when | Deleted when |
| :--- | :--- | :--- | :--- |
| Face embeddings of enrolled people | `doorvouch.db` | `doorvouch enroll`, or naming a pending visitor | `doorvouch people remove ID` |
| Embedding and snapshot of a pending visitor | `doorvouch.db`, `snapshots/` | An admin's thumbs-up vouches for an unknown face | The visitor is named (the snapshot goes, the embedding moves to the person) or discarded (both go) |
| Event log | `doorvouch.db` | Something happens: a person is seen, a vouch, an ignored gesture | Events that mention a person are deleted with that person |
| Model files | `models/` | `doorvouch models download` | You delete the folder |

An embedding is a list of numbers computed from a face. It is not a picture, and the
picture cannot be reconstructed exactly from it, but it can be used to recognise the
same person again. Treat it like the photo it came from.

## What is not stored

- **Unknown faces.** A face that matches nobody is analysed in memory and forgotten
  with the frame. Only an admin's deliberate gesture creates a record for a visitor.
- **Video.** V0 records nothing. The preview window shows the live picture and keeps
  none of it.
- **Anything outside your computer.** The only network connection DoorVouch itself
  makes is the model download, which sends nothing but the request for the files.

## Deleting data

```sh
doorvouch people remove ID      # a person, their samples and their events
doorvouch pending discard ID    # a pending visitor and the snapshot
```

To erase everything, delete the data folder.

There is no automatic retention period in V0. Pending visitors and events stay until
you remove them. Retention settings are planned for the MVP and Phase 2.

## Before you use it on other people

- **Tell them.** People who live in or visit the house should know that a camera
  recognises faces, what is kept and how to have it removed. A visible notice at the
  entrance is a good habit.
- **Ask before enrolling.** Enrolling somebody, or naming a pending visitor, stores
  their biometric data. Do it with their agreement.
- **Keep the camera on your own property.** Privacy laws usually leave purely personal
  or household activities alone: the GDPR does so in Article 2(2)(c) and Brazil's LGPD
  in Article 4, I. That exemption has limits. In *Ryneš* (C-212/13, 2014) the Court of
  Justice of the European Union held that a home camera which also covers public space
  is not a purely household activity. Aim the camera at your door, not at the pavement
  or at a neighbour.
- **Shared entrances are different.** In a building entrance, a workplace or a rented
  property, other rules apply and consent of the household is not enough.
- **Recognition makes mistakes.** Accuracy varies with lighting, angle, age and between
  demographic groups. Never treat a match, or the lack of one, as proof of who somebody
  is.

## Repository hygiene

- The data folder, databases and model files are listed in `.gitignore`.
- The tests use synthetic embeddings. No photo, video or embedding of a real person
  belongs in this repository, including in issues and pull requests.
