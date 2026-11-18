"""Errors meant to be shown to the person running DoorVouch."""


class DoorVouchError(Exception):
    """A problem the user can act on. The CLI prints it without a traceback."""
