"""What every minifier module shares."""


class MinifyError(Exception):
    """A minifier could not produce text; the message says why."""
