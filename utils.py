"""Module with utility printing functions."""

LOG_FILE = None


def init_logging(fp):
    """Initialized file for logging output.

    Args:
        fp (pathlib.Path): Path to the log file.

    """
    global LOG_FILE
    LOG_FILE = open(fp, 'w')


def close_logging():
    """Closes logging file."""
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()


def my_print(*args, **kwargs):
    """Overloaded `print` function that also writes to the log file, if it
    is initialized.
    """
    global LOG_FILE
    print(*args, **kwargs)
    if LOG_FILE is not None:
        print(*args, **kwargs, file=LOG_FILE)
