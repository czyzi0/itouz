LOG_FILE = None


def init_logging(fp):
    global LOG_FILE
    LOG_FILE = open(fp, 'w')


def close_logging():
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()


def my_print(*args, **kwargs):
    global LOG_FILE
    print(*args, **kwargs)
    if LOG_FILE is not None:
        print(*args, **kwargs, file=LOG_FILE)
