import sys


class MultiLogger(object):
    def __init__(self, loggers):
        self.loggers = loggers

    def write(self, msg):
        for f in self.loggers:
            f.write(msg)

    def flush(self):
        for f in self.loggers:
            f.flush()

    def capture(self, func, *args, **kwargs):
        stdout = sys.stdout
        sys.stdout = self
        res = func(*args, **kwargs)
        sys.stdout = stdout
        return res
