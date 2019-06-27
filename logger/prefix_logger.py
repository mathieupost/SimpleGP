class PrefixLogger(object):
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = f"[{prefix}]"

    def write(self, msg):
        if msg == "\n":
            self.logger.write(msg)
        else:
            msg = msg.replace("\n", f"\n{self.prefix}")
            self.logger.write(f"{self.prefix} {msg}")

    def flush(self):
        self.logger.flush()
