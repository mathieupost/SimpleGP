class MultiCaster(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def write(self, str):
        for f in self.file_list:
            f.write(str)
