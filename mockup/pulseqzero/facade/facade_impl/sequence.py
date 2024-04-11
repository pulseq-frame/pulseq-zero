class Sequence:
    def __init__(self) -> None:
        self.blocks = []

    def add_block(self, *args):
        self.blocks.append(args)

    def check_timing(self):
        return True, []
    
    def write(self, file_name):
        pass

    def plot(self):
        pass
