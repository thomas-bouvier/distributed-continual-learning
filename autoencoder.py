from modules import ContinualLearner, MemoryBuffer

class AutoEncoder(ContinualLearner, MemoryBuffer):
    '''Model for reconstructing images, "enriched" as ContinualLearner- and MemoryBuffer-object.'''

    def __init__():
        super().__init__()
    

    def train_one_step(self, x, y, meters):
        pass