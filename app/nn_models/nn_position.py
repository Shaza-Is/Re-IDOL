from tensorflow.keras import Model, layers

class RePositionNet(Model):
    def __init__(self):
        super(RePositionNet, self).__init__()

    def call(self, x, is_training: bool = False):
        pass