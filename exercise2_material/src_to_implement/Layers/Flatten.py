from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.finalShape = input_tensor.shape
        return input_tensor.reshape(self.finalShape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.finalShape)