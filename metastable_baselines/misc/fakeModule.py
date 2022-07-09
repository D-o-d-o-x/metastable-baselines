import torch as th
from torch import nn


class FakeModule(nn.Module):
    """
    A torch.nn Module, that drops the input and returns a tensor given at initialization.
    Gradients can pass through this Module and affect the given tensor.
    """
    # In order to reduce the code required to allow suppor for contextual covariance and parametric covariance, we just channel the parametric covariance through such a FakeModule

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return self.tensor

    def string(self):
        return '<FakeModule: '+str(self.tensor)+'>'
