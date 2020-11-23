from torch import nn


class ModelThrowAwayLogDet(nn.Module):
    def __init__(self, model):
        super(ModelThrowAwayLogDet, self).__init__()
        self.model = model

    def forward(self, x):
        x, logdet = self.model(x)
        return x

    def invert(self, y):
        x, logdet = self.model.invert(y)
        return x

#Alias
NoLogDet = ModelThrowAwayLogDet