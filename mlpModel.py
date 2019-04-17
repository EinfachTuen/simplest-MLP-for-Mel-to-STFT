import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def training(input, wanted):
    model = LinearRegressionModel(input.size, wanted.size)
    criterion = nn.MSELoss()

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    epochs = 100

    for epoch in range(epochs):
        epoch += 1
        input_var = Variable(torch.from_numpy(input))
        wanted_var = Variable(torch.from_numpy(wanted))
        optimizer.zero_grad()
        output_model = model(input_var)
        loss = criterion(output_model,wanted_var)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.data))
    return model


input = np.array([0,1,2,3,4,5],dtype= np.float32)
wanted = np.array([3,4,5,6,7,8],dtype= np.float32)

model = training(input,wanted)

predicted = model(Variable(torch.from_numpy(input))).data.numpy()
print(predicted)

