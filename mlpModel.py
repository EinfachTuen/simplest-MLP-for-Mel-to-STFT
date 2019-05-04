import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        self.linear1 = nn.Linear(input_dim, output_dim*10)
        self.linear2 = nn.Linear(output_dim*10, output_dim*10)
        self.linear3 = nn.Linear(output_dim*10, output_dim)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        out = f.relu(self.linear3(x))
        return out


def training(input, wanted):
    model = LinearRegressionModel(input.shape[0], wanted.shape[0]).cuda()
    criterion = nn.MSELoss()

    learning_rate = 0.00001
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    epochs = 1000

    for epoch in range(epochs):
        epoch += 1
        loss_list = []
        for step in range(input.shape[0]):
            step += 1
            value_in = input[:, step]
            value_res = wanted[:, step]

            np_ft32_in = np.array(value_in, dtype=np.float32)
            np_ft32_res = np.array(value_res, dtype=np.float32)
            input_var = Variable(torch.from_numpy(np_ft32_in)).cuda()
            wanted_var = Variable(torch.from_numpy(np_ft32_res)).cuda()
            optimizer.zero_grad()
            output_model = model(input_var)
            loss = criterion(output_model,wanted_var)
            loss.backward()
            loss_list.append(loss.data.cpu().numpy())
            optimizer.step()
            #print('step {}, loss {}'.format(step, loss.data))
        loss_np = np.array(loss_list)
        print('epoch {}, loss {}'.format(epoch, np.average(loss_np)))

    return model


