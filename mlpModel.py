import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        self.linear1 = nn.Linear(input_dim, output_dim * 10)
        self.linear2 = nn.Linear(output_dim * 10, output_dim * 10)
        self.linear3 = nn.Linear(output_dim * 10, output_dim)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        out = f.relu(self.linear3(x))
        return out


def generateSTFTFromMel(mel_input, model):
    mel_input = Variable(torch.from_numpy(np.array(mel_input, dtype=np.float32)))
    stft = []
    for step in range(mel_input.shape[1]):
        result = model(mel_input[:, step])
        step_result = result.detach().numpy()
        print(step_result.shape)
        stft.append(step_result)
    stft = np.array(stft)
    stft = np.swapaxes(stft,0,1)
    print(stft.shape)
    return stft


def training(input, wanted):
    model = LinearRegressionModel(input.shape[0], wanted.shape[0]).cuda()
    criterion = nn.MSELoss()

    learning_rate = 0.00001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    input_asNumpy = np.array(input, dtype=np.float32)
    input_reshape_test = np.reshape(input_asNumpy,(3,input.shape[1],-1))
    print(input_reshape_test.shape)
    epochs = 1000000
    input = Variable(torch.from_numpy(input_asNumpy)).cuda()
    wanted = Variable(torch.from_numpy(np.array(wanted, dtype=np.float32))).cuda()

    for epoch in range(epochs):
        epoch += 1
        loss_list = []
        for step in range(input.shape[1]):
            step += 1
            input_var = input[:, step]
            wanted_var = wanted[:, step]

            optimizer.zero_grad()
            output_model = model(input_var)

            loss = criterion(output_model, wanted_var)
            loss.backward()
            loss_list.append(loss.data.cpu().numpy())
            optimizer.step()
            # print('step {}, loss {}'.format(step, loss.data))
        loss_np = np.array(loss_list)
        print('epoch {}, loss {}'.format(epoch, np.average(loss_np)))
        if (epoch % 100) == 99:
            torch.save(model, "MLP1-" + str(epoch))
    return model
