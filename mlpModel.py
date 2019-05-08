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

def prepareInput(input,amount_timesteps_at_once):
    original_timesteps = input.shape[1]
    input_after_insertion = input
    for incrementSize in range(amount_timesteps_at_once):
        increaseSizeWithZeros = np.zeros(input.shape[0])
        input_after_insertion = np.insert(input_after_insertion, -1, increaseSizeWithZeros, axis=1)
        input_after_insertion = np.insert(input_after_insertion, 0, increaseSizeWithZeros, axis=1)
    #TODO: unfinished
    for element in range(original_timesteps):
        model_input_array = []
        for step in range(amount_timesteps_at_once):
            model_input_array = model_input_array.insert(input[:, element],-1)




def training(input, wanted):
    steps_before_and_after = 5
    input_asnumpy = np.array(input, dtype=np.float32)
    inputTimeLength = input_asnumpy.shape[1]
    preparedInput = prepareInput(input_asnumpy,steps_before_and_after)

    model = LinearRegressionModel(input.shape[0]*steps_before_and_after, wanted.shape[0]).cuda()
    criterion = nn.MSELoss()

    learning_rate = 0.00001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

   # input_reshape_test = np.reshape(input_asnumpy,(3,input.shape[0],-1))
   # print(input_reshape_test.shape)
    epochs = 1000000

    input = Variable(torch.from_numpy(preparedInput)).cuda()
    wanted = Variable(torch.from_numpy(np.array(wanted, dtype=np.float32))).cuda()

    for epoch in range(epochs):
        epoch += 1
        loss_list = []
        for step in range(inputTimeLength):
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
