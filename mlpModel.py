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
        self.linear1 = nn.Linear(input_dim, input_dim * 10)
        self.linear2 = nn.Linear(input_dim * 10, input_dim * 10)
        self.linear3 = nn.Linear(input_dim * 10, output_dim)

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
        stft.append(step_result)
    stft = np.array(stft)
    stft = np.swapaxes(stft,0,1)
    print(stft.shape)
    return stft

def prepareInput(input,amount_timesteps_before_and_after):
    input_after_insertion = input
    for incrementSize in range(amount_timesteps_before_and_after):
        increaseSizeWithZeros = np.zeros(input.shape[0])
        input_after_insertion = np.insert(input_after_insertion, -1, increaseSizeWithZeros, axis=1)
        input_after_insertion = np.insert(input_after_insertion, 0, increaseSizeWithZeros, axis=1)
    #TODO: unfinished
    elements =[]
    timesteps_per_run = amount_timesteps_before_and_after * 2 + 1
    print("timesteps_per_run",timesteps_per_run)
    for element in range(input_after_insertion.shape[1]):
        model_input_array = []
        if element < (input_after_insertion.shape[1]-amount_timesteps_before_and_after*2):
            for step in range(timesteps_per_run):
                actual_position = element+step
                model_input_array.append(input_after_insertion[:, actual_position])
            elements.append(model_input_array)
    elements = np.asarray(elements)
    return Variable(torch.from_numpy(elements)).cuda()

def training(input, wanted):
    steps_before_and_after = 5
    input_asnumpy = np.array(input, dtype=np.float32)
    inputTimeLength = input_asnumpy.shape[1]
    preparedInput = prepareInput(input_asnumpy,steps_before_and_after)
    model = LinearRegressionModel(preparedInput.shape[1]*preparedInput.shape[2], wanted.shape[0]).cuda()
    criterion = nn.MSELoss()

    learning_rate = 0.0001
    last_loss =9999999
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

   # input_reshape_test = np.reshape(input_asnumpy,(3,input.shape[0],-1))
   # print(input_reshape_test.shape)
    epochs = 1000000
    wanted = Variable(torch.from_numpy(np.array(wanted, dtype=np.float32))).cuda()
    order_of_training_steps = np.arange(inputTimeLength)
    for epoch in range(epochs):
        epoch += 1
        loss_list = []
        for step in range(inputTimeLength):
            input_var = preparedInput[order_of_training_steps[step]].flatten()

           # print(input_var)
            wanted_var = wanted[:, [order_of_training_steps[step]]]

            optimizer.zero_grad()
            output_model = model(input_var)

            loss = criterion(output_model, wanted_var)
            loss.backward()
            loss_list.append(loss.data.cpu().numpy())
            optimizer.step()
            # print('step {}, loss {}'.format(step, loss.data))
        np.random.shuffle(order_of_training_steps)
        loss_np = np.array(loss_list)
        average_loss = np.average(loss_np)
        print('epoch {}, loss {}'.format(epoch,average_loss))

        name= "MLP1-mightier"
        log_file = open('loss_log.txt', 'a')
        log_file.write(name+str(epoch) + "," + "{:.4f}".format(np.average(loss_np)) + ',\n')
        if (epoch % 100) == 99:
            torch.save(model, name + str(epoch))
        if (epoch % 500) == 499:
            if average_loss >= last_loss :
                learning_rate *=-0.5
                print("learning_rate changed to"+learning_rate)
            last_loss = average_loss
    return model
