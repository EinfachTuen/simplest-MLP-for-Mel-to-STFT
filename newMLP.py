import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from tacotron2.layers import TacotronSTFT

from MultiThreadDataset2 import MultiThreadDataset2
from STFT_taco_like import STFT_taco_like

from TestDataset import TestDataset
import gc
from scipy.io.wavfile import write
import os

class StateClass():
    def __init__(self,first_hidden_layer_factor,second_hidden_layer_factor,trainingFolder,threads):
        self.epochs = 100000
        self.learning_rate = 0.001
        self.model_input_size = 21 * 80
        self.model_output_size = 513
        self.last_loss = 999999
        self.single_dataloader = None
        self.first_hidden_layer_factor = first_hidden_layer_factor
        self.second_hidden_layer_factor = second_hidden_layer_factor
        self.model = LinearRegressionModel(self.model_input_size, self.model_output_size,first_hidden_layer_factor,second_hidden_layer_factor).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()
        self.modelname= "MLP-ADAM-MSE-10Hidden"
        self.model_to_load= ""
        self.model_storage =""
        self.single_file = "./inWav/16kLJ001-0001.wav"
        self.test_folder = "./mels-for-test/"
        self.training_folder= trainingFolder
        self.file_iterations_per_loss = None
        self.iterations_per_save = None
        self.epochs_per_learning_change = 500
        self.result_filename = "result_audio"
        self.normalization_test_filename = "normalization_test_audio"
        self.sampling_rate = 22050
        self.lossfile = 'loss_log.txt'
        self.threads = 1
        self.debug = False
        self.data = None

    def run_training(self):
        dataset = MultiThreadDataset2('./mels-training/','./output-training/',self.threads)
        dataset.initialize()
        self.single_dataloader = DataLoader(dataset,
                                batch_size=1000,
                                shuffle=True)
        Training(self)

    def do_inference(self):
        test = Test(self)
        test.run_on_whole_folder(self)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim,first_hidden_layer_factor,second_hidden_layer_factor):
        super(LinearRegressionModel, self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        hidden_layer_size = input_dim * first_hidden_layer_factor
        second_hidden_layer_size = input_dim * second_hidden_layer_factor

        self.sequencial =nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, second_hidden_layer_size),
            nn.ReLU()
        )
        self.linearImag = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )

        self.linearReal = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )
        self.linearMag = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )

    def forward(self, x):
        intermediate = self.sequencial(x)
        imag = self.linearImag(intermediate)
        real = self.linearReal(intermediate)
        mag = self.linearMag(intermediate)
        #
        return imag,real,mag

class Training():
    def __init__(self,state):
        iterations = 0
        loss_list_imag = []
        loss_list_real = []
        loss_list_magnitudes = []
        loss_list_average = []
        for epoch in range(state.epochs):
            for i, (mel, imag,real,magnitudes) in enumerate(state.single_dataloader):
                mel = mel.cuda()
                imag = imag.cuda().float()
                real = real.cuda().float()

                magnitudes = magnitudes.cuda()
                state.optimizer.zero_grad()
                #
                imag_out,real_out,mag_out = state.model(mel)

                loss1 = state.criterion1(imag_out, imag)
                loss2 = state.criterion2(real_out, real)
                loss3 = state.criterion3(mag_out, magnitudes)
                #
                loss = loss1+loss2+loss3
                loss.backward()
                loss_list_imag.append(loss1.data.cpu().numpy())
                loss_list_real.append(loss2.data.cpu().numpy())
                loss_list_magnitudes.append(loss3.data.cpu().numpy())
                loss_list_average.append(loss.data.cpu().numpy())
                state.optimizer.step()

                if(iterations%100 == 0):
                    average_loss_imag = np.average(np.array(loss_list_imag))
                    average_loss_real = np.average(np.array(loss_list_real))

                    average_loss_magnitudes = np.average(np.array(loss_list_magnitudes))
                    average_loss = np.average(np.array(loss_list_average))
                    string ='i {}, imag: {}, real: {}, magnitudes: {}, average: {}'.format(iterations, average_loss_imag, average_loss_real, average_loss_magnitudes,average_loss)
                    print(string)
                    log_file = open(state.lossfile, 'a')
                    log_file.write(
                        state.modelname + string + ',\n')
                    loss_list_imag = []
                    loss_list_real = []
                    loss_list_magnitudes = []
                    loss_list_average = []
                    gc.collect()
                if (iterations % state.iterations_per_save) == (state.iterations_per_save - 1):
                    print("===> model saved", state.model_storage + state.modelname + str(iterations))
                    torch.save(state.model.state_dict(), state.model_storage + state.modelname + str(iterations))
                iterations += 1

class Test():
    def __init__(self, state):
        self.dataset = TestDataset(state.single_file)
        self.dataset.initialize()
        self.test_output_folder = "./test-result-wav/"
        self.generateAudioFromMel = GenerateAudioFromMel(state)

    def run_on_whole_folder(self,state):
        file_list = os.listdir(state.test_folder)
        for i, file_name in enumerate(file_list):
            file = state.test_folder + file_name
            self.dataset = TestDataset(file)
            self.dataset.initialize()
            self.createAudioFromAudio(state, file_name)

    def createAudioFromAudio(self,state,file_name):
        state.single_dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        wave = self.generateAudioFromMel.runModel(state)
        write(self.test_output_folder+file_name,22050,wave.numpy())


class GenerateAudioFromMel:
    def __init__(self, state):
        print("init")
        self.stft_converter = STFT_taco_like()
        state.model = LinearRegressionModel(state.model_input_size, state.model_output_size,
                                        state.first_hidden_layer_factor, state.second_hidden_layer_factor)
        state.model.load_state_dict(torch.load(state.model_to_load))
        state.model.cuda()
        state.model.eval()


    def runModel(self, state):
        imag_list = []
        real_list = []
        mag_list = []
        for i, mel in enumerate(state.single_dataloader):
            mel = mel.cuda()
            #Normalized STFT For Testing
            imag,real,mag = state.model(mel)
            imag_list.append(imag)
            real_list.append(real)
            mag_list.append(mag)

        mag_list,phases_res = self.stft_converter.convertToAudio(imag_list,real_list,mag_list)
        print(mag_list.shape)
        print(phases_res.shape)
        return self.stft_converter.inverse_phase(mag_list,phases_res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--training", dest='training', action='store_true')
    parser.add_argument('-d', "--debug", dest='debug', action='store_true')
    parser.add_argument('-i', '--inference', dest='inference', action='store_true')
    parser.add_argument('-tf', '--trainingFolder', default="./inWav/")
    parser.add_argument('-m', '--modelname', default="actual-Model")
    parser.add_argument('-sf', '--single_file', default="./mels-training/0_LJ001-0001.wav.npy")
    parser.add_argument('-ips', '--iterationsPerSave', default=10000,type=int)
    parser.add_argument('-lr', '--learningRate', default=0.001,type=float)
    parser.add_argument('-ms', '--modelStorage', default="")
    parser.add_argument('-c', '--modelCheckpoint', default="")
    parser.add_argument('-h1f','--firstHiddenlayer', default=5,type=int)
    parser.add_argument('-h2f','--secondHiddenlayer', default=5,type=int)
    parser.add_argument('-ts','--threads', default=1,type=int)
    parser.add_argument('-ti', '--test_input_folder', default="./mels-training/")

    parser.set_defaults(debug=False)
    parser.set_defaults(training=False)
    parser.set_defaults(inference=False)
    args = parser.parse_args()

    state = StateClass(args.firstHiddenlayer,args.secondHiddenlayer,args.trainingFolder,args.threads)
    print('Model Storage', args.modelStorage)
    state.modelname= args.modelname
    state.single_file= args.single_file
    state.iterations_per_save= args.iterationsPerSave
    state.learningRate= args.learningRate
    state.lossfile= "loss_"+args.modelname+".txt"
    state.model_storage= args.modelStorage
    state.debug= args.debug
    state.test_folder = args.test_input_folder

    if args.modelCheckpoint != "":
        if args.training:
            state.model.load_state_dict(torch.load(args.modelCheckpoint))
            state.model.cuda()
            state.model.eval()
        if args.inference:
            state.model_to_load = args.modelCheckpoint
    if args.training:
        state.run_training()
    if args.inference:
        state.do_inference()
