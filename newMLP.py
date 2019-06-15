import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from random import shuffle

from multiThreadDataset import AudioDataset
import gc

class StateClass():
    def __init__(self,first_hidden_layer_factor,second_hidden_layer_factor,trainingFolder):
        self.epochs = 100000
        self.learning_rate = 0.001
        self.model_input_size = 7 * 80
        self.model_output_size = 513
        self.last_loss = 999999
        self.single_dataloader = None
        self.first_hidden_layer_factor = first_hidden_layer_factor
        self.second_hidden_layer_factor = second_hidden_layer_factor
        self.model = LinearRegressionModel(self.model_input_size, self.model_output_size,first_hidden_layer_factor,second_hidden_layer_factor).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.modelname= "MLP-ADAM-MSE-10Hidden"
        self.model_to_load= ""
        self.model_storage =""
        self.single_file = "./inWav/16kLJ001-0001.wav"
        self.training_folder= trainingFolder
        self.file_iterations_per_loss = None
        self.iterations_per_save = None
        self.epochs_per_learning_change = 500
        self.result_filename = "result_audio"
        self.normalization_test_filename = "normalization_test_audio"
        self.sampling_rate = 22050
        self.lossfile = 'loss_log.txt'
        self.debug = False
        self.data = None

    def run_training(self):
        dataset = AudioDataset(self.training_folder)
        dataset.initialize()
        self.single_dataloader = DataLoader(dataset,
                                batch_size=500,
                                shuffle=True,pin_memory=False)
        Training(self)

    def do_inference(self):
        Test(self)

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
            nn.ReLU(),
            nn.Linear(second_hidden_layer_size, output_dim)
        )

    def forward(self, x):
        out = self.sequencial(x)
        return out

class Training():
    def __init__(self,state):
        iterations = 0
        loss_list = []
        for epoch in range(state.epochs):
            for i, (mel, stft,) in enumerate(state.single_dataloader):
                mel = mel.cuda()
                stft = stft.cuda()
                state.optimizer.zero_grad()
                output_model = state.model(mel)
                loss = state.criterion(output_model, stft)
                loss.backward()
                loss_list.append(loss.data.cpu().numpy())
                state.optimizer.step()

                if(iterations%200 == 0):
                    loss_np = np.array(loss_list)
                    average_loss = np.average(loss_np)
                    print('iteration {}, loss {}'.format(iterations, average_loss))
                    log_file = open(state.lossfile, 'a')
                    log_file.write(
                        state.modelname + str(iterations) + "," + "{:.4f}".format(np.average(loss_np)) + ',\n')
                    loss_list = []
                    gc.collect()
                if (iterations % state.iterations_per_save) == (state.iterations_per_save - 1):
                    print("===> model saved", state.model_storage + state.modelname + str(iterations))
                    torch.save(state.model.state_dict(), state.model_storage + state.modelname + str(iterations))
                iterations += 1

class Test():
    def __init__(self, state):
        self.createAudioFromAudio(state)

    def convertFileToMel(self,state):
        dataset = AudioDataset("")
        fileAsMelAndSTFT = dataset.loadMelAndStft(state.single_file)
        return fileAsMelAndSTFT

    def createAudioFromAudio(self, state):
        data = self.convertFileToMel(state)
        state.single_dataloader = DataLoader(data, batch_size=1, shuffle=False)
        GenerateAudioFromMel.load_and_inference_and_convert(None,state)


class GenerateAudioFromMel:
    def load_and_inference_and_convert(self, state):
        stft_list = []
        # Normalized STFT For Testing
        stft_list_for_testing = []
        state.model = LinearRegressionModel(state.model_input_size, state.model_output_size,state.first_hidden_layer_factor, state.second_hidden_layer_factor)
        print("load model", state.model_to_load)
        state.model.load_state_dict(torch.load(state.model_to_load))
        state.model.cuda()
        state.model.eval()

        for i, (mel,stft) in enumerate(state.single_dataloader):
            mel = mel.cuda()
            #Normalized STFT For Testing
            stft_list_for_testing.append(stft[0].numpy())
            stft_part = state.model(mel).cpu().detach().numpy()
            stft_list.append(stft_part[0])

        stft_list = np.asarray(stft_list)
        stft_list = np.swapaxes(stft_list, 0, 1)
        GenerateAudioFromMel.stft_to_audio(self,state,stft_list,"Result STFT",state.result_filename)

        #After Normalization for Testing
        stft_list_for_testing = np.swapaxes(stft_list_for_testing, 0, 1)
        print(stft_list.shape)
        print(stft_list_for_testing.shape)
        GenerateAudioFromMel.stft_to_audio(self,state,stft_list_for_testing,"Normalized STFT",state.result_filename)


    def stft_to_audio(self,state,stft,diagramm_name,filename):
        wav = librosa.istft(stft)
        librosa.output.write_wav(state.modelname+'_'+filename+diagramm_name+".wav",wav,state.sampling_rate)
        GenerateAudioFromMel.plotSTFT(self,stft,diagramm_name)


    def plotSTFT(self,stft,title):
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 2, 1)
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--training", dest='training', action='store_true')
    parser.add_argument('-d', "--debug", dest='debug', action='store_true')
    parser.add_argument('-i', '--inference', dest='inference', action='store_true')
    parser.add_argument('-tf', '--trainingFolder', default="./inWav/")
    parser.add_argument('-m', '--modelname', default="actual-Model")
    parser.add_argument('-sf', '--single_file', default="./reserveWav/LJ001-0001.wav")
    parser.add_argument('-ips', '--iterationsPerSave', default=10000,type=int)
    parser.add_argument('-lr', '--learningRate', default=0.001,type=float)
    parser.add_argument('-ms', '--modelStorage', default="")
    parser.add_argument('-c', '--modelCheckpoint', default="")
    parser.add_argument('-h1f','--firstHiddenlayer', default=5,type=int)
    parser.add_argument('-h2f','--secondHiddenlayer', default=5,type=int)

    parser.set_defaults(debug=False)
    parser.set_defaults(training=False)
    parser.set_defaults(inference=False)
    args = parser.parse_args()

    state = StateClass(args.firstHiddenlayer,args.secondHiddenlayer,args.trainingFolder)
    print('Model Storage', args.modelStorage)
    state.modelname= args.modelname
    state.single_file= args.single_file
    state.iterations_per_save= args.iterationsPerSave
    state.learningRate= args.learningRate
    state.lossfile= "loss_"+args.modelname+".txt"
    state.model_storage= args.modelStorage
    state.debug= args.debug
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
