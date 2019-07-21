import os
import numpy as np
from tacotron2.layers import TacotronSTFT
import torch
from scipy.io.wavfile import read

class ConverterClass():
    def __init__(self,wav_folder,model_input_folder,model_output_folder):
        self.wav_folder = wav_folder
        self.wav_file_list = os.listdir(wav_folder)
        self.model_input_folder = model_input_folder
        self.model_output_folder = model_output_folder

        #Converter Settings
        self.overlap = 10
        self.sampling_rate = 22050
        self.win_length = 1024
        self.hop_length = 256
        self.filter_length=1024
        self.MAX_WAV_VALUE = 32768.0
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0
        self.stft = TacotronSTFT(filter_length=self.filter_length,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            sampling_rate=self.sampling_rate,
                            mel_fmin=self.mel_fmin,
                            mel_fmax=self.mel_fmax)
        self.iterate_files()

    def iterate_files(self):
        print('File Number:')
        for i,name in enumerate(self.wav_file_list):
            input_file_path= self.model_input_folder + str(i) + '_' + name + '.npy'
            output_file_path= self.model_output_folder + str(i) + '_' + name + '.npy'
            input, output = self.loadMelAndStft(self.wav_folder + name)
            np.save(input_file_path,input)
            np.save(output_file_path,output)
            if i%100 == 99:
                print(i)

    def loadMelAndStft(self, filename):
        audio = self.readAudio(filename)
        loadedMel = self.loadMel(audio)
        imag, real, magnitudes = self.stft.getOutput(audio)
        return self.convert_input_output(loadedMel, imag, real, magnitudes)

    def convert_input_output(self,loadedMel,imag,real,magnitudes):
        loadedMel = np.swapaxes(loadedMel, 0, 1)
        imag = np.swapaxes(imag[0], 0, 1)
        real = np.swapaxes(real[0], 0, 1)
        magnitudes = np.swapaxes(magnitudes[0], 0, 1)


        imag = np.asarray(imag, dtype=np.float32)
        real = np.asarray(real, dtype=np.float32)
        magnitudes = np.asarray(magnitudes, dtype=np.float32)
        input_mels_with_overlap = []
        output = []

        for element in range(loadedMel.shape[0]):
            if (element > self.overlap and element < loadedMel.shape[0] - self.overlap):
                mel_in_with_overlap = []
                for number in range(self.overlap * 2 + 1):
                    actual_mel_index = element - self.overlap + number
                    mel_in_with_overlap.append(loadedMel[actual_mel_index])
                mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
                input_mels_with_overlap.append(mel_in_with_overlap)
                output.append([imag[element], real[element], magnitudes[element]])
        return input_mels_with_overlap,output

    def readAudio(self, filename):
        # Read audio
        audio, sampling_rate = self.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        return audio_norm

    def loadMel(self, audio):
        mel = self.get_mel(audio)
        return mel.numpy()

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = read(full_path)
        return torch.from_numpy(data).float(), sampling_rate

    def get_mel(self, audio):
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)

        return melspec

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-wf', '--wav_folder', default="./inWav2/")
    parser.add_argument('-mel', '--model_input_folder', default="./mels-training2/")
    parser.add_argument('-o', '--model_output_folder', default="./output-training2/")

    args = parser.parse_args()

    state = ConverterClass(args.wav_folder,args.model_input_folder,args.model_output_folder)

