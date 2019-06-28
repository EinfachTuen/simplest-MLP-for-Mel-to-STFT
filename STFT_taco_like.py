from tacotron2.layers import TacotronSTFT
import torch
import numpy as np

class STFT_taco_like():
    def __init__(self):
        self.sampling_rate = 22050
        self.win_length = 1024
        self.hop_length = 256
        self.filter_length = 1024
        self.MAX_WAV_VALUE = 32768.0
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0
        self.stft = TacotronSTFT(filter_length=self.filter_length,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 sampling_rate=self.sampling_rate,
                                 mel_fmin=self.mel_fmin,
                                 mel_fmax=self.mel_fmax)

    def inverse_phase(self,mag,phase):
        return self.stft.stft_fn.inverse(mag.data,phase.data)

    def convertToAudio(self,imag_list,real_list,mag_list):
        imag_list = torch.cat(imag_list).transpose(1,2).cpu()
        real_list = torch.cat(real_list).transpose(1,2).cpu()
        mag_list = torch.cat(mag_list).transpose(1,2).cpu()
        phases = torch.autograd.Variable(torch.atan2(imag_list.data, real_list.data))
        return mag_list, phases