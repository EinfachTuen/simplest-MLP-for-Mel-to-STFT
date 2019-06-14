from mel2samp import Mel2Samp
from torch.utils.data import DataLoader
import librosa
from tacotron2.audio_processing import griffin_lim
from tacotron2.layers import TacotronSTFT


trainset = Mel2Samp("train_files.txt", 16000, 1024,
                    256, 1024, 22050, 0.0, 8000.0)
d = DataLoader(trainset, batch_size=50, shuffle=False)
stft = []
stft_fn= None

tacotronSTFT = TacotronSTFT(filter_length=1024,
                         hop_length=256,
                         win_length=1024,
                         sampling_rate=22050,
                         mel_fmin=0.0, mel_fmax=8000.0)

for i, (mel,magnitudes,audio) in enumerate(d):
    stft = magnitudes
    print(mel)

stft = stft[0][0]
# stft = stft.numpy()
print(stft)

wav = griffin_lim(stft,tacotronSTFT.stft_fn, 60)
librosa.output.write_wav("test.wav",wav,16000)