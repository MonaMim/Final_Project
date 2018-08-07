import numpy as np
import os
from pydub import AudioSegment
import librosa

class AudioProccessor:
    def createDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    def mp3Split(self, source_file_path, destination_folder_path):
        audio = AudioSegment.from_mp3(source_file_path)
        l = len(audio)
        duration = 30000
        startFrom = 000
        listOfNewAudios = []
        i = 1
        while startFrom + duration <= l:
            curFrame = audio[startFrom:startFrom + duration]
            startFrom = startFrom + duration
            listOfNewAudios.append(curFrame)
            curFrame.export(destination_folder_path + '/splitted' + str(i) + '.wav', format="wav")
            i = i + 1
        return listOfNewAudios

#    def wav2mfcc(self, source_path, max_pad_len=11):
#        fixpath = os.path.join(source_path, "splitted1.wav")
        #wave, sr = librosa.load(fixpath, mono=True, sr=None)
    def wav2mfcc(this, file_path, max_len=11): #def wav2mfcc(file_path, max_len=11):
            wave, sr = librosa.load(os.path.abspath(file_path), mono=True, sr=None)
            print(wave.shape)
            print(wave)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)

            # If maximum length exceeds mfcc lengths then pad the remaining ones
            if (max_len > mfcc.shape[1]):
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # Else cutoff the remaining parts
            else:
                mfcc = mfcc[:, :max_len]
            
            return mfcc

    # Input: Folder Path
    # Output: Tuple (Label, Indices of the labels)
    #def get_labels(self, path):
    #    labels[] = os.listdir(path)
    #    label_indices = np.arange(0, len(labels))
    #    return labels, label_indices

    def save_data_to_array(path, max_pad_len=11):
      #  labels, _ = AudioProccessor().get_labels(path)
        mfcc_vectors = []
        #for label in labels:
            # Init mfcc vectors
        mfcc_vector = []
           # wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        wavfiles = ['CHAHARGAH/' + wavfile for wavfile in os.listdir('CHAHARGAH/')]
        for wavfile in wavfiles:
            mfcc = AudioProccessor().wav2mfcc(wavfile)
            mfcc_vector.append(mfcc)
            np.save('CHAHARGAH' + '.npy', mfcc_vector)
            np.vstack((mfcc_vectors, mfcc_vector))
        return mfcc_vectors

AudioProccessor().createDirectory('CHAHARGAH')
AudioProccessor().mp3Split('Shadjarian_Overture_CHAHARGAH.mp3', 'CHAHARGAH')
#data = AudioProccessor().save_data_to_array('CHAHARGAH')
data = AudioProccessor().wav2mfcc('CHAHARGAH/splitted1.wav')
#print(data)
