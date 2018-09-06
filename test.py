import numpy as np
import os
from pydub import AudioSegment
import librosa
from tqdm import tqdm
from keras.utils import np_utils
#from keras import utils as np_utils

class AudioProccessor:
    counter = 1

    def createDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    def mp3Split(self, source_file_path, destination_folder_path):
        audio = AudioSegment.from_mp3(source_file_path)
        l = len(audio)
        duration = 30000
        startFrom = 000
        listOfNewAudios = []
        #i = 1
        while startFrom + duration <= l:
            curFrame = audio[startFrom:startFrom + duration]
            startFrom = startFrom + duration
            listOfNewAudios.append(curFrame)
            curFrame.export(destination_folder_path + '/splitted' + str(AudioProccessor.counter) + '.wav', format="wav")
            AudioProccessor.counter = AudioProccessor.counter + 1
        return listOfNewAudios

    def wav2mfcc(this, file_path, max_len=1): 
        #sr = sample rate
        wave, sr = librosa.load(os.path.abspath(file_path), mono=True, sr=None)
        #print(wave.shape)
        #print(wave)
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

    def save_data_to_array(this, path, max_len=1):

        mfcc_vectors = []
        mfcc_vector_main = []

        wavfiles = [path + '/' + wavfile for wavfile in os.listdir(path + '/' )]
        #wavfiles = [path + '/' + wavfile for wavfile in os.listdir(path)]
        #for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(path)):
        for wavfile in wavfiles:
            mfcc = AudioProccessor().wav2mfcc(wavfile, max_len=max_len) 
            #print "mfcc", mfcc
            mfcc_vectors.append(mfcc)
            #print mfcc_vectors.shape
            #stack list>np
            
        mfcc_vector_main = np.vstack(mfcc_vectors)
        #print "shape: ", mfcc_vector_main.shape
        np.save(path + '.npy', mfcc_vector_main)
        return mfcc_vector_main
        #print mfcc_vector_main

    def split_group(this, DASTGAH_source_path, DASTGAH_destination_path):
        AudioProccessor().createDirectory(DASTGAH_destination_path)
        wavfiles = [DASTGAH_source_path + '/' + wavfile for wavfile in os.listdir(DASTGAH_source_path + '/' )]
        for wavfile in wavfiles:
            AudioProccessor().mp3Split(wavfile, DASTGAH_destination_path)

    def set_X_Y(x_vector, y_vector):
        AudioProccessor().save_data_to_array(x_vector)
        y = numpy.full(len(x_vector), "CHAHARGAH", dtype=None)


AudioProccessor().split_group("CHAHARGAH_Raw", "CHAHARGAH_Splitted")
ChahargahData = AudioProccessor().save_data_to_array('CHAHARGAH_Splitted')


np.set_printoptions(threshold=np.nan)
#print ChahargahData

CH_y = np.full(len(ChahargahData), "CHAHARGAH", dtype=None)
print "CHAHARGAH: ", ChahargahData.shape
print "CH_Y: ", CH_y.shape

AudioProccessor().split_group("NAVA_Raw", "NAVA_Splitted")
NavaData = AudioProccessor().save_data_to_array('NAVA_Splitted')
NA_y = np.full(len(NavaData), "NAVA", dtype=None)
print "NavaData: ", NavaData.shape
print "NA_Y: ", NA_y.shape


Y = np.append(CH_y, NA_y)
X = np.append(ChahargahData, NavaData)


print Y.shape
print X.shape

#Y = keras.to_categorical(Y, num_classes=None)
#Y = np_utils.to_categorical(Y, 10)
Y = np_utils.to_categorical(Y, 3)
print Y, Y.shape
#
#all_data = np.concatenate((ChahargahData, y[:, np.newaxis]), axis=1)
#print all_data, all_data.shape
