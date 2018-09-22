import numpy as np
import os
from pydub import AudioSegment
import librosa
from tqdm import tqdm

class AudioProccessor:
    np.set_printoptions(threshold=np.nan)
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

    def to_categorical(this, vector_y): #lookup table
        y = np.empty([len(vector_y), 7])
        #categorical_vec = np.empty([len(vector_y),7], dtype=int)
        #["Nava", "Mahour", "Shour", "Chahargah", "Homayoun", "Segah", "Rastgepanjgah"]
        #y = np.empty(len(vector_y))

        for x , a in enumerate(vector_y):
            categorical_vec = [0,0,0,0,0,0,0]
            
            if vector_y[x]=="Nava":
                categorical_vec[0]=1
            if vector_y[x]=="Mahour":
                categorical_vec[1]=1
            if vector_y[x]=="Shour":
                categorical_vec[2]=1
            if vector_y[x]=="Chahargah":
                categorical_vec[3]=1
            if vector_y[x]=="Homayoun":
                categorical_vec[4]=1
            if vector_y[x]=="Segah":
                categorical_vec[5]=1
            if vector_y[x]=="Rastepanjgah":
                categorical_vec[6]=1
            
            y =  np.vstack((y, categorical_vec))

        #print y
        return y


    def set_X_Y(this, DASTGAH):
        AudioProccessor().split_group(DASTGAH+"_Raw", DASTGAH+ "_Splitted")
        DASTGAH_X = AudioProccessor().save_data_to_array(DASTGAH+'_Splitted')
        DASTGAH_Y = len(DASTGAH_X) *[DASTGAH]
        return DASTGAH_X, DASTGAH_Y


Chahargah_X, Chahargah_Y = AudioProccessor().set_X_Y("CHAHARGAH")
Nava_X, Nava_Y = AudioProccessor().set_X_Y("NAVA")

Y = Chahargah_Y + Nava_Y
Y_new = AudioProccessor().to_categorical(Y)
X = np.vstack((Chahargah_X, Nava_X))

print Y_new
