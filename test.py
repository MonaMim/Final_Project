import numpy as np
import os
from pydub import AudioSegment
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from keras import backend as K


class AudioProccessor:

    dastgah_to_index = {"NAVA":0, "MAHOUR":1, "SHOUR":2, "CHAHARGAH":3, "HOMAYOUN":4, "SEGAH":5, "RASTEPANJGAH":6}
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

    def wav2mfcc(this, file_path, max_len): 
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
            #print (mfcc.shape)
        
        return mfcc



    def save_data_to_array(this, path, max_len=11):

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


    def set_X_Y(this, DASTGAH):
        AudioProccessor().split_group(DASTGAH+"_Raw", DASTGAH+ "_Splitted")
        DASTGAH_X = AudioProccessor().save_data_to_array(DASTGAH+'_Splitted')
        DASTGAH_Y = len(DASTGAH_X) *[DASTGAH]
        return DASTGAH_X, DASTGAH_Y

    def to_categorical(this, vector_y): #lookup table
        y2 = np.zeros([len(vector_y), 7])
        for x , a in enumerate(vector_y):
            y2[x, AudioProccessor.dastgah_to_index[a]] = 1 
        return y2

    def get_train_test(this, split_ratio, random_state, X, Y):
        return train_test_split(X, Y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



#mfcc = AudioProccessor().wav2mfcc('Ahmad_Ebadi_CHAHARGAH.mp3')
#print (mfcc.shape)


Chahargah_X, Chahargah_Y = AudioProccessor().set_X_Y("CHAHARGAH")
Nava_X, Nava_Y = AudioProccessor().set_X_Y("NAVA")

#Y = Chahargah_Y + Nava_Y 
Y = np.append(Chahargah_Y, Nava_Y)
#print len(Nava_Y), len(Chahargah_Y), len(Y) 
X = np.vstack((Chahargah_X, Nava_X))
Y = AudioProccessor().to_categorical(Y)

#assert X.shape[0] == len(Y)

X_train, X_test, Y_train, Y_test = AudioProccessor().get_train_test(0.6, 42, X, Y)
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

print (X_train.shape)
print (X_train.size)
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(1,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

#mfcc = AudioProccessor().wav2mfcc("CHAHARGAH_Raw/Ahmad_Ebadi_CHAHARGAH.mp3", max_len=11) 
#print (mfcc)
#print (X_test.shape)
#print (X_train.shape)

    # Append all of the dataset into one single array, same goes for y

'''
