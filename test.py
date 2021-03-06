import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pydub import AudioSegment
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
import keras 
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model



class AudioProccessor:

    #dastgah_to_index = {"NAVA":0, "MAHOUR":1, "SHOUR":2, "CHAHARGAH":3, "HOMAYOUN":4, "SEGAH":5, "RASTEPANJGAH":6}
    dastgah_to_index = {"NAVA":0,"SEGAH":1, "MAHOUR": 2}
    
    np.set_printoptions(threshold=0) #np.nan
    counter = 1

    def createDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    def mp3Split(self, source_file_path, destination_folder_path):
        audio = AudioSegment.from_mp3(source_file_path)
        l = len(audio)
        #duration = 120000 #2mins
        duration = 20000 #20 secs
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
        #print (mfcc.shape)
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
            
        #mfcc_vector_main = np.vstack(mfcc_vectors)
        mfcc_vector_main = np.array(mfcc_vectors)
        #print "shape: ", mfcc_vector_main.shape
        np.save(path + '.npy', mfcc_vector_main)

        #print (mfcc_vector_main.shape, "this one")
        return mfcc_vector_main

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
        y2 = np.zeros([len(vector_y),2])
        for x , a in enumerate(vector_y):
            y2[x, AudioProccessor.dastgah_to_index[a]] = 1 
        return y2

    def get_train_test(this, split_ratio, random_state, X, Y):
        return train_test_split(X, Y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



#mfcc = AudioProccessor().wav2mfcc('Ahmad_Ebadi_CHAHARGAH.mp3')
#print (mfcc.shape)

'''
Chahargah_X, Chahargah_Y = AudioProccessor().set_X_Y("CHAHARGAH")
Homayoun_X, Homayoun_Y = AudioProccessor().set_X_Y("HOMAYOUN")
Rastepanjgah_X, Rastepanjgah_Y = AudioProccessor().set_X_Y("RASTEPANJGAH")
Shour_X, Shour_Y = AudioProccessor().set_X_Y("SHOUR")
'''
Nava_X, Nava_Y = AudioProccessor().set_X_Y("NAVA")
Segah_X, Segah_Y = AudioProccessor().set_X_Y("SEGAH")
#Mahour_X, Mahour_Y = AudioProccessor().set_X_Y("MAHOUR")
'''

Y = Chahargah_Y + Nava_Y 
Y = np.append(Chahargah_Y, Nava_Y)
Y= np.append(Y, Homayoun_Y)
Y= np.append(Y, Rastepanjgah_Y)
Y = np.append(Y, Segah_Y)
Y = np.append(Y, Mahour_Y)
Y = np.append(Y, Shour_Y)

#print len(Nava_Y), len(Chahargah_Y), len(Y) 
X = np.vstack((Chahargah_X, Nava_X))
X = np.vstack((X, Homayoun_X))
X = np.vstack((X, Rastepanjgah_X))
X = np.vstack((X, Segah_X))
X = np.vstack((X, Mahour_X))
X = np.vstack((X, Shour_X)) '''

#Y = Nava_Y + Segah_Y 
Y = np.append(Nava_Y, Segah_Y)
#Y = np.append(Y, Mahour_Y)
X = np.vstack((Nava_X, Segah_X))
#X = np.vstack((X, Mahour_X)) 

Y = AudioProccessor().to_categorical(Y)



#assert X.shape[0] == len(Y)

X_train, X_test, Y_train, Y_test = AudioProccessor().get_train_test(0.6, 42, X, Y)

#print (X_train.shape) 

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
#X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

#print (X_train.shape)
#print (X_train.size)
dim1, dim2 = X_train.shape[1], X_train.shape[2]

print (dim1)
print (dim2)

model = Sequential()
model.add(Conv2D(64 , kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(dim1, dim2, 1)))
#increase 
#32 * (3*3) 
#model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#max between 4 windows 
model.add(Flatten())
#model.add(Dense(64))
model.add(Dense(1024, activation='relu'))
#64 node
model.add(Dense(2, activation='softmax'))

#model.add(Dense(1))
#model.add(Activation("sigmoid"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(X_train.shape, "X_train")
print(Y_train.shape, "Y_train")
print(X_test.shape, "X_test")
print(Y_test.shape, "Y_test")

model.fit(X_train, Y_train,
          batch_size=32,
          validation_data=(X_test, Y_test),
          verbose= 1)
#increase to 128

print(model.evaluate(X_test, Y_test, verbose=0))

print(model.predict(X_test))

model.save('my_model.h5') 

del model # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
