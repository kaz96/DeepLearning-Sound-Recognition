



import tensorflow as tf
import matplotlib.pyplot as plt
import os
from os.path import isdir, join
from pathlib import Path
from tqdm import tqdm
import time

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
import IPython.display as ipd
from tensorflow.python.platform import gfile
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from sklearn.model_selection import train_test_split
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import LabelEncoder



#train_audio_path_test = '/Users/Kaz/Desktop/tst'
train_audio_path = '/Users/Kaz/Desktop/tst/audio_path/'

filename = '/left.wav'
new_sample_rate = 8000


#sample_rate, samples = wavfile.read(str(train_audio_path_test) + filename)
samples1, sample_rate1 = librosa.load(str(train_audio_path_test) + filename, res_type='kaiser_fast')

mfcc = librosa.feature.mfcc(samples1, sr=16000)
#mfccs = np.mean(librosa.feature.mfcc(y=samples1, sr="8000", n_mfcc=13).T, axis=0)

#print(sample_rate,samples)



# LOG SPECTOGRAM - NOT USED ANYMORE
#def log_specgram(audio, sample_rate, window_size=20,
#                 step_size=10, eps=1e-10):
#    #nperseg = int(round(window_size * sample_rate / 1e3))
#    #noverlap = int(round(step_size * sample_rate / 1e3))
#    freqs, times, spec = signal.spectrogram(audio,
#                                    fs=sample_rate,
#                                    window='hann',
#                                    nperseg=None,
#                                    noverlap=None,
#                                    detrend=False)
#  #  if ( spec. == 0 ):
#       # print("ZERO")
#
#
#    return freqs, times, np.log(spec)


#resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))


#freqs, times, spectrogram = log_specgram(samples, sample_rate)
#
#freqs1, times1, spectrogram1 = log_specgram(resampled, new_sample_rate)





#normalize

#mean = np.mean(spectrogram, axis=0)
#std = np.std(spectrogram, axis=0)
#spectrogram = (spectrogram - mean) / std



# get name of each of the labels, e.g. dog, eight, follow, foward, bird
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]

# soft in alphabetical order
dirs.sort()


print('Total Number of Labels in Directory: ' + str(len(dirs)))

number_of_recordings = []
# for loop to get every file in each folder
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    number_of_recordings.append(len(waves))



#Print histogram
data = [go.Histogram(x=dirs, y=number_of_recordings)]
trace = go.Bar(
    x=dirs,
    y=number_of_recordings,
    marker=dict(color = number_of_recordings, colorscale='Viridis', showscale=True
    ),
)

layout = go.Layout(
    title='Number of recordings in given label',
    xaxis = dict(title='Words'),
    yaxis = dict(title='Number of recordings')
)
#py.plot(go.Figure(data=[trace], layout=layout))



# IMPLEMENTATION
to_keep = 'down left up'.split()
dirs = [d for d in dirs if d in to_keep]

number=0
labels = []
labelsarray = []
trainingsetlabelarray = []

trainingset = []
print("Begin Reading Files")
#for each folder e.g. left
labelno = 0

getfromfile = False

if (getfromfile):
    for index,direct in enumerate(dirs):
        # Get individual file e.g. audio_path/left/032jd2.wav
        print("Processing Values For: ", direct, " [",index+1,"/",len(dirs),"]")
        waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
        no = 0
        for wav in tqdm(waves, total= len(waves)):

            wave, sr = librosa.load(train_audio_path + direct + '/' + wav, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            pad_width = 11 - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

           # y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

            trainingsetlabelarray.append(mfcc)
            labelsarray.append(direct)
            no += 1
        labelno += 1


    #translate to numpy array
    trainingset = np.array(trainingsetlabelarray)
    labels = np.array(labelsarray)

    #Save as a npy file
    np.save('trainingset.npy', trainingset)
    np.save('labels.npy', labels,)

print("LOAD FROM FILE")
#load from file
trainingset = np.load('trainingset.npy')
labels = np.load('labels.npy')

print("LOAD FROM FILE1")

#normalizaation
trainingset = (trainingset - np.mean(trainingset,axis=0)) / np.std(trainingset, axis=0)
y = np.zeros(trainingset.shape[0])


#Encode labels into vectors e.g. up = [1,0,0], down = [0,1,0]
lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(labels))


# Split into Train, and validation 60/40 split
X_train, X_test, y_train, y_test = train_test_split(trainingset, y, test_size= .40, random_state=44, shuffle=True)
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)










# Deep Learning Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=100, epochs=10, verbose=1, validation_data=(X_test, y_test))



# PREDICTION
predict = []

wave, sr = librosa.load("/Users/Kaz/Desktop/tst/up1.wav", mono=True, sr=None)
wave = wave[::3]
mfcc = librosa.feature.mfcc(wave, sr=16000)
pad_width = 11 - mfcc.shape[1]
mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
sample_reshaped = mfcc.reshape(1, 20, 11, 1)


predict.append(sample_reshaped)

output = model.predict(np.array(sample_reshaped))
model.evaluate()



print(model.summary())

print(output)

print(np.argmax(model.predict(np.array(sample_reshaped))))

output = np.argmax(model.predict(np.array(sample_reshaped)))
if(output == 0):
    print("OUTPUT: Down")
if(output == 1):
    print("OUTPUT: Up")
if(output == 2):
    print("OUTPUT: Left")
print("end")
