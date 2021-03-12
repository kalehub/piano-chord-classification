# mfcc and KNN for key classification
# guide on mfcc : https://www.youtube.com/watch?v=YCQa-AwO9kU
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.feature
import librosa.display
import glob
import os

# knn module
from knn import KNN

# librosa -> audio processing library

def read_mffc(ddir):
    '''read sound using .load method and returning the array(y) and sample rate (sr) '''
    y, sr = librosa.load(ddir)
    mfcc = librosa.feature.mfcc(y)
    print(mfcc, type(mfcc))

    # plotting
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(ddir)
    plt.tight_layout()
    plt.show()


def extract_feature(song):
    y,_ = librosa.load(song)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values betweeen(-1,1)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000]


def init_extraction(ddir):
    all_ciri = list()
    label = list()
    for root,subdir,files in os.walk(ddir):
        for fi in files:
            print(f'reading {os.path.join(root,fi)}..')
            # print(os.path.join(root, fi))
            sound_data = extract_feature(os.path.join(root,fi))
            # print(sound_data)
            # print(fi)
            all_ciri.append(sound_data)
            label.append(fi[:-4])
    print(all_ciri, label)
            

        

def main():
    DATASET_DIR = 'dataset/'
    # read_mffc(DATASET_DIR)
    init_extraction(DATASET_DIR)



if __name__ == '__main__':
    main()
