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
    return np.ndarray.flatten(mfcc)[:1000]


def init_extraction(ddir):
    # all_ciri = list()
    all_ciri = np.empty((0,1001), int)
    for root,subdir,files in os.walk(ddir):
        for fi in files:
            print(f'reading {os.path.join(root,fi)}..')
            # print(os.path.join(root, fi))
            sound_data = extract_feature(os.path.join(root,fi))
            sound_data = np.append(sound_data, fi[:-4])
            # print(sound_data, sound_data.size)
            all_ciri = np.append(all_ciri, np.array([sound_data]), axis=0)
    # print(all_ciri, label)
    return all_ciri
            

        

def main():
    DATASET_DIR = 'dataset/'
    # read_mffc(DATASET_DIR)
    data_feature = init_extraction(DATASET_DIR)
    print(f'data feature:\n{data_feature}')



if __name__ == '__main__':
    main()
