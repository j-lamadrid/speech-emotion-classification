import opendatasets as od
import pandas as pd
import numpy as np
import shutil
import os
from IPython.display import Audio, display

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio
from torchvision import models

import transformers

import utils

import matplotlib.pyplot as plt

import warnings



def load_data(fp_head='speech-emotion-recognition-en'):
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # download from https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en
    crema_fp = fp_head + '/Crema'
    
    emotion_dict = {'HAP': 0,
                'NEU': 1,
                'SAD': 2,
                'DIS': 3,
                'ANG': 4,
                'FEA': 5}

    int_dict = {'XX': 0,
                'X': 0,
                'LO': 1,
                'MD': 2,
                'HI': 3}
    
    # download from https://github.com/CheyneyComputerScience/CREMA-D
    demo_df = pd.read_csv('speech-emotion-recognition-en/VideoDemographics.csv').set_index('ActorID')
    
    crema_lst = [f for f in os.listdir(crema_fp)]

    dataset = {}
    dataset['file'] = crema_lst
    dataset['label'] = [emotion_dict[f.split('_')[2]] for f in crema_lst]
    dataset['intensity'] = [int_dict[f.replace('.wav', '').split('_')[-1]] for f in crema_lst]
    dataset['sex'] = [demo_df.loc[int(f.split('_')[0])]['Sex'] for f in crema_lst]
    dataset['waveform'] = [torchaudio.load(crema_fp + '/' + f)[0] for f in crema_lst]
    dataset['spectrogram'] = [utils.load_spec(crema_fp + '/' + f)[0] for f in crema_lst]
    dataset['prediction'] = [float('nan')] * len(crema_lst)
    
    dataset['waveform'] = [F.pad(w, (0, 100000-w.size()[1])) for w in dataset['waveform']]

    dataset['mfcc'] = [torch.unsqueeze(torch.unsqueeze(torchaudio.compliance.kaldi.mfcc(w, num_ceps=40, num_mel_bins=60), 0), 0) for w in dataset['waveform']]

    dataset['mfcc_sum_freq'] = [torch.tensor(np.mean(m[0][0].numpy(), axis=1)) for m in dataset['mfcc']]

    dataset['mfcc_sum_time'] = [torch.tensor(np.mean(m[0][0].numpy(), axis=0)) for m in dataset['mfcc']]

    dataset['summary'] = [torch.cat((dataset['mfcc_sum_freq'][i], dataset['mfcc_sum_time'][i]), 0) for i in range(len(crema_lst))]
    
    m_idx = np.where(np.array(dataset['sex']) == 'Male')[0].astype(int)
    f_idx = np.where(np.array(dataset['sex']) == 'Female')[0].astype(int)

    m_set = {}
    f_set = {}

    for k, v in dataset.items():
        m_set[k] = list(np.array(dataset[k])[m_idx])

    for k, v in dataset.items():
        f_set[k] = list(np.array(dataset[k])[f_idx])
        
    print('Full Dataset Length: ' + str(len(dataset['spectrogram'])))
    print(' ')
    print('Male Only Dataset Length: ' + str(len(m_set['spectrogram'])))
    print('Female Only Dataset Length: ' + str(len(f_set['spectrogram'])))
    
    return dataset, m_set, f_set


def generate_samples(dic, variable, train=0.8, test=0.2):
    stop = False
    while not stop:
        X_train, X_test, y_train, y_test = train_test_split(dic[variable], dic['label'], train_size=train, test_size=test)
        if (len(np.unique(y_train)) == 6) and (len(np.unique(y_test)) == 6):
            stop = True
    return X_train, X_test, y_train, y_test


def get_accuracies(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []

    for i in range(len(X)):
        x = X[i].type(torch.FloatTensor).to(device)
        scores = model(x)
        _, out = scores.max(1)
        preds.append(out.item())
    
    preds = np.array(preds)
    labels = np.array(y)

    return np.mean(preds == labels), preds