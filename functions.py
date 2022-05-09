# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler

from datetime import datetime as speedtime


SETTINGS = {'path' : r'Database\data',
            'algorithm' : "DBScan",
            'sample_num' : 100e3,
            'fillna' : 0,
            'folds' : 3,
            'batch_size' : 200,
            
            'n_components' : None,
            'eps' : 0.8,
            'min_samples' : 5,
            'leaf_size' : 30
            }


def load_settings():
    return SETTINGS



def load_data(SETTINGS):
    t1 = speedtime.now()
    
    folder = [f for f in listdir(SETTINGS['path']) if isdir(join(SETTINGS['path'], f))]
    classes_num = len(folder)
    L = int(SETTINGS['sample_num']/classes_num)

    data = pd.DataFrame()
    labels = pd.DataFrame()
    for i in folder:
        folder_path = join(SETTINGS['path'], i)
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        
        temp_data = pd.DataFrame()
        for j in files:
            loaded = load_csv(folder_path, [i, j])
            temp_data = pd.concat([temp_data, loaded])
            
            if temp_data.shape[0] >= L:
                temp_data = temp_data.iloc[:L-1]
                break
            
        data = pd.concat([data, temp_data])
    
    data, labels = data_label_sep(data)
    data = normalization(data)
    data, labels = idx_fixing(data, labels)
    data = PCA_apply(data, SETTINGS['n_components'])

    print("Data Loaded " + "   Duration: " + str(round((speedtime.now()-t1).total_seconds()/60, 2)) + " min")

    return data, labels


def load_csv(folder_path, iters):
    i, j = iters
    
    file_path = join(folder_path, j)
    loaded = pd.read_csv(file_path)
    loaded = data_cleaning(SETTINGS, loaded, file_path)
    loaded = label_correction(loaded, int(i))
    loaded = loaded.drop('timestamp', axis=1)
    
    return loaded


def label_correction(data, label):
    L = data.shape[0]
    data['class'] = label * np.ones(L)
    
    return data


def data_cleaning(SETTINGS, loaded, file_path):
    loaded = fill_na(SETTINGS, loaded, file_path)
    loaded.drop("T-JUS-CKGL", axis = 1, inplace = True)
    
    return loaded
    
    
def fill_na(SETTINGS, loaded, file_path):
    if SETTINGS['fillna'] == 0:
        loaded = loaded.fillna(SETTINGS['fillna'])
    elif SETTINGS['fillna'] == 'mean':
        loaded = loaded.fillna(loaded.mean())
    
    return loaded


def data_label_sep(data):
    labels = data.loc[:, data.columns == 'class']
    data = data.loc[:, data.columns != 'class']  

    return data, labels
    

def idx_fixing(data, labels):
    idx = np.array([i for i in range(data.shape[0])])     
    data.index = idx
    labels.index = idx
    
    return data, labels
    

def normalization(data):
    norm = StandardScaler()
    values = norm.fit_transform(data)
    cols = data.columns
    out = pd.DataFrame(values, columns = cols)
    
    return out
   
 

def PCA_apply(data, n_components):
    
    if type(n_components) == type(None):
        return data
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components)
    data = pd.DataFrame(pca.fit_transform(data))
    
    return data
    





def main_processing(SETTINGS, data, labels):
    t1 = speedtime.now()
    
    model = clustering(SETTINGS, data)
        
    print("Clustering ended.  Duration: {} min".format(str(round((speedtime.now()-t1).total_seconds()/60, 2))))

    return model


def clustering(SETTINGS, data):
    
    if SETTINGS['algorithm'].lower() == 'kmeans':
        from sklearn.cluster import MiniBatchKMeans as KMeans
        
        model = KMeans(n_clusters = 9)
        model = batch_training(SETTINGS, model, data)
    
    elif SETTINGS['algorithm'].lower() == 'dbscan':
        from sklearn.cluster import DBSCAN
        
        model = DBSCAN(eps = SETTINGS['eps'], min_samples = SETTINGS['min_samples'], leaf_size = SETTINGS['leaf_size'])
        model.fit(data)
    
    elif SETTINGS['algorithm'].lower() == 'spectral':
        from sklearn.cluster import SpectralClustering
        
        model = SpectralClustering(n_clusters = 9)
        model.fit(data)
    
    return model


def batch_training(SETTINGS, model, data):
    batch_size = SETTINGS['batch_size']
    L = data.shape[0]
    
    for i in range(0, int(L/batch_size)):
        model.partial_fit(data.iloc[i*batch_size:(i+1)*batch_size, :])
    
    if L % batch_size != 0:
        idx = int(L/batch_size)*batch_size
        model.partial_fit(data.iloc[idx:, :])
        
    return model






def evaluation(SETTINGS, model, data, labels):
    
    if SETTINGS['algorithm'].lower() == 'kmeans':
        accuracy = acc(labels, model.predict(data))
    else:
        accuracy = acc(labels, model.labels_)
    
    print("Accuracy: {:.2f}\n".format(100*accuracy))

    return accuracy * 100







def plot_by_label(x, y, labels):
    from matplotlib import pyplot as plt
    
    for i in np.unique(labels):
        plt.scatter(x[labels.squeeze() == i], y[labels.squeeze() == i], s = 2)
        plt.legend(str(i))
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



