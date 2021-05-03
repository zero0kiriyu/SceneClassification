import numpy as np
import pandas as pd
import cv2 as cv
from cyvlfeat.kmeans import kmeans
from cyvlfeat.sift import phow
from pandarallel import pandarallel
from scipy.spatial import distance
import os 
import pickle
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# %matplotlib inline
# %config Completer.use_jedi = False

def cal_descriptors(row): # calculate the descriptors
    features = []
    img = cv.imread(row.path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, descriptors = phow(gray.astype(np.float32),step=4)
    for feature in descriptors:
        features.append(feature)
    return features

def cal_cluster(cluster_num=500): # calculate the cluster
    
    pandarallel.initialize(nb_workers=50,use_memory_fs = False)
    train_list = pd.read_csv('train_list.csv')
    bag_of_features = []
    features = train_list.parallel_apply(cal_descriptors,axis=1)
    for f in features:
        bag_of_features += f
    clusters = kmeans(np.array(bag_of_features).astype('float32'),500, initialization="PLUSPLUS") #kmean cluster
    return clusters

def cal_phow_features(row,clusters):
    img = cv.imread(row.path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, descriptors = phow(gray.astype(np.float32),step=4)
    dist = distance.cdist(clusters, descriptors, metric='euclidean') #cal the distance to the each cluster centre
    idx = np.argmin(dist, axis=0) 
    hist, bin_edges = np.histogram(idx, bins=len(clusters))
    bow_vector = [float(i)/sum(hist) for i in hist] #(128,)
    return bow_vector

def cal_path2phow_features(clusters):
    pandarallel.initialize(nb_workers=70,use_memory_fs = False)
    
    train_list = None
    if not os.path.exists('train_phow_feature.pkl'):
        print('train not exist')
        train_list = pd.read_csv('train_list.csv')
        train_list['phow_feature'] = train_list.parallel_apply(cal_phow_features,axis=1,args=(clusters,))
        pickle.dump(train_list, open("train_phow_feature.pkl", "wb"))
    else:
        print('train exist')
        train_list = pickle.load(open("train_phow_feature.pkl", "rb"))
    
    
    test_list = None
    if not os.path.exists('test_phow_feature.pkl'):
        print('test not exist')
        test_list = pd.read_csv('test_list.csv')
        test_list['phow_feature'] = test_list.parallel_apply(cal_phow_features,axis=1,args=(clusters,))
        pickle.dump(test_list, open("test_phow_feature.pkl", "wb"))
    else:
        print('test exist')
        test_list = pickle.load(open("test_phow_feature.pkl", "rb"))
    
    df = pd.concat([train_list[['path','phow_feature']],test_list[['path','phow_feature']]],ignore_index=True)
    #df = train_list[['path','phow_feature']]
    path2phow_features = df.set_index('path').T.to_dict('list')
    return path2phow_features
    
    
    