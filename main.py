import os
import pickle
import torch
from preprocess import preprocess
from phow import cal_cluster,cal_path2phow_features
from torch.utils.data import DataLoader
from dataset import CourseworkDataset
from model import ClassifierNet
from torchbearer import Trial
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.optim as optim
import torch.nn as nn
from torchbearer import deep_to
import torchbearer



def custom_loader(state):
    img,phow_feature, label = deep_to(next(state[torchbearer.ITERATOR]), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
    batch_size = img.shape[0]
    tmp = phow_feature.resize_(batch_size,1,224,224)
    tmp2 = torch.cat([img,tmp],1)
    
    state[torchbearer.X], state[torchbearer.Y_TRUE] = tmp2, label
    
if __name__ == "__main__":
    # check if list file exist
    print('checking the datalist')
    if not os.path.exists('train_list.csv') or not os.path.exists('test_list.csv'):
        #do preprocess again
        print('datalist not exist')
        preprocess()
    else:
        print('datalist exist, skip preprocess')
    
    # calculate the phow cluster
    clusters = None
    print('checking the phow clusters')
    if not os.path.exists('clusters.pkl'):
        print('clusters not exist')
        clusters = cal_cluster()
        pickle.dump(clusters, open("clusters.pkl", "wb"))
    else:
        print('clusters exist')
        clusters = pickle.load(open("clusters.pkl", "rb"))
    
    # pre-calulate the phow features
    path2phow_features = {}
    print('checking the phow features')
    if not os.path.exists('path2phow_features.pkl'):
        print('features not exist')
        path2phow_features = cal_path2phow_features(clusters)
        pickle.dump(path2phow_features, open("path2phow_features.pkl", "wb"))
    else:
        print('features exist')
        path2phow_features = pickle.load(open("path2phow_features.pkl", "rb"))
        
    # define data augmentation
    tfms = A.Compose(
        [
            A.Resize(224,224,always_apply=True),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # load dataset
    trainset = CourseworkDataset(tfms,'train',path2phow_features)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    valset = CourseworkDataset(tfms,'test',path2phow_features)
    valloader = DataLoader(valset, batch_size=256, shuffle=True)
    testset = CourseworkDataset(tfms,'predict',path2phow_features)
    testloader = DataLoader(testset, batch_size=256, shuffle=True)
    
    # define model and loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierNet().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss = nn.CrossEntropyLoss()


    # begin to train
    trial = Trial(model, optimizer, loss, metrics=['acc', 'loss']).to(device)
    trial.with_loader(custom_loader)
    trial.with_generators(train_generator=trainloader, val_generator=valloader,test_generator=testloader)
    history = trial.run(epochs=50, verbose=1)
