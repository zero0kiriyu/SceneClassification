from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np

    

class CourseworkDataset(Dataset):
    """A Coursework dataset."""
    def __init__(self, transform,mode,path2phow_features):
 
        super(CourseworkDataset, self).__init__()
        self.transform = transform
        self.mode = mode        
        self.path2phow_features = path2phow_features
        if mode == "predict":
            predict_list = pd.read_csv('test_list.csv')
            self.images = predict_list['path'].tolist()
            self.labels = predict_list['predict_label'].tolist()
        else:
            train_list = pd.read_csv('train_list.csv')
            train, test = train_test_split(train_list, test_size=0.33, random_state=42)
            if mode == "train":
                self.images = train['path'].tolist()
                self.labels = train['label'].tolist()
            elif mode == "test":
                self.images = test['path'].tolist()
                self.labels = test['label'].tolist()
            else:
                raise ValueError
            
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        phow_feature = self.path2phow_features[img][0]
        phow_feature = torch.tensor(phow_feature)
        
        image = Image.open(img).convert("RGB")
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Apply transformations
        augmented = self.transform(image=image_np)
        # Convert numpy array to PIL Image
        img = augmented['image']
        
        label = torch.tensor(label)
 
        return img,phow_feature, label