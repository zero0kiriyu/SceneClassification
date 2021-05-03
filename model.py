from efficientnet_pytorch import EfficientNet
import json
import torch.nn as nn

import torch

class ClassifierNet(nn.Module):
    def __init__(self,phow_feature_num=500,classes = 15):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        
        # replace the full connection layer
        for p in self.efficientnet.parameters():
            p.requires_grad=False

        num_in_features = self.efficientnet._fc.in_features
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(num_in_features+phow_feature_num, 512),
                                 nn.ReLU(),  
                                 nn.Dropout(0.25),

                                 nn.Linear(512, 128), 
                                 nn.ReLU(),  
                                 nn.Dropout(0.50), 

                                 nn.Linear(128,classes))
        
    def forward(self, img):#, phow_feature
        batch_size = img.shape[0]
        phow_feature = img[:,3:]
        img = img[:,:3]
        
        phow_feature = phow_feature.resize_(batch_size,500)
        efficient_feature = self.efficientnet.extract_features(img)
        out = self.pooling(efficient_feature)
        out = out.flatten(start_dim=1)
        out = torch.cat((out,phow_feature),1)
        out = self.fc(out)
        return out

