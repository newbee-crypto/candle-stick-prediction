# model.py

import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(384 + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, img, features):
        img_feat = self.vit(img)
        x = torch.cat((img_feat, features), dim=1)
        return self.fc(x)