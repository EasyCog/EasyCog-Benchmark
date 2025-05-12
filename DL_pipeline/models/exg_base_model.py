import torch
import torch.nn as nn
import torch.nn.functional as F
from DL_pipeline.models.cbramod import CBraMod
from DL_pipeline.models.exg_backbone import CBraMod_ExGBackbone, CBraMod_ExGRegressor, CBraMod_ExGClassifier
from DL_pipeline.models.basic_modules import MLP

class CBraMod_BaseModel(nn.Module):
    def __init__(self, input_channels=16, num_classes=9, n_layer=12, n_dim=200, load_pretrained_backbone=True, freeze_backbone=False,
                 pretrained_path='/home/mmWave_group/EasyCog/DL_pipeline/models/CBraMod_pretrained_weights.pth'):
        super(CBraMod_BaseModel, self).__init__()

        self.backbone = CBraMod_ExGBackbone(input_channels=input_channels, 
                                            num_classes=num_classes, 
                                            n_layer=n_layer, 
                                            load_pretrained_backbone=load_pretrained_backbone, 
                                            freeze_backbone=freeze_backbone,
                                            pretrained_path=pretrained_path)
        
        self.classifier = CBraMod_ExGClassifier(input_channels=input_channels, 
                                               dim=n_dim, 
                                               num_classes=num_classes, 
                                               output_last_layer=False)
        
        self.final_layer = MLP(input_dim=n_dim, hidden_dim=n_dim, output_dim=num_classes, use_batch_norm=False)
    
    def get_features(self, x):
        if isinstance(x, list):
            x = x[0]
        
        feats = self.backbone(x)
        feats = self.classifier(feats)[0]

        return feats
    
    
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        
        feats = self.backbone(x)
        cls_out = self.final_layer(self.classifier(feats)[0])
        return cls_out, 0


class CBraMod_BaseModel_FeatureExtractor(nn.Module):
    def __init__(self, input_channels=16, num_classes=9, n_layer=12, n_dim=200, load_pretrained_backbone=True, freeze_backbone=False,
                 pretrained_path='/home/mmWave_group/EasyCog/DL_pipeline/models/CBraMod_pretrained_weights.pth'):
        super(CBraMod_BaseModel_FeatureExtractor, self).__init__()

        self.backbone = CBraMod_ExGBackbone(input_channels=input_channels, 
                                            num_classes=num_classes, 
                                            n_layer=n_layer, 
                                            load_pretrained_backbone=load_pretrained_backbone, 
                                            freeze_backbone=freeze_backbone,
                                            pretrained_path=pretrained_path)
        
        self.classifier = CBraMod_ExGClassifier(input_channels=input_channels, 
                                               dim=n_dim, 
                                               num_classes=num_classes, 
                                               output_last_layer=False)
        
    
    def get_features(self, x):
        if isinstance(x, list):
            x = x[0]
        
        feats = self.backbone(x)
        feats = self.classifier(feats)[0]

        return feats
    
    
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        
        feats = self.backbone(x)
        feats = self.classifier(feats)[0]

        return feats
    


    
if __name__ == "__main__":
    model = CBraMod_BaseModel(input_channels=16, num_classes=10, n_layer=12, n_dim=200, load_pretrained_backbone=True, freeze_backbone=False)
    x = torch.randn(10, 16, 375)
    cls_out = model(x)
    print(cls_out.shape)

