# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch.autograd import Function
from .bnneck import BNNeck, BNNeck3
import copy
import cv2
@REID_HEADS_REGISTRY.register()
class DistillHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer
        self.hashbit=2048
        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':          self.classifier = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")
        self.fc_split=nn.Linear(in_feat,self.hashbit)
        self.classifier_real = nn.Linear(self.hashbit, num_classes, bias=False)
        self.classifier_hash = nn.Linear(self.hashbit, num_classes, bias=False)
        self.tanh=nn.Tanh()
        self.classifier.apply(weights_init_classifier)
        self.classifier_real.apply(weights_init_classifier)
        self.classifier_hash.apply(weights_init_classifier)
        self.bn_add=nn.BatchNorm1d(in_feat)

        
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        

        #第二个branch
        
        self.bnneck_new = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck_new.apply(weights_init_kaiming)
        self.classifier_real_new = BNClassifier(2048,num_classes)
        self.classifier_real_new.apply(weights_init_kaiming)
        self.pool_layer_new = nn.AdaptiveAvgPool2d(1)
        
        


    def forward(self, features, targets=None,masks=None):
        """
        See :class:`ReIDHeads.forward`.
        """

        if self.training:
            global_feat = self.pool_layer(features)
            bn_feat = self.bnneck(global_feat)
            bn_feat = bn_feat[..., 0, 0]
            bn_feat=self.bn_add(bn_feat)
            h_return=self.fc_split(bn_feat)
            h_return=self.tanh(h_return)
            b_return=hash_layer(h_return)
            cls_score_real=self.classifier_real(h_return)
            cls_score_hash=self.classifier_hash(b_return) 
            cls_outputs = self.classifier(bn_feat)
            
            
            masks=masks.cuda()
            masks=masks.unsqueeze(0)
            masks=F.interpolate(masks,size=[features.shape[2],features.shape[3]] )
            masks=masks.expand(features.shape[1],features.shape[0],features.shape[2],features.shape[3])
            masks=masks.reshape((features.shape[0],features.shape[1],features.shape[2],features.shape[3]))
            features_new=features*masks+features
            features_new=self.pool_layer_new(features_new)
            features_new=self.bnneck_new(features_new)
            features_new = features_new[..., 0, 0]
            #newfeat=torch.cat([features_new,bn_feat],dim=1)
            _,newscore=self.classifier_real_new(features_new)

            
            return cls_outputs,bn_feat,b_return,h_return,cls_score_hash,cls_score_real,features,features_new,newscore#,cls_score_new
        # Evaluation
        if not self.training:
            global_feat = self.pool_layer(features)
            bn_feat = self.bnneck(global_feat)
            bn_feat = bn_feat[..., 0, 0]
            bn_feat=self.bn_add(bn_feat)
            h_return=self.fc_split(bn_feat)
            h_return=self.tanh(h_return)
            b_return=hash_layer(h_return)
            return b_return 
    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


import numpy as np
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input) 




class BNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score


