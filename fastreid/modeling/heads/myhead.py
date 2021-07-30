# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch.autograd import Function

@REID_HEADS_REGISTRY.register()
class MyHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer
        self.num_classes=751
        if num_classes==0:
            raise AssertionError
        self.hashbit=2048
        self.parts=3
        reduced_dim_pcb=256

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':          
            self.classifier_hash = nn.Linear(self.hashbit, self.num_classes, bias=False)
            self.classifier_real = nn.Linear(self.hashbit, self.num_classes,bias=False)
        elif cls_type == 'arcSoftmax':    
            self.classifier_hash = ArcSoftmax(cfg, self.hashbit, self.num_classes)
            self.classifier_real = ArcSoftmax(cfg, self.hashbit, self.num_classes)
        elif cls_type == 'circleSoftmax': 
            self.classifier_hash = CircleSoftmax(cfg, self.hashbit, self.num_classes)
            self.classifier_real = CircleSoftmax(cfg, self.hashbit, self.num_classes)
        elif cls_type == 'amSoftmax':     
            self.classifier_hash = AMSoftmax(cfg, self.hashbit, self.num_classes)
            self.classifier_real = AMSoftmax(cfg, self.hashbit, self.num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")

        self.classifier_hash.apply(weights_init_classifier)
        self.classifier_real.apply(weights_init_classifier)

       
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout_pcb = nn.Dropout(p=0.5)
        self.parts_avgpool2 = nn.AdaptiveAvgPool2d((1,2))
        #self.dimension=nn.Sequential(nn.Linear(2048,1024),
        #                             nn.BatchNorm1d(1024),
        #                             nn.ReLU())
        self.dropout_pcb2 = nn.Dropout(p=0.5)




        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier_BoT = BNClassifier(2048,self.num_classes)

        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier_BoT_gmp = BNClassifier(2048,self.num_classes)


        self.bn_pcb = nn.BatchNorm1d(256)
        self.bn_real = nn.BatchNorm1d(self.hashbit)
        self.split_num=10
        self.fc_split = nn.Linear(2048*2+2048*self.parts, self.hashbit*self.split_num)


        self.classifier_v = nn.Linear(self.hashbit*self.split_num,self.num_classes)
        self.dropout=nn.Dropout(0.5)
        self.de1 = DivideEncode(self.hashbit*self.split_num, self.split_num)
     






    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """

        f=features
        v_g = self.parts_avgpool(f)
        
        v_g = self.dropout_pcb(v_g)
        v_pcb=v_g.view(v_g.size(0), -1)

        #v_g_2= self.parts_avgpool(f)
        
        #v_g_2 = self.dropout_pcb(v_g_2)
        #v_pcb_2=v_g.view(v_g_2.size(0), -1)


        x_BoT=self.gap(f)
        features_BoT=x_BoT.squeeze(dim=2).squeeze(dim=2)

        _, cls_score = self.classifier_BoT(features_BoT)


        x_BoT_gmp=self.gmp(f)
        features_BoT_gmp=x_BoT_gmp.squeeze(dim=2).squeeze(dim=2)
        _, cls_score_mean = self.classifier_BoT_gmp(features_BoT_gmp)


        v=torch.cat([v_pcb,features_BoT,features_BoT_gmp],dim=1)
        v=self.dropout(v)
        v_split=self.fc_split(v)

        v_classify=self.classifier_v(v_split)
        h_return=self.de1(v_split)
        h_return=self.bn_real(h_return)
        b_return=hash_layer(h_return)
        
        cls_outputs_hash = self.classifier_hash(b_return)
        cls_outputs_real = self.classifier_real(h_return)
        if not self.training: return b_return#cls_outputs_hash,cls_outputs_real, cls_outputs_real, cls_outputs_real,b_return,h_return,v_classify,v_pcb,cls_score,cls_score_mean,features


        # Evaluation

        # Training



        pred_class_logits_hash = F.linear(b_return, self.classifier_hash.weight)
        pred_class_logits_real = F.linear(h_return, self.classifier_real.weight)
        return cls_outputs_hash,cls_outputs_real, pred_class_logits_hash, pred_class_logits_real,b_return,h_return,v_classify,v_pcb,cls_score,cls_score_mean,features





def weights_init_kaiming(m):
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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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



class DivideEncode(nn.Module):
    '''
    Implementation of the divide-and-encode module in,
    Simultaneous Feature Learning and Hash Coding with Deep Neural Networks
    https://arxiv.org/pdf/1504.03410.pdf
    '''
    def __init__(self, num_inputs, num_per_group):
        super().__init__()
        assert num_inputs % num_per_group == 0, \
            "num_per_group should be divisible by num_inputs."
        self.num_groups = num_inputs // num_per_group
        self.num_per_group = num_per_group
        weights_dim = (self.num_groups, self.num_per_group)
        self.weights = nn.Parameter(torch.empty(weights_dim))
        nn.init.xavier_normal_(self.weights)

    def forward(self, X):
        X = X.view((-1, self.num_groups, self.num_per_group))
        return X.mul(self.weights).sum(2)

class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU())
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
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
