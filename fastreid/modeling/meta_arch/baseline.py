# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torch.nn.modules import loss

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from torch.nn import functional as F
from torch.autograd import Variable
from queue import Queue,LifoQueue,PriorityQueue
import numpy as np
import cv2
@META_ARCH_REGISTRY.register()

class Baseline(nn.Module):
    def __init__(self, cfg,cfg_teacher=None):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        
        self.iter=0

        self.prv_b_q=[]
        self.prv_h_q=[]
        self.prv_gt_label_q=[]
        self.prv_global_q=[]


        self.flag=False
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'fastavgpool':  pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':    pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.pids_num=num_classes
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)
        
        self.simloss=SIMSelfDistillLoss()



        self.mseloss=torch.nn.MSELoss(reduce=True, size_average=True)
        self.losssp=SP()
    @property
    def device(self):
        return self.pixel_mean.device
    def attent_erase(self,outputs,width=384,height=128):
        erase_x=[]
        erase_y=[]
        for j in range(outputs.size(0)):
            am = outputs[j, ...].detach().cpu().numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            m=np.argmax(am)
            r, c = divmod(m, am.shape[1])
            erase_x.append(r)
            erase_y.append(c)

        erase_x=torch.tensor(erase_x).cuda()
        erase_y=torch.tensor(erase_y).cuda()
        return erase_x,erase_y
    def forward(self, batched_inputs,masks=None):
        images = self.preprocess_image(batched_inputs)
        if self.training:
            features= self.backbone(images)
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)
            #print(targets)
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            return self.heads(features, targets,masks=masks), targets
        else:
            features= self.backbone(images)
            return self.heads(features)

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs,gt_labels,outputs_teacher,gt_labels_teacher):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        v, global_feat,b_return,h_return,cls_score_hash,cls_score_real,featuremap_s,features_new,newscore= outputs



        cls_score_hash_t,cls_score_real_t ,_,_,b_return_teacher,h_return_teacher,v_clssifty,_,_,_,featuremap_t=outputs_teacher


        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        pids_g=self.parse_pids(gt_labels,self.pids_num)
        
        
        #greedy loss
        x=v_clssifty
        b=b_return
        h=h_return
        pids_ap=pids_g.to(self.device)
        target_b = F.cosine_similarity(b[:pids_ap.size(0) // 2], b[pids_ap.size(0) // 2:])
        target_x = F.cosine_similarity(x[:pids_ap.size(0) // 2], x[pids_ap.size(0) // 2:])
        loss1 = F.mse_loss(target_b, target_x)
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
        loss_greedy = loss1 + 0.2 * loss2
        
        

        #crossentropy loss
        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_greedy']=loss_greedy
            loss_dict['loss_cls'] =  CrossEntropyLoss(self._cfg)(cls_score_hash, gt_labels)+CrossEntropyLoss(self._cfg)(cls_score_real, gt_labels)+CrossEntropyLoss(self._cfg)(v, gt_labels)*0.6
        pids_distill=gt_labels.long()
        
        #loss_dict['loss_center']=(dualcenter(h_return,gt_labels))*0.04
        #distillation loss
        loss_dict['loss_distill']=0.3*(torch.sum(1-torch.cosine_similarity(h_return,h_return_teacher,dim=1))+torch.sum(1-torch.cosine_similarity(b_return,b_return_teacher,dim=1)))#+((loss_kd(cls_score_hash,pids_distill,cls_score_hash_t)+loss_kd(cls_score_real,pids_distill,cls_score_real_t)))*5
        loss_dict['loss_distill_2']=(loss_kd(cls_score_hash,pids_distill,cls_score_hash_t)+loss_kd(cls_score_real,pids_distill,cls_score_real_t))*15+(loss_kd(cls_score_hash,pids_distill,cls_score_real.detach()))*12
        #featuremap loss
        loss_dict['loss_crd']=(self.losssp(featuremap_s.reshape(featuremap_s.shape[0],featuremap_s.shape[1]*featuremap_s.shape[2]*featuremap_s.shape[3]),\
            featuremap_t.reshape(featuremap_t.shape[0],featuremap_t.shape[1]*featuremap_t.shape[2]*featuremap_t.shape[3])))*1200
        
        #self distill loss
        loss_dict['loss_sim']=(self.simloss(h_return=b_return_teacher,b_return=b_return))*40

        loss_dict['loss_selfdis']=loss_kd(cls_score_real,pids_distill,newscore.detach())*1
        loss_dict['loss_trisal']=TripletLoss(self._cfg)(features_new, gt_labels)+CrossEntropyLoss(self._cfg)(newscore, gt_labels)*0.3

        loss_dict['loss_mse']=self.mseloss(b_return.float(),b_return_teacher.detach().float())*0.1+self.mseloss(h_return.float(),h_return_teacher.detach().float())*0.1

        if "TripletLoss" in loss_names:
            if self.flag:
                prv_b_1=torch.Tensor(self.prv_b_q[0]).cuda()
                prv_b_2=torch.Tensor(self.prv_b_q[1]).cuda()
                prv_b_3=torch.Tensor(self.prv_b_q[2]).cuda()
                prv_b_4=torch.Tensor(self.prv_b_q[3]).cuda()
                prv_b_5=torch.Tensor(self.prv_b_q[4]).cuda()
                prv_b_6=torch.Tensor(self.prv_b_q[5]).cuda()
                prv_b_7=torch.Tensor(self.prv_b_q[6]).cuda()

                prv_h_1=torch.Tensor(self.prv_h_q[0]).cuda()
                prv_h_2=torch.Tensor(self.prv_h_q[1]).cuda()
                prv_h_3=torch.Tensor(self.prv_h_q[2]).cuda()
                prv_h_4=torch.Tensor(self.prv_h_q[3]).cuda()
                prv_h_5=torch.Tensor(self.prv_h_q[4]).cuda()
                prv_h_6=torch.Tensor(self.prv_h_q[5]).cuda()
                prv_h_7=torch.Tensor(self.prv_h_q[6]).cuda()
                prv_labels_1=torch.Tensor(self.prv_gt_label_q[0]).cuda()
                prv_labels_2=torch.Tensor(self.prv_gt_label_q[1]).cuda()
                prv_labels_3=torch.Tensor(self.prv_gt_label_q[2]).cuda()
                prv_labels_4=torch.Tensor(self.prv_gt_label_q[3]).cuda()
                prv_labels_5=torch.Tensor(self.prv_gt_label_q[4]).cuda()         
                prv_labels_6=torch.Tensor(self.prv_gt_label_q[5]).cuda()
                prv_labels_7=torch.Tensor(self.prv_gt_label_q[6]).cuda()  

                global_1=torch.Tensor(self.prv_global_q[0]).cuda()
                global_2=torch.Tensor(self.prv_global_q[1]).cuda()
                global_3=torch.Tensor(self.prv_global_q[2]).cuda()
                global_4=torch.Tensor(self.prv_global_q[3]).cuda()
                global_5=torch.Tensor(self.prv_global_q[4]).cuda()         
                global_6=torch.Tensor(self.prv_global_q[5]).cuda()
                global_7=torch.Tensor(self.prv_global_q[6]).cuda()       

                new_global=torch.cat([global_1,global_2,global_3,global_4,global_5,global_6,global_7,global_feat],dim=0)

                new_b_return=torch.cat([prv_b_1,prv_b_2,prv_b_3,prv_b_4,prv_b_5,prv_b_6,prv_b_7,b_return],dim=0)
                new_h_return=torch.cat([prv_h_1,prv_h_2,prv_h_3,prv_h_4,prv_h_5,prv_h_6,prv_h_7,h_return],dim=0)
                new_gt_labels=torch.cat([prv_labels_1,prv_labels_2,prv_labels_3,prv_labels_4,prv_labels_5,prv_labels_6,prv_labels_7,gt_labels],dim=0)
                loss_dict['loss_triplet']=(TripletLoss(self._cfg)(new_b_return, new_gt_labels)+TripletLoss(self._cfg)(new_h_return, new_gt_labels)+TripletLoss(self._cfg)(new_global, new_gt_labels))#*weight_tri
            else:
                loss_dict['loss_triplet']=(TripletLoss(self._cfg)(b_return, gt_labels)+TripletLoss(self._cfg)(h_return, gt_labels)+TripletLoss(self._cfg)(global_feat, gt_labels))#*weight_tri
        loss_all=loss_dict['loss_sim']+loss_greedy+loss_dict['loss_triplet']+loss_dict['loss_cls']+loss_dict['loss_distill']+loss_dict['loss_distill_2']+loss_dict['loss_crd']

        
        #loss_dict['loss_selfdis']=loss_kd(v,pids_distill,cls_score_new.detach())*3
        #loss_all+=loss_dict['loss_selfdis']


        if loss_all<13:        
            self.flag=True
        if len(self.prv_b_q)<7:
            self.prv_global_q.append(global_feat.detach().cpu().float())
            self.prv_b_q.append(b_return.detach().cpu().float())
            self.prv_h_q.append(h_return.detach().cpu().float())
            self.prv_gt_label_q.append(gt_labels.detach().cpu().float())
        else:
            self.prv_global_q.pop(0)
            self.prv_b_q.pop(0)
            self.prv_h_q.pop(0)
            self.prv_gt_label_q.pop(0)

            self.prv_global_q.append(global_feat.detach().cpu().float())
            self.prv_b_q.append(b_return.detach().cpu().float())
            self.prv_h_q.append(h_return.detach().cpu().float())
            self.prv_gt_label_q.append(gt_labels.detach().cpu().float())

        return loss_dict
    def parse_pids(self,pids,num_classes):
        pids_return=torch.zeros(pids.shape[0],num_classes)
        for i in range(pids.shape[0]):
            pids_return[i][pids[i]-1]=1
        return pids_return.cuda()

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()
def kd_loss_function(output, target_output,temperature):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd
def loss_distill_pow(outputs_s,outputs_t,labels):
    s_similarity = torch.mm(outputs_s, outputs_s.transpose(0, 1))
    s_similarity = F.normalize(s_similarity, p=2, dim=1)
    t_similarity = torch.mm(outputs_t, outputs_t.transpose(0, 1)).detach()
    t_similarity = F.normalize(t_similarity, p=2, dim=1)
    loss = (s_similarity - t_similarity).pow(2).mean()
    return loss

def loss_kd_regularization(outputs, labels):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = 0.95
    T = 6
    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*1.0

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss
def loss_kd(outputs, labels, teacher_outputs):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = 0.95
    T = 6
    #multiplier=2
    #shape=outputs.shape[1]
    #outputs = F.layer_norm(
    #    outputs, torch.Size((shape,)), None, None, 1e-7)*multiplier
    #teacher_outputs = F.layer_norm(
    #    teacher_outputs, torch.Size((shape,)), None, None, 1e-7)*multiplier

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis





class SIMSelfDistillLoss:
    '''
    Self-Distillation Loss of Similarity
    Reference:
    Paper: Wang et al. Faster Person Re-Identification. ECCV2020
    Args:
        feats_list(list): a list of feats. the first one is teacher, all the others are students
    '''

    def __init__(self):
        super(SIMSelfDistillLoss, self).__init__()

    def __call__(self, h_return,b_return):
        s_feats_list = b_return
        t_feats_list = h_return
        return self.sim_distill_loss(s_feats_list, t_feats_list)

    def sim_distill_loss(self, s_feats_list, t_feats_list):
        '''
        compute similarity distillation loss
        :param score_list:
        :param mimic(list): [teacher, student]
        :return:
        '''
        loss = 0
        s_similarity = torch.mm(s_feats_list, s_feats_list.transpose(0, 1))
        s_similarity = F.normalize(s_similarity, p=2, dim=1)
        t_similarity = torch.mm(t_feats_list, t_feats_list.transpose(0, 1)).detach()
        t_similarity = F.normalize(t_similarity, p=2, dim=1)
        loss += (s_similarity - t_similarity).pow(2).mean()*10
        return loss


class SP(nn.Module):
	'''
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	'''
	def __init__(self):
		super(SP, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), -1)
		G_s  = torch.mm(fm_s, fm_s.t())
		norm_G_s = F.normalize(G_s, p=2, dim=1)

		fm_t = fm_t.view(fm_t.size(0), -1)
		G_t  = torch.mm(fm_t, fm_t.t())
		norm_G_t = F.normalize(G_t, p=2, dim=1)

		loss = F.mse_loss(norm_G_s, norm_G_t)

		return loss
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


import random
import math
class RandomErasing_v2(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing_v2, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y):
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area#area * (0.35 - 0.2 * epoch / 70)#
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img.size()[3] and h < img.size()[2]:
                    x1 = erase_x[i]
                    y1 = erase_y[i]
                    if x1+h>img.size()[2]:
                        x1=img.size()[2]-h
                    if y1+w>img.size()[3]:
                        y1=img.size()[3]-w
                    if img.size()[1] == 3:
                        img[i, 0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                        img[i, 1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                        img[i, 2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
       
                        break
        return img
