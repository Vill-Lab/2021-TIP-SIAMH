# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from torch.nn import functional as F
from torch.autograd import Variable
from queue import Queue,LifoQueue,PriorityQueue
@META_ARCH_REGISTRY.register()

class Baseline(nn.Module):
    def __init__(self, cfg):
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


        self.prv_h=torch.zeros(size=(cfg.SOLVER.IMS_PER_BATCH,128))
        self.prv_b=torch.zeros(size=(cfg.SOLVER.IMS_PER_BATCH,128))
        self.prv_gt_label=torch.zeros(size=(cfg.SOLVER.IMS_PER_BATCH,751))
        self.prv_global=torch.zeros(size=(cfg.SOLVER.IMS_PER_BATCH,2048))
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

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)
            #print(targets)
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            return self.heads(features, targets), targets
        else:
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

    def losses(self, outputs, gt_labels,outputs_teacher,gt_labels_teacher):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        v,pred_class_logits, pred_class_logits_real,pred_class_logits_hash, global_feat,b_return,h_return,cls_score_hash,cls_score_real,features_map= outputs

        cls_score_hash_t,cls_score_real_t ,pred_class_logits_real_t,pred_class_logits_hash_t,b_return_teacher,h_return_teacher,v_clssifty,_,_,_,features_t=outputs_teacher

        pred_class_logits_real_t=pred_class_logits_real_t.cuda()
        pred_class_logits_hash_t=pred_class_logits_hash_t.cuda()

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        pids_g=self.parse_pids(gt_labels,self.pids_num)
        x=v_clssifty
        #x=cls_score_real
        b=b_return
        h=h_return
        pids_ap=pids_g.cuda()
        target_b = F.cosine_similarity(b[:pids_ap.size(0) // 2], b[pids_ap.size(0) // 2:])
        target_x = F.cosine_similarity(x[:pids_ap.size(0) // 2], x[pids_ap.size(0) // 2:])
        loss1 = F.mse_loss(target_b, target_x)
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
        loss_greedy = loss1 + 0.1 * loss2
        


        # Log prediction accuracy
        CrossEntropyLoss.log_accuracy(pred_class_logits_hash.detach(), gt_labels)
        CrossEntropyLoss.log_accuracy(pred_class_logits_real.detach(), gt_labels)
        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_greedy']=loss_greedy
            loss_dict['loss_cls'] = CrossEntropyLoss(self._cfg)(cls_score_hash, gt_labels)+CrossEntropyLoss(self._cfg)(cls_score_real, gt_labels)+CrossEntropyLoss(self._cfg)(v, gt_labels)
        pids_distill=gt_labels.long()
        #loss_dict['loss_distill']=0.4*(torch.sum(1-torch.cosine_similarity(h_return,h_return_teacher,dim=1))+torch.sum(1-torch.cosine_similarity(b_return,b_return_teacher,dim=1)))#+((loss_kd(cls_score_hash,pids_distill,cls_score_hash_t)+loss_kd(cls_score_real,pids_distill,cls_score_real_t)))*5
        loss_dict['loss_distill_2']=(loss_kd(cls_score_hash,pids_distill,cls_score_hash_t)+loss_kd(cls_score_real,pids_distill,cls_score_real_t))*12
        #loss_dict['loss_map']=torch.sum(1-torch.cosine_similarity(features_map,features_t))*0.05/features_map.shape[0]
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

                #print(self.prv_gt_label_q[:-2])
                new_global=torch.cat([global_1,global_2,global_3,global_4,global_5,global_6,global_7,global_feat],dim=0)

                new_b_return=torch.cat([prv_b_1,prv_b_2,prv_b_3,prv_b_4,prv_b_5,prv_b_6,prv_b_7,b_return],dim=0)
                new_h_return=torch.cat([prv_h_1,prv_h_2,prv_h_3,prv_h_4,prv_h_5,prv_h_6,prv_h_7,h_return],dim=0)
                new_gt_labels=torch.cat([prv_labels_1,prv_labels_2,prv_labels_3,prv_labels_4,prv_labels_5,prv_labels_6,prv_labels_7,gt_labels],dim=0)
                loss_dict['loss_triplet']=(TripletLoss(self._cfg)(new_b_return, new_gt_labels)+TripletLoss(self._cfg)(new_h_return, new_gt_labels)+TripletLoss(self._cfg)(new_global, new_gt_labels))#*weight_tri
            else:
                loss_dict['loss_triplet']=(TripletLoss(self._cfg)(b_return, gt_labels)+TripletLoss(self._cfg)(h_return, gt_labels)+TripletLoss(self._cfg)(global_feat, gt_labels))#*weight_tri
       
       
       
        """
        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet']=(TripletLoss(self._cfg)(b_return, gt_labels)+TripletLoss(self._cfg)(h_return, gt_labels))#*weight_tri
        """
        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = CircleLoss(self._cfg)(b_return, gt_labels)+CircleLoss(self._cfg)(h_return, gt_labels)#+CircleLoss(self._cfg)(v_pcb, gt_labels)

        #loss_all=loss_greedy+loss_dict['loss_triplet']+loss_dict['loss_cls']+loss_dict['loss_distill_2']+loss_dict['loss_distill']
        loss_all=loss_greedy+loss_dict['loss_triplet']+loss_dict['loss_cls']+loss_dict['loss_distill_2']#+loss_dict['loss_distill_2']

        if loss_all<12:        
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
        return pids_return



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
def CriterionPairWiseforWholeFeatAfterPool(preds_S,preds_T):
    loss=0
    for i in range(preds_S.shape[0]):
        for j in range(preds_S.shape[1]):
            feat_S = preds_S[i][j]
            feat_T = preds_T[i][j]
            feat_T.detach()
            print(feat_S.shape)
            #total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            #maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
            #loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
            loss=cos#loss+sim_dis_compute(feat_S,feat_T)
    return loss