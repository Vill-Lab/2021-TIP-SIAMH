# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import glob
import numpy as np

@DATASET_REGISTRY.register()
class Last(ImageDataset):
    dataset_url = None
    dataset_name = 'last'
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir="last/"
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.query_test_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_test_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir
        ]
        self.check_before_run(required_files)
        self.pid2label = self.get_pid2label(self.train_dir)
        self.train = self._process_dir(self.train_dir, pid2label=self.pid2label, relabel=True)
        self.query = self._process_dir(self.query_test_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_test_dir, relabel=False, recam=len(self.query))
        
        
        self.query_new = self._process_dir(self.query_dir, relabel=False)
        self.gallery_new = self._process_dir(self.gallery_dir, relabel=False, recam=len(self.query))




        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self.num_query_test_pids, self.num_query_test_imgs, self.num_query_test_cams = self.get_imagedata_info(self.query)
        self.num_gallery_test_pids, self.num_gallery_test_imgs, self.num_gallery_test_cams = self.get_imagedata_info(self.query)
        super(Last, self).__init__(self.train, self.query, self.gallery, **kwargs)
    def get_pid2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))            # [103367,]

        pid_container = set()
        for img_path in img_paths:
            pid = int(os.path.basename(img_path).split('_')[0])
            pid_container.add(pid)
        pid_container = np.sort(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label
    def _process_dir(self, dir_path, pid2label=None, relabel=False, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        img_paths = sorted(img_paths)
        dataset = []
        for ii, img_path in enumerate(img_paths):
            pid = int(os.path.basename(img_path).split('_')[0])
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
    
    
    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams