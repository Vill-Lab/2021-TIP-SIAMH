# Salience-Guided Iterative Asymmetric Mutual Hashing for Fast Person Re-Identification (IEEE TIP 2021) 

This is the pytorch implementation of the [paper](https://ieeexplore.ieee.org/document/9531552) (accpted by IEEE TIP 2021).

<img src='figures/21_tip_igoas.png'>

**Fig 1**.SIAMH framework

# Training
Training details for differnet lenght of hash codes are shown in the folder of training logs. 

Pretrained-weights: https://pan.baidu.com/s/16UZK6i45r56MkOL95a3zKg (1aiv)

To train/evaluate SIAMH onMarket-1501, do

    python tools/train_net.py --config-file configs/Market1501/myconfig.yml --teacher-config-file configs_teacher/Market1501/myconfig.yml  

    python tools/train_net.py --config-file configs/Market1501/myconfig.yml --teacher-config-file configs_teacher/Market1501/myconfig.yml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file  

## Citation 
If you find SIAMH useful in your research, please consider citing.

```
@article{zhao2021salience,
  title={Salience-guided iterative asymmetric mutual hashing for fast person re-identification},
  author={Zhao, Cairong and Tu, Yuanpeng and Lai, Zhihui and Shen, Fumin and Shen, Heng Tao and Miao, Duoqian},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={7776--7789},
  year={2021},
  publisher={IEEE}
}
```

## Reference
This code is based on [torchreid](https://github.com/KaiyangZhou/deep-person-reid).


