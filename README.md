# SIAMH IEEE Transaction on Image Processing 2021

Training details for differnet lenght of hash codes are shown in the folder of training logs.  

Train: python tools/train_net.py --config-file configs/Market1501/myconfig.yml --teacher-config-file configs_teacher/Market1501/myconfig.yml  

Test:  python tools/train_net.py --config-file configs/Market1501/myconfig.yml --teacher-config-file configs_teacher/Market1501/myconfig.yml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file  

pretrained-weights: https://pan.baidu.com/s/16UZK6i45r56MkOL95a3zKg (1aiv)
