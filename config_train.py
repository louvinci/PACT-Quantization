import sys
import time
import numpy as np

from easydict import EasyDict as edict

C = edict()
config = C
cfg = C


C.dataset = 'cifar10'
#C.model ='MBv2_cf10'
#C.model = 'RN20'
C.model='ToyNet'

if 'cifar' in C.dataset:
    """Data Dir and Weight Dir"""
    #C.dataset_path = "F:\Paper\Pytorch\cifar_data"
    C.dataset_path = "/root/workspace/cifar_data"
    if C.dataset == 'cifar10':
        C.num_classes = 10
    elif C.dataset == 'cifar100':
        C.num_classes = 100
    else:
        print('Wrong dataset.')
        sys.exit()

    """Image Config"""

    C.num_train_imgs = 50000
    C.num_eval_imgs = 10000
    ####################  Modle Config #################### 
    C.num_classes = 10
    C.strides=[1,2,1,2]
    #C.strides=[1,2,2,2]
    
    C.num_layer_list = [1, 3, 3, 3]
    C.num_channel_list = [16, 32,32, 64]
    #C.num_channel_list = [32, 64,128, 256]
    C.stem_channel = 16
    #C.stem_channel = 32
    C.header_channel = 512
    #C.header_channel = 1024
    if C.model == 'ToyNet':
        C.layer_abit = [32,  8, 8,8,8, 8,8,8, 8,8,8,  8,16]# last 16 s
        C.layer_wbit = [8,   8, 8,8,8, 8,8,8, 8,8,8,  8,8]
    elif C.model =='MBv2_cf10':
        #stem + [ 1, 2, 3, 4, 3, 3, 1] + head + fc
        C.layer_abit = [32, 8, 8,8, 8,8,8, 8,8,8,8, 8,8,8, 8,8,8, 8, 32,32]# last 16 s
        C.layer_wbit = [32, 8, 8,8, 8,8,8, 8,8,8,8, 8,8,8, 8,8,8, 8, 16,16]
    elif C.model=='RN20':
        #C.layer_abit = [  8,8,8, 8,8,8, 8,8,8]
        #C.layer_wbit = [  8,8,8, 8,8,8, 8,8,8]
        C.layer_abit = [  32]*9
        C.layer_wbit = [  32]*9
    else:
        print('Wrong Model.')
        sys.exit()

    #C.layer_abit = [32, 32, 32,32,32, 32,32,32, 32,32,32, 32,32]
    #C.layer_wbit = [32, 32, 32,32,32, 32,32,32, 32,32,32, 32,32]
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1
   ####################  Train Config #################### 
    C.opt = 'Sgd'
    C.momentum = 0.9
    C.weight_decay = 5e-4
    C.betas=(0.5, 0.999)
    C.num_workers = 4
    C.pretrain = "ckpt/finetune-{0}".format(C.model)

    C.batch_size = 128 #128 # 96->128
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32

    C.save = "finetune-{0}".format(C.model)
    C.nepochs = 300 #600->300
    C.eval_epoch = 1
    C.lr_schedule = 'cosine'
    C.lr = 0.01
    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = 'ckpt/search'

    C.eval_only = False
elif C.dataset == 'imagenet':
    C.dataset_path = "/root/datasets/imagenet"
    C.num_workers = 16 # workers per gpu
    C.batch_size = 512

    C.num_classes = 1000

    ####################  Modle Config #################### 


    C.num_train_imgs = 1281167
    C.num_eval_imgs = 50000

    C.bn_eps = 1e-5
    C.bn_momentum = 0.1
    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1984

    ####################  Train Config #################### 

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 4e-5

    C.betas=(0.5, 0.999)


    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/finetune'




        
    ########################################
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224

    C.save = "finetune"
    ########################################

    # C.nepochs = 360
    C.nepochs = 180

    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.2

    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [90, 180, 270]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0

    C.load_path = 'ckpt/search'

    C.eval_only = False

    C.efficiency_metric = 'flops'

else:
    print('Wrong dataset.')
    sys.exit()