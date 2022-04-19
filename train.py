from __future__ import division
from locale import normalize
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

from config_train import config
from ToyModel import ToyNet
from MobileNetv2 import MobileNetV2_CF
from ResNet_CF import resnet20_cifar10_Q as RN20_CF

from quant_fn import Conv2d_Q,Linear_Q
from thop import profile
from thop.vision.basic_hooks import count_convNd,count_linear
custom_ops = {Conv2d_Q: count_convNd,Linear_Q:count_linear}
import numpy as np
from PIL import Image
import argparse



parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers per gpu')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
args = parser.parse_args()


best_acc = 0
best_epoch = 0


def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers


    main_worker(config)


def main_worker(config):
    global best_acc
    global best_epoch

    pretrain = config.pretrain




    if type(pretrain) == str:
        config.save = pretrain
    else:
        config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(config))

    if config.model == 'ToyNet':
        model = ToyNet(config=config)
    elif config.model =='MBv2_cf10':
        model = MobileNetV2_CF(config=config)
    elif config.model == 'RN20':
        model = RN20_CF(config=config)
    else:
        raise NotImplementedError
    
    
    #print(model)
    #return
    #logging.info("model = %s",str(model))
    #
    # flops, params = profile(model, inputs=(torch.randn(1, 3, config.image_height, config.image_width),), custom_ops=custom_ops)
    # logging.info("params = %fM, FLOPs = %fM", params / 1e6, flops / 1e6)
    model = torch.nn.DataParallel(model).cuda()


    # for param, val in model.named_parameters():
    #     print(param, val.device)
        
    #     if val.device.type == 'cpu':
    #         print('This tensor is on CPU.')
    #         sys.exit()

    #criterion = nn.CrossEntropyLoss()
    model_trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model_trainable_parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model_trainable_parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ############################## 
    # total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

    cudnn.benchmark = True


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    if type(pretrain) == str and os.path.exists(pretrain + "/weights_latest.pt"):
        pretrained_model = torch.load(pretrain + "/weights_latest.pt")
        partial = pretrained_model['state_dict']
        
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        optimizer.load_state_dict(pretrained_model['optimizer'])
        lr_policy.load_state_dict(pretrained_model['lr_scheduler'])
        start_epoch = pretrained_model['epoch'] + 1

        best_acc = pretrained_model['acc']
        best_epoch = pretrained_model['epoch']

        print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')


    # data loader ############################
    if 'cifar' in config.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if config.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
        elif config.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
        else:
            print('Wrong dataset.')
            sys.exit()


    elif config.dataset == 'imagenet':
        train_dir = config.dataset_path+'/train'
        test_dir  = config.dataset_path+'/val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_data = torchvision.datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))
        test_data = torchvision.datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

    else:
        print('Wrong dataset.')
        sys.exit()



    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        pin_memory=True, num_workers=config.num_workers, sampler=None)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=config.num_workers)

    if config.eval_only:
        acc1,acc5 = infer(0, model, test_loader, logger)
        logging.info('Eval: acc1 = %f, acc5 = %f', acc1,acc5)
        state = {}
        state['state_dict'] = model.state_dict()
        state['acc'] = acc1
        torch.save(state, os.path.join(config.save, 'weights_best.pt'))
        sys.exit(0)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(start_epoch, config.nepochs):
        
        logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))
        start_t = time.time()
        train(train_loader, model, optimizer, lr_policy, logger, epoch, config)
        total_t = time.time()-start_t
        logging.info('Consuming {0:.2f}s'.format(total_t))
        torch.cuda.empty_cache()
        lr_policy.step()


        eval_epoch = config.eval_epoch

        #validation
        if (epoch+1) % eval_epoch == 0:

            with torch.no_grad():
                acc1,acc5 = infer(epoch, model, test_loader, logger)

            logger.add_scalar('acc/val', acc1, epoch)
            logging.info("Test Acc1:%.3f,Acc5:%.3f, Best Acc1:%.3f,Best Epoch:%d\n" % (acc1,acc5, best_acc,best_epoch))

            if acc1 > best_acc:
                best_acc = acc1
                best_epoch = epoch+1
                state = {}
                state['state_dict'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['lr_scheduler'] = lr_policy.state_dict()
                state['epoch'] = epoch 
                state['acc'] = acc1
                torch.save(state, os.path.join(config.save, 'weights_best.pt'))

            




def train(train_loader, model, optimizer, lr_policy, logger, epoch, config):
    model.train()

    lambda_alpha = 0.0002
    for batch_idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()

        start_time = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data_time = time.time() - start_time

        criterion = model.module._criterion
        logit = model(input)
        loss = criterion(logit, target)
        
        l2_alpha = 0.0
        for name, param in model.named_parameters():
            if "scale_coef" in name:
                l2_alpha += torch.pow(param, 2)
        alpha_loss = lambda_alpha * l2_alpha
        total_loss = loss+alpha_loss

        total_loss.backward()
        
        #nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


        total_time = time.time() - start_time

        if batch_idx % 10 == 0:
            
            logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Alpha_loss =%.3f Time=%.3f Data Time=%.3f" % (epoch+1, config.nepochs, batch_idx + 1, len(train_loader), loss.item(), alpha_loss.item(),total_time, data_time))
            
            logger.add_scalar('loss/train_loss', loss, epoch*len(train_loader)+batch_idx)
            logger.add_scalar('loss/alpha_loss', alpha_loss, epoch*len(train_loader)+batch_idx)
            logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
            '''
            id=0
            for name, param in model.named_parameters():
                if "scale_coef" in name:
                    logger.add_scalar('layer{0}'.format(id), param.item(), epoch*len(train_loader)+batch_idx)
                    id+=1
                if id==11:
                    break
            '''
            
    torch.cuda.empty_cache()
    del loss


def infer(epoch, model, test_loader, logger):
    model.eval()
    prec1_list ,prec5_list= [],[]
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_var = Variable(data[0]).cuda()
            target_var = Variable(data[1]).cuda()

            output = model(input_var)
            prec1, prec5 = accuracy(output.data, target_var, topk=(1,5))
            prec1_list.append(prec1)
            prec5_list.append(prec5)

        acc  = sum(prec1_list)/len(prec1_list)
        acc5 = sum(prec5_list)/len(prec5_list)
    
    torch.cuda.empty_cache()  
    del input_var,target_var,output
    return acc,acc5



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
