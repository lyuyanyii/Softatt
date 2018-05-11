import argparse
import os
import time
import pickle
import sys
import numpy as np
import cv2
import models
import shutil
import utils
from utils import AverageMeter
import datasets
import torchvision.transforms as transforms
import torchvision

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torch.autograd import Variable
import tqdm
#import matplotlib.pyplot as plt
from imagenet1000_clsid_to_human import labels

from pascal_voc_dataset import PascalVOCSegmentation 
import torch
from PIL import Image
from pascal_voc_utils import convert_pascal_berkeley_augmented_mat_annotations_to_png
from pascal_voc_transform import ComposeJoint, RandomHorizontalFlipJoint, ResizeAspectRatioPreserve, RandomCropJoint


model_names = ['A', 'B', 'C', 'CDNet', 'PRNet', 'SNet', 'R18CAM', 'VGG', 'DsNet121', 'VGGC']

parser = argparse.ArgumentParser( description='Low Supervised Semantic Segmentation' )

parser.add_argument( '--arch', metavar='ARCH', choices=model_names )
parser.add_argument( '--save-folder', type=str, metavar='PATH' )
parser.add_argument( '--lr', type=float, help='initial learning rate' )
#parser.add_argument( '--lr-step', type=float, help='lr will be decayed at these steps' )
#parser.add_argument( '--lr-decay', type=float, help='lr decayed rate' )
parser.add_argument( '--data', type=str, help='the directory of data' )
parser.add_argument( '--dataset', type=str, choices=['mnist', 'cifar10', 'imgnet', 'chestx', 'place365', 'cub200', 'pascalvoc', 'object_discover'] )
parser.add_argument( '--tot-iter', type=int, help='total number of iterations' )
#parser.add_argument( '--val-iter', type=int, help='do validation every val-iter steps' )
parser.add_argument( '--workers', type=int, default=4, help='number of data loading workers (default:4)' )
parser.add_argument( '-b', '--batch-size', type=int, help='mini-batch size' )
parser.add_argument( '--pretrained', type=str, metavar='PATH' )
parser.add_argument( '--resume', type=str, metavar='PATH' )
parser.add_argument( '--momentum', type=float, default=0.9, help='momentum in optim' )
parser.add_argument( '--weight-decay', type=float, default=1e-4, help='weight decay' )
parser.add_argument( '--L1', type=float, default=0, help='adding L1-loss on mask' )
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument( '--evaluation',dest='evaluation',action='store_true' )
parser.add_argument( '--print-mask',dest='print_mask',action='store_true' )
parser.add_argument( '--no-mask',dest='no_mask',action='store_true' )
parser.add_argument( '--advtrain',dest='advtrain',action='store_true', help='using a training strategy as GAN' )
parser.add_argument( '--joint',dest='joint',action='store_true',help='jointly trainging' )
parser.add_argument( '--trained-cls', dest='trained_cls', action='store_true' )
parser.add_argument( '--binary', dest='binary', action='store_true' )
parser.add_argument( '--grad-check',dest='grad_check',action='store_true' )
parser.add_argument( '--single-batch-exp',dest='single_batch_exp',action='store_true')
parser.add_argument( '--save-img', type=str,default=None, help='path to save images' )
parser.add_argument( '--double', dest='double', action='store_true', help='using mask & 1-mask to compute 2 losses' )
parser.add_argument( '--threshold',type=int,default=1,help='threshold in evaluation' )
parser.add_argument( '--noise',dest='noise',action='store_true', help='adding noise when training mask' )
parser.add_argument( '--KL', type=float, default=0, help='using KL-divergence loss' )
parser.add_argument( '--gauss', dest='gauss', action='store_true', help='using Gaussian noise proportional to the mask' )
parser.add_argument( '--def-iter', type=int, default=0 )
parser.add_argument( '--myval', type=str, default=None, help='using my own validation')
parser.add_argument( '--advattack', dest='advattack', action='store_true', help='adversarial attack testing' )
parser.add_argument( '--noise-rate', type=float, default=1, help='noise rate in mask' )
parser.add_argument( '--large-reg', dest='large_reg', action='store_true', help='using large regnet' )
parser.add_argument( '--fb', dest='fb', action='store_true', help='fb server' )
parser.add_argument( '--hard-threshold-training', dest='hard_threshold_training', action='store_true', help='using hard threshold in training' )
parser.add_argument( '--sharp-reg', dest='sharp_reg', action='store_true', help='using sharp regulization' )
parser.add_argument( '--sharp-noise', dest='sharp_noise', action='store_true', help='using noise to induce sharpness' )
parser.add_argument( '--const-lr', dest='const_lr', action='store_true', help = 'disable learning rate schedule' )
parser.add_argument( '--quarter', dest='quarter', action='store_true', help='outputing mask with quarter width and quarter height' )
parser.add_argument( '--bbox', dest='bbox', action='store_true', help='enabling bbox' )
parser.add_argument( '--new-fc', dest='new_fc', action='store_true', help='new fc' )
parser.add_argument( '--quad', type=float, default=0, help='using quadratic regulization' )
parser.add_argument( '--quantiled', dest='quantiled', action='store_true', help='using quantiled mask' )
parser.add_argument( '--tot-var', type=float, default=0, help='adding total variation loss' )
#parser.add_argument( '--segment', dest='segment', action='store_true', help='computing acc of segmentation in evaluation' )
parser.add_argument( '--loc', dest='loc', action='store_true', help='validating on imgnet loc' )
parser.add_argument( '--reweight', dest='reweight', action='store_true', help='using random initilization' )
parser.add_argument( '--birdsAdogs', dest='birdsAdogs', action='store_true', help='training on birds and dogs dataset' )
parser.add_argument( '--category', type=str, help='category in object discover dataset' )
parser.add_argument( '--temperature', type=float, default=1, help='temperature in knowledge distillation' )
parser.add_argument( '--adapt-threshold', type=int, default=1, help='adaptive threshold' )
parser.add_argument( '--crop', type=int, default=-1, help='crop in loc, -1 for standard, 0 for groundtruth, and 1 for calculated' )
parser.add_argument( '--wogt', dest='wogt', action='store_true', help='using top1 prediction to train mask net' )
parser.add_argument( '--CAM', dest='CAM', action='store_true', help='using CAM in masknet' )
parser.add_argument( '--new-CAM', dest='new_CAM', action='store_true', help='using new CAM' )
parser.add_argument( '--new-tconv', dest='new_tconv', action='store_true', help='replace tcon with upsample' )

class Env():
    def __init__(self, args):
        self.best_acc = 0
        self.args = args

        if args.evaluation or args.advattack:
            torch.manual_seed(0)
        if args.save_img:
            if not os.path.exists( args.save_img ):
                os.system( "mkdir {}".format(args.save_img) )
        if args.gauss:
            args.noise = True
        if args.hard_threshold_training:
            args.binary = True

        logger = utils.setup_logger( os.path.join( args.save_folder, 'log.log' ) )
        self.logger = logger

        for key, value in sorted( vars(args).items() ):
            logger.info( str(key) + ': ' + str(value) )

        model = getattr(models, args.arch)(large_reg=args.large_reg, quarter_reg=args.quarter,
                                            new_fc=False,
                                            pretrained=(not args.pretrained and not args.resume),
                                            CAM=args.CAM, new_CAM=args.new_CAM,
                                            new_tconv=args.new_tconv)
        if args.reweight:
            model.apply( utils.weight_init )
        if args.birdsAdogs:
            model.cls.fc = nn.Linear( 512, 20 )
            model.cls.fc.apply( utils.weight_init )
        if args.dataset == 'cub200' and False:
            model.cls.fc = nn.Linear( 512, 200 )

        model = torch.nn.DataParallel( model ).cuda()

        if args.pretrained:
            logger.info( '=> using a pre-trained model from {}'.format(args.pretrained) )
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict( checkpoint['model'] )
        else:
            logger.info( '=> initailizing the model, {}, with random weights.'.format(args.arch) )
        if self.args.new_fc:
            model.module.cls.fc = nn.Linear( 512, 200 ).cuda()
        self.model = model

        logger.info( 'Dims: {}'.format( sum([m.data.nelement() if m.requires_grad else 0
            for m in model.parameters()] ) ) )

        if args.dataset != 'chestx':
            if args.dataset == 'pascalvoc':
                self.optimizer_cls = optim.SGD( model.module.cls.fc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
            else:
                self.optimizer_cls = optim.SGD( model.module.cls.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
            self.optimizer_reg = optim.SGD( model.module.reg.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        else:
            self.optimizer_cls = optim.Adam( model.module.cls.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08 )
            self.optimizer_reg = optim.Adam( model.module.reg.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08 )
        if args.dataset not in ['chestx', 'pascalvoc']:
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = utils.WeightedBCELoss()
        self.criterion2 = self.criterion
        if args.dataset == 'pascalvoc' and args.trained_cls:
            self.criterion2 = nn.CrossEntropyLoss().cuda()
        self.entropy = utils.Entropy().cuda()

        self.it = 0
        self.reg_it = 0
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info( '=> loading checkpoint from {}'.format(args.resume) )
                checkpoint = torch.load( args.resume )
                self.it = checkpoint['it']
                self.reg_it = checkpoint['reg_it']
                if 'best_acc' in checkpoint.keys():
                    self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict( checkpoint['model'] )
                self.optimizer_cls.load_state_dict( checkpoint['optimizer_cls'] )
                self.optimizer_reg.load_state_dict( checkpoint['optimizer_reg'] )
                logger.info( '=> loaded checkpoint from {} (iter {})'.format(
                    args.resume, max(self.it, self.reg_it) ) )
            else:
                raise Exception("No checkpoint found. Check your resume path.")

        if args.dataset =='mnist':
            train_dataset = datasets.mnist_dataset( args.data, train=True  )
            valid_dataset = datasets.mnist_dataset( args.data, train=False )
        elif args.dataset == 'cifar10':
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
            
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=train_transform)
            valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=valid_transform)
        elif args.dataset == 'imgnet':
            self.labels = labels
            if args.fb:
                args.data = '/scratch/data/imagenet/'        
            else:
                args.data = '/scratch/datasets/imagenet/'
            if args.birdsAdogs:
                args.data = '/scratch/datasets/imagenet_subset/'
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            if args.myval:
                valdir = args.myval
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            if args.arch == 'VGGC':
                normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                valid_dataset = torchvision.datasets.ImageFolder(
                    valdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        normalize,
                    ]))
            if args.loc:
                valid_dataset = datasets.imagenet_loc_dataset(
                    data_dir=args.data,
                    crop=args.crop,
                    transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        elif args.dataset == 'chestx':
            args.data = '/scratch/datasets/chestx-ray14'
            data_dir = os.path.join( args.data, 'images' )
            train_list_file = os.path.join( args.data, 'train_list.txt' )
            test_list_file = os.path.join( args.data, 'test_list.txt' )
            self.labels = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = datasets.chestx_dataset(
                data_dir=data_dir,
                image_list_file=train_list_file,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_dataset = datasets.chestx_dataset(
                data_dir=data_dir,
                image_list_file=test_list_file,
                transform=transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif args.dataset == 'place365':
            self.labels = labels
            args.data = '/scratch/datasets/place365/'        
            traindir = os.path.join(args.data, 'data_large')
            valdir = os.path.join(args.data, 'val_large')
            if args.myval:
                valdir = args.myval
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif args.dataset == 'cub200':
            self.labels = labels
            args.data = '/scratch/datasets/CUB200/CUB_200_2011'
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = datasets.cub200_dataset(
                data_dir=args.data,
                mode='train',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_dataset = datasets.cub200_dataset(
                data_dir=args.data,
                mode='valid',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif args.dataset == 'pascalvoc':
            args.data = '/scratch/datasets/pascalvoc'
            train_dataset = PascalVOCSegmentation( data_dir=args.data, train=True,
                joint_transform = ComposeJoint([
                    [ResizeAspectRatioPreserve(smaller_side_size=256),
                    ResizeAspectRatioPreserve(smaller_side_size=256, interpolation=Image.NEAREST)],
                    RandomCropJoint(crop_size=(224, 224)),
                    RandomHorizontalFlipJoint(),
                    [transforms.ToTensor(), None],
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ],
                    [None, utils.ToOnehot()], 
                    ]),
                )
            valid_dataset = PascalVOCSegmentation( data_dir=args.data, train=False,
                joint_transform = ComposeJoint([
                    [transforms.ToTensor(), None, None],
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None, None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()),transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ],
                    [None, utils.ToOnehot(), None],
                    ])
                )
            self.labels = train_dataset.CLASS_NAMES
        elif args.dataset == 'object_discover':
            args.data = '/scratch/datasets/object_discover/Data/'
            if args.arch != 'B' and args.arch != 'VGGC':
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                transform_img = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                transform_gt = transforms.ToTensor()

                valid_dataset = datasets.object_discover_dataset(
                    data_dir = args.data, category = args.category,
                    transform_img = transform_img, transform_gt = transform_gt )
                train_dataset = valid_dataset
            else:
                normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                transform_img = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        normalize,
                    ])
                transform_gt = transforms.ToTensor()

                valid_dataset = datasets.object_discover_dataset(
                    data_dir = args.data, category = args.category,
                    transform_img = transform_img, transform_gt = transform_gt )
                train_dataset = valid_dataset
        else:
            raise NotImplementedError('Dataset has not been implemented')

        self.train_loader = data.DataLoader( train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )
        #torch.save(torch.random.get_rng_state(), 'rng_state.data')
        if self.args.evaluation:
            torch.random.set_rng_state( torch.load('rng_state.data') )
        val_batch_size = args.batch_size
        if args.dataset == 'pascalvoc' or args.dataset == 'object_discover':
            val_batch_size = 1
        self.valid_loader = data.DataLoader( valid_dataset,
            batch_size=val_batch_size, shuffle=True, num_workers=args.workers, 
            pin_memory=True, 
            #worker_init_fn=utils.worker_init,
            )
        #if args.gauss:
        self.noise_loader = data.DataLoader( datasets.random_gauss(len(train_dataset), args.gauss), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True )
        self.uniform_loader = data.DataLoader( datasets.random_uniform(len(train_dataset), args.binary), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True )

        self.args = args
        self.save( self.best_acc )

        if args.def_iter == 0:
            args.def_iter = args.tot_iter
        tot_epoch1 = (args.tot_iter - self.it) * args.batch_size // len(train_dataset) + 1
        tot_epoch2 = (args.tot_iter - self.reg_it) * args.batch_size // len(train_dataset) + 1
        if args.trained_cls:
            tot_epoch1 = 0
        if args.dataset == 'pascalvoc':
            if args.evaluation:
                self.pascalvoc_valid()
            else:
                for i in range(tot_epoch1):
                    self.train_cls()
                    if i % 5 == 4:
                        self.save(0)
                for i in range(tot_epoch2):
                    self.pascalvoc_train_reg()
                    if i % 5 == 4:
                        self.save(0)
            exit()
        if args.evaluation:
            self.valid()
        elif args.advattack:
            self.adv_attack()
        elif args.grad_check:
            self.grad_check()
        elif args.single_batch_exp:
            self.single_batch_exp()
        elif args.joint or args.advtrain:
            for i in range(tot_epoch1):
                if args.advtrain:
                    self.advtrain()
                else:
                    self.train_joint()
                self.valid()
                if self.it >= args.tot_iter:
                    exit()
        else:
            for i in range(tot_epoch1):
                self.train_cls()
                if (args.dataset != 'cub200' or i % 5 == 4) and args.dataset != 'pascalvoc':
                    self.valid()
                if self.it >= args.tot_iter:
                    break
                if args.dataset == 'pascalvoc' and i % 5 == 4:
                    self.save( 0 )
            for i in range(tot_epoch2):
                self.train_reg()
                if (args.dataset != 'cub200' or i % 5 == 4) and args.dataset != 'pascalvoc':
                    self.valid()
                if self.reg_it >= args.tot_iter:
                    break
                if args.dataset == 'pascalvoc' and i % 5 == 4:
                    self.save( 0 )

    def save( self, acc ):
        logger = self.logger
        is_best = acc > self.best_acc
        self.best_acc = max( self.best_acc, acc )
        logger.info( '=> saving checkpoint' )
        utils.save_checkpoint({
            'it': self.it,
            'reg_it': self.reg_it,
            'arch': self.args.arch,
            'model': self.model.state_dict(),
            'optimizer_cls': self.optimizer_cls.state_dict(),
            'optimizer_reg': self.optimizer_reg.state_dict(),
        }, is_best, self.args.save_folder)

    def train_cls( self ):
        logger = self.logger
        losses = AverageMeter()

        self.model.train()
        logger.info("Classification Training Epoch")
        for param in self.model.module.cls.parameters():
            param.requires_grad = True
        self.model.module.reg.eval()
        for param in self.model.module.reg.parameters():
            param.requires_grad = False
        
        for i, batch in enumerate(self.train_loader):
            self.it += 1

            if self.it % (self.args.def_iter // 3) == 0 and not self.args.const_lr:
                for group in self.optimizer_cls.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_cls.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()
            if len(gt.size()) == 2 and gt.size(1) == 1:
                gt = gt[:, 0]

            pred0, pred1, mask = self.model( inp, 0 )

            loss1 = self.criterion( pred0, gt )
            loss = loss1
            losses.update( loss.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_cls.step()
            if self.it % self.args.print_freq == 0:
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format( iter=self.it, loss=losses )
                self.logger.info( log_str )
            if self.it >= self.args.tot_iter:
                return

    def pascalvoc_train_reg( self ):
        logger = self.logger
        logger.info("PascalVOC Mask Training Epoch")
        losses = AverageMeter()
        self.model.train()
        self.model.module.cls.eval()
        for param in self.model.module.cls.parameters():
            param.requires_grad = False
        for i, batch in enumerate( self.train_loader ):
            self.reg_it += 1
            if self.reg_it % (self.args.tot_iter // 3) == 0 and not self.args.const_lr:
                for group in self.optimizer_reg.param_groups:
                    group['lr'] *= 0.1
            self.optimizer_reg.zero_grad()
            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()
            index, gt2 = utils.random_select( batch[1] )
            
            pred0, pred0_i, pred1, pred1_i, mask_g, mask = self.model( inp, 1, noise=True, index = index )

            loss2 = self.criterion( pred1, gt2 )
            loss2_g = (torch.nn.Softmax()(pred0_i) * (torch.log(torch.nn.Softmax()(pred0_i + 1e-5)) - torch.log(torch.nn.Softmax()(pred1_i + 1e-5)))).sum(1).mean(0)
            loss = loss2 + loss2_g
            loss += mask_g.mean(0).sum() * self.args.L1
            losses.update( loss.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_reg.step()
            if self.reg_it % self.args.print_freq == 0:
                self.logger.info( "mean={m:.5f}, std={s:.5f}".format(m=mask_g.mean().data[0], s=mask_g.std().data[0]))
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f}), Loss2:{loss1:.5f}, Loss2_g:{loss2:.5f}'.format( iter=self.reg_it, loss=losses, loss1=loss2.data[0], loss2=loss2_g.data[0] )
                self.logger.info( log_str )
            if self.reg_it >= self.args.tot_iter:
                return

    def train_reg( self ):
        logger = self.logger
        logger.info("Mask Training Epoch")
        losses = AverageMeter()
        diff = AverageMeter()
        self.model.train()
        self.model.module.cls.eval()
        for param in self.model.module.cls.parameters():
            param.requires_grad = False
        for i, (batch, noise, UR) in enumerate(zip(self.train_loader, self.noise_loader, self.uniform_loader)):
            self.reg_it += 1
            if self.reg_it % (self.args.tot_iter // 3) == 0 and not self.args.const_lr:
                for group in self.optimizer_reg.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_reg.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            """
            if self.args.dataset == 'pascalvoc':
                index, gt2 = utils.random_select( batch[1] )
            else:
                index = None
            """
            index = None

            if len(gt.size()) == 2 and gt.size(1) == 1:
                gt = gt[:, 0]

            if self.args.gauss:
                noise = Variable(noise).cuda()
            else:
                noise = None
            if self.args.binary:
                UR = Variable(UR).cuda()
            else:
                UR = None

            if not self.args.double:
                pred0, pred1, mask = self.model( inp, 1, self.args.binary, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training, sharp=self.args.sharp_noise, index=index )
                pred2 = None
            else:
                pred0, pred1, pred2, mask = self.model( inp, 1, self.args.binary, single=False, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training , sharp=self.args.sharp_noise, index=index)

            loss1 = self.criterion( pred0, gt )
            if self.args.wogt:
                score0, gt  = torch.max( pred0, 1 )
            if self.args.KL == 0:
                """
                if self.args.dataset == 'pascalvoc':
                    loss2 = self.criterion2( pred1, gt2 )
                else:
                """
                loss2 = self.criterion( pred1, gt )
            else:
                T = self.args.temperature
                loss2 = self.KL_div(pred0, pred1).mean(0) * self.args.KL + self.criterion( pred1, gt ) * (1 - self.args.KL) / T**2
            loss = loss1 + loss2
            if pred2 is not None:
                #loss -= self.criterion( pred2, gt ) * 0.01
                loss -= self.entropy( pred2 ) * 0.1
            loss += mask.mean(0).sum() * self.args.L1
            if self.args.quad > 0:
                loss += (mask * (1-mask)).mean(0).sum() * self.args.quad
            if self.args.tot_var > 0:
                loss += (((mask[:, :, :mask.size(2)-1, :mask.size(3)-1] - mask[:, :, 1:, :mask.size(3)-1])**2 + (mask[:, :, :mask.size(2)-1, :mask.size(3)-1] - mask[:, :, :mask.size(2)-1, 1:])**2 + 1e-5)**0.5).mean(0).sum() * self.args.tot_var
            if self.args.sharp_reg:
                mask_A = (mask > 0.5).type( torch.cuda.FloatTensor )
                mask_B = (mask < 0.5).type( torch.cuda.FloatTensor )
                loss += -(mask * mask_A).mean(0).sum() * self.args.L1 * 0.5 + (mask * mask_B).mean(0).sum() * self.args.L1 * 0.5
            diff0 = loss2 - loss1
            losses.update( loss.data[0], inp.size(0) )
            diff.update( diff0.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_reg.step()
            if self.reg_it % self.args.print_freq == 0:
                self.logger.info( "mean={m:.5f}, std={s:.5f}".format(m=mask.mean().data[0], s=mask.std().data[0]))
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f}), Loss1:{loss1:.5f}, Loss2:{loss2:.5f}, Loss2-Loss1:{dif.val:.5f} ({dif.avg:.5f})'.format( iter=self.reg_it, loss=losses, loss1=loss1.data[0], loss2=loss2.data[0], dif=diff )
                self.logger.info( log_str )
            if self.reg_it >= self.args.tot_iter:
                return

    def KL_div( self, pred0, pred1 ):
        T = self.args.temperature
        return (torch.nn.Softmax()(pred0 / T) * (torch.log(torch.nn.Softmax()(pred0/T + 1e-5)) - torch.log(torch.nn.Softmax()(pred1/T + 1e-5)))).sum(1)

    def train_joint( self ):
        logger = self.logger
        logger.info("Joint Training Epoch")
        losses = AverageMeter()
        diff = AverageMeter()
        self.model.train()
        for i, (batch, noise, UR) in enumerate(zip(self.train_loader, self.noise_loader, self.uniform_loader)):
            self.reg_it += 1
            self.it += 1
            if self.reg_it % (self.args.tot_iter // 3) == 0 and not self.args.const_lr:
                for group in self.optimizer_reg.param_groups:
                    group['lr'] *= 0.1
                for group in self.optimizer_cls.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_reg.zero_grad()
            self.optimizer_cls.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()
            if self.args.gauss:
                noise = Variable(noise).cuda()
            else:
                noise = None
            if self.args.binary:
                UR = Variable(UR).cuda()
            else:
                UR = None

            if not self.args.double:
                pred0, pred1, mask = self.model( inp, 1, self.args.binary, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training, sharp=self.args.sharp_noise, quantiled=self.args.quantiled )
                pred2 = None
            else:
                pred0, pred1, pred2, mask = self.model( inp, 1, self.args.binary, single=False, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training , sharp=self.args.sharp_noise, quantiled=self.args.quantiled)

            loss1 = self.criterion( pred0, gt )
            if not self.args.KL:
                loss2 = self.criterion( pred1, gt )
            else:
                loss2 = (torch.nn.Softmax()(pred0) * (torch.log(torch.nn.Softmax()(pred0)) - torch.log(torch.nn.Softmax()(pred1)))).sum(1).mean(0) * 0.5 + self.criterion( pred1, gt ) * 0.5
            loss = loss1 + loss2
            if pred2 is not None:
                #loss -= self.criterion( pred2, gt ) * 0.01
                loss -= self.entropy( pred2 ) * 0.1
            loss += mask.mean(0).sum() * self.args.L1
            if self.args.sharp_reg:
                mask_A = (mask > 0.5).type( torch.cuda.FloatTensor )
                mask_B = (mask < 0.5).type( torch.cuda.FloatTensor )
                loss += -(mask * mask_A).mean(0).sum() * self.args.L1 * 0.5 + (mask * mask_B).mean(0).sum() * self.args.L1 * 0.5
            diff0 = loss2 - loss1
            loss += diff0**2
            losses.update( loss.data[0], inp.size(0) )
            diff.update( diff0.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_reg.step()
            self.optimizer_cls.step()
            if self.reg_it % self.args.print_freq == 0:
                self.logger.info( "mean={m:.5f}, std={s:.5f}".format(m=mask.mean().data[0], s=mask.std().data[0]))
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f}), Loss1:{loss1:.5f}, Loss2:{loss2:.5f}, Loss2-Loss1:{dif.val:.5f} ({dif.avg:.5f})'.format( iter=self.reg_it, loss=losses, loss1=loss1.data[0], loss2=loss2.data[0], dif=diff )
                self.logger.info( log_str )
            if self.reg_it >= self.args.tot_iter:
                return

    def advtrain( self ):
        logger = self.logger
        losses = AverageMeter()

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            self.it += 1

            if self.it % (self.args.tot_iter // 3) == 0:
                for group in self.optimizer_cls.param_groups:
                    group['lr'] *= 0.1
                for group in self.optimizer_reg.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_cls.zero_grad()
            self.optimizer_reg.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            stage = (self.it % 5 == 0)
            if not self.args.binary:
                pred0, pred1, mask = self.model( inp, stage )
            else:
                pred0, pred1, mask = self.model( inp, stage, True )

            loss1 = self.criterion( pred0, gt )
            loss2 = self.criterion( pred1, gt )
            if stage == 0:
                loss2 *= 0
            loss = loss1 + loss2
            if stage != 0:
                loss += mask.mean(0).sum() * self.args.L1 
            losses.update( loss.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_cls.step()
            if stage > 0:
                self.optimizer_reg.step()
            if self.it % self.args.print_freq == 0:
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f}), Loss1:{loss1:.5f}, Loss2:{loss2:.5f}, Loss2-Loss1:{dif:.5f}'.format( iter=self.it, loss=losses, loss1=loss1.data[0], loss2=loss2.data[0], dif=loss2.data[0]-loss1.data[0] )
                self.logger.info( log_str )

            if self.it >= self.args.tot_iter:
                return

    def single_batch_exp( self ):
        self.model.train()
        class Mask(nn.Module):
            def __init__(self):
                super().__init__()
                mask = torch.ones( 64, 1, 32, 32 )
                self.mask = nn.Parameter( mask, requires_grad=True )

            def forward( self, x ):
                x = x * self.mask.expand( x.size() )
                return x
        mask = Mask()
        mask = mask.cuda()
        optimizer = optim.SGD( mask.parameters(), lr=100, momentum=0.9 )
        prebatch = None
        for i, batch in enumerate(self.train_loader):
            optimizer.zero_grad()
            if prebatch is not None:
                batch = prebatch
            prebatch = batch
            
            inp = Variable(batch[0], requires_grad = True).cuda()
            gt = Variable(batch[1]).cuda()
            
            inp = mask( inp )
            pred0, pred1, _ = self.model( inp, 0 )
            loss = self.criterion( pred0, gt )
            loss.backward()
            optimizer.step()
            print(i)
            print( mask.mask.grad.mean(0) )
            input()
            if i >= 10:
                break
        for pic, img in zip(mask.mask, batch[0]):
            img = img.type(torch.FloatTensor)
            img = img.numpy()
            img = img.transpose( 1, 2, 0 )
            mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
            std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
            img = (img * std + mean) * 255
            img = img.astype(np.uint8)

            pic = pic[0].type( torch.FloatTensor )
            print(pic)
            print(pic.max())
            print(pic.min())
            print(pic.mean())
            pic = pic.data.numpy()
            pic -= pic.min()
            pic /= pic.max()
            pic *= 255
            pic = pic.astype( np.uint8 )
            pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )
            cv2.imshow('x', pic)
            cv2.imshow('y', img)
            cv2.waitKey(0)

    def grad_check( self ):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            self.optimizer_cls.zero_grad()
            self.optimizer_reg.zero_grad()
            
            inp = Variable(batch[0], requires_grad = True).cuda()
            gt = Variable(batch[1]).cuda()

            mask = torch.ones( inp.size(0), 1, inp.size(2), inp.size(3) )
            mask = Variable( mask, requires_grad = True ).cuda()

            inp1 = inp * mask.expand( inp.size() )

            pred0, pred1, _ = self.model( inp1, 0 )

            loss = self.criterion( pred0, gt )
            #loss.backward()
            
            """
            print(loss)
            print(mask.requires_grad)
            print(inp1.requires_grad)
            for param in self.model.module.parameters():
                print( param.grad is not None )
            """
            grad = torch.autograd.grad( outputs=loss, inputs=mask, grad_outputs=torch.ones(1).cuda(), create_graph = True, retain_graph = True, only_inputs = True )[0]
            print(grad.size())
            for pic, img in zip(grad, batch[0]):
                img = img.transpose( 1, 2, 0 )
                mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
                std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
                img = (img * std + mean) * 255
                img = img.astype(np.uint8)

                pic = pic[0]
                print(pic)
                print(pic.mean(), pic.max(), pic.min())
                pic = pic.type( torch.FloatTensor )
                pic = pic.data.numpy()
                pic -= pic.min()
                pic /= pic.max()
                pic *= 255
                pic = pic.astype(np.uint8)
                pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )
                cv2.imshow('x', pic)
                cv2.imshow('y', img)
                cv2.waitKey(0)


    def train_single( self ):
        logger = self.logger
        losses = AverageMeter()

        self.model.train()
        if self.args.no_mask:
            self.model.module.reg.eval()
            wc_m = utils.WeightsCheck( self.model.module.cls )
            self.optimizer = self.optimizer_cls
        else:
            self.model.module.cls.eval()
            wc_m = utils.WeightsCheck( self.model.module.reg )
            self.optimizer = self.optimizer_reg

        for i, batch in enumerate(self.train_loader):
            self.it += 1

            if self.it % (self.args.tot_iter // 3) == 0:
                for group in self.optimizer.param_groups:
                    group['lr'] *= 0.1

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            pred0, pred1, mask = self.model( inp )
            loss1 = self.criterion( pred0, gt )
            loss2 = self.criterion( pred1, gt )
            if self.args.no_mask:
                loss2 *= 0
            loss = (loss1 + loss2)
            if not self.args.no_mask:
                loss += mask.mean()

            losses.update( loss.data[0], inp.size(0) )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.args.no_mask:
                wc_m.check( self.model.module.cls )
            else:
                wc_m.check( self.model.module.reg )

            if self.it % self.args.print_freq == 0:
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val} ({loss.avg}), Loss1:{loss1}, Loss2:{loss2}'.format( iter=self.it, loss=losses, loss1=loss1.data[0], loss2=loss2.data[0] )
                self.logger.info( log_str )

            if self.it >= self.args.tot_iter:
                return

    def toRGB( self, img ):
        if isinstance(img, Variable):
            img = img.type( torch.FloatTensor )
            img = img.data.numpy()
        if len(img.shape) == 2:
            img *= 255
            img = img.astype(np.uint8)
            img = cv2.applyColorMap( img, cv2.COLORMAP_JET )
        elif self.args.dataset == 'mnist':
            img = (img[0] + 0.5) * 255
        elif self.args.dataset == 'cifar10':
            img = img.transpose( 1, 2, 0 )
            mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
            std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
        elif self.args.dataset in ['imgnet','cub200', 'pascalvoc', 'object_discover', 'place365']:
            img = img.transpose( 1, 2, 0 )
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
            img = np.maximum( np.minimum( img, 255 ), 0)
        elif self.args.dataset == 'chestx':
            img = img.transpose( 1, 2, 0 )
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
            img = np.maximum( np.minimum( img, 255 ), 0)
        img = img.astype(np.uint8)
        return img

    def adv_attack( self ):
        self.model.eval()
        logger = self.logger
        for j in range(5):
            j = 2**(j-2)
            j = 8
            fool_rate0 = AverageMeter()
            fool_rate1 = AverageMeter()
            for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
                inp = Variable( batch[0], requires_grad=True ).cuda()
                gt  = Variable( batch[1] ).cuda()
                if self.args.dataset == 'cub200':
                    gt = gt[:, 0]

                pred0, pred1, mask = self.model( inp )
                mask_ori = mask
                loss0 = self.criterion( pred0, gt )
                loss1 = self.criterion( pred1, gt )
                score0, pred0 = torch.max( pred0, 1 )
                score1, pred1 = torch.max( pred1, 1 )

                grad0 = torch.autograd.grad( outputs=loss0, inputs=inp, grad_outputs=torch.ones(1).cuda(), create_graph = True, retain_graph = True, only_inputs = True )[0]
                adv_noise0 = torch.sign( grad0 )
                grad1 = torch.autograd.grad( outputs=loss1, inputs=inp, grad_outputs=torch.ones(1).cuda(), create_graph = True, retain_graph = True, only_inputs = True )[0]
                adv_noise1 = torch.sign( grad1 )

                adv_inp0 = inp + adv_noise0 * j * (1 / 255) / 0.22
                adv_inp1 = inp + adv_noise1 * j * (1 / 255) / 0.22

                img0 = self.toRGB( adv_inp0[0] )
                img1 = self.toRGB( adv_inp1[0] )
                cv2.imshow( 'x', img0 )
                cv2.imshow( 'y', img1 )
                cv2.waitKey(0)

                adv_pred0, _, _ = self.model( adv_inp0 )
                _, adv_pred1, _ = self.model( adv_inp1 )
                score0, adv_pred0 = torch.max( adv_pred0, 1 )
                score1, adv_pred1 = torch.max( adv_pred1, 1 )

                fr0 = (pred0 != adv_pred0).type( torch.FloatTensor ).mean()
                fr1 = (pred1 != adv_pred1).type( torch.FloatTensor ).mean()
                fool_rate0.update( fr0.data[0], inp.size(0) )
                fool_rate1.update( fr1.data[0], inp.size(0) )
            logger.info( "J: {}, Fool_rate0: {}, Fool_rate1: {}".format( j, fool_rate0.avg, fool_rate1.avg ))

    def compute_AUCs(self, gt, pred):
        from sklearn.metrics import roc_auc_score
        AUROCs = []
        gt_np = gt.cpu().data.numpy()
        pred_np = pred.cpu().data.numpy()
        for i in range(14):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return AUROCs

    def compute_APs( self, gt, pred ):
        from sklearn.metrics import average_precision_score 
        APs = []
        gt_np = gt.cpu().data.numpy()
        pred_np = pred.cpu().data.numpy()
        for i in range(pred_np.shape[1]):
            APs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))
        return APs

    def pascalvoc_valid( self ):
        logger = self.logger
        self.model.eval()

        accs0 = AverageMeter()
        accs1 = AverageMeter()

        cnt = 0
        pred0_l = []
        pred1_l = []
        gt_l = []
        bbox_accs = []
        IoUs = [[] for i in range(21)]
        mAPs = []
        for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
            inp = Variable( batch[0], volatile=True ).cuda()
            gt  = Variable( batch[1], volatile=True ).cuda()

            pred0, pred0_i, mask_g, mask = self.model( inp, stage=-1 )
            pred0_l.append( pred0 )
            gt_l.append(gt)

            """
            seg_mask = batch[2][0].numpy()
            pred0 = nn.Sigmoid() (pred0)
            pred0 = pred0 / pred0.sum()
            pred0 = pred0.type(torch.FloatTensor).data.numpy()[0]
            gt = batch[1].numpy()[0]
            print( "Overall **********" )
            for i in range(len(pred0)):
                print( self.labels[i + 1], pred0[i], gt[i] )
            for i in range(21):
                if (seg_mask == i).sum() > 0:
                    obj_mask = (seg_mask == i).astype( np.float32 )
                    obj_mask = Variable( torch.from_numpy( obj_mask ) ).cuda()
                    obj_inp = inp * obj_mask

                    obj_pred0, obj_pred0_i, obj_pred_g, obj_mask = self.model( obj_inp, stage=-1 )
                    obj_pred0 = nn.Sigmoid()( obj_pred0 )
                    obj_pred0 = obj_pred0 / obj_pred0.sum()
                    obj_pred0 = obj_pred0.type( torch.FloatTensor ).data.numpy()
                    obj_pred0 = obj_pred0[0]
                    print( "Mask: {} *******".format( self.labels[i] ) )
                    for i in range(len(obj_pred0)):
                        print( self.labels[i + 1], obj_pred0[i], gt[i] )
                    seg_mask_cv2 = seg_mask * (seg_mask != 255) / 20
                    seg_mask_cv2 = self.toRGB( seg_mask_cv2 )
                    img = self.toRGB( obj_inp[0] )
                    mask_g = 1 - obj_mask[0, -1]
                    mask_g = self.toRGB( mask_g )
                    cv2.imshow( 'x', img )
                    cv2.imshow( 'y', seg_mask_cv2 )
                    cv2.imshow( 'z', mask_g )
                    cv2.waitKey(0)
            """

            if self.args.dataset == 'pascalvoc':
                gt = batch[2][0].numpy()
                pic = mask
                pic = F.upsample( pic, (gt.shape[0], gt.shape[1]), mode = 'bilinear' )
                pic = pic[0].type( torch.FloatTensor ).data.numpy()
                pred = np.zeros( gt.shape )
                gap = np.zeros( gt.shape )
                img = self.toRGB( inp[0] )
                """
                print(batch[1][0].numpy())
                print(pred0.type(torch.FloatTensor).data.numpy())
                """
                gt_label = batch[1][0].numpy()
                for i in range(len(gt_label)):
                    if gt_label[i] == 1:
                        print( self.labels[i+1] )
                l = pred0.type(torch.FloatTensor).data.numpy()[0]
                l = np.argmax(l)
                for i in range(pic.shape[0] - 1):
                    pic_s = self.toRGB( pic[i] )
                    if self.args.print_mask:
                        print(self.labels[i+1])
                        cv2.imshow('x', img)
                        cv2.imshow('y', pic_s)
                        cv2.waitKey(0)
                    mask_i = (pic[i] >= pic[i].max() * 0.8)
                    gap_i = pic[i] - pic[i].max() * 0.8
                    gap_i *= mask_i
                    mask_i_new = (gap_i > gap + 1e-5)
                    pred = pred * (1 - mask_i_new) + (l+1) * mask_i_new
                    gap = np.maximum( gap, gap_i )
                mAPs.append( (pred == gt).sum() / (gt.size - (gt == 255).sum()) )
                #print((pred == gt).sum() / (gt.size - (gt == 255).sum()))
                for i in range(21):
                    A = (pred == i).astype( np.float32 )
                    B = (gt == i).astype( np.float32 )
                    I = A * B
                    U = A + B - I
                    if U.sum() > 0:
                        IoUs[i].append( I.sum() / U.sum() )
                
                gt = gt * (gt != 255) / 20
                pred = pred / 20
                gt = self.toRGB(gt)
                pred = self.toRGB(pred)
                cv2.imshow( 'x', img )
                cv2.imshow( 'y', pred )
                cv2.imshow( 'z', gt )
                pic = self.toRGB( 1-pic[-1] )
                cv2.imshow( 'a', pic )
                cv2.waitKey(0)

                continue

        s = []
        for a in IoUs:
            s.append( np.array(a).mean() )
        print(s)
        print(np.array(s).mean())
        log_str = "VAL FINAL -> Accuracy0: {}, Accuracy1: {}, mAP: {}".format( accs0.avg, accs1.avg, np.array(mAPs).mean() )
        pred0 = torch.cat( pred0_l, 0 )
        gt = torch.cat( gt_l, 0 )
        AUC0 = np.array(self.compute_APs( gt, pred0 ))
        print(AUC0)
        log_str += " mAP0: {}".format( AUC0.mean() )
        logger.info( log_str )
    
    def valid( self ):
        logger = self.logger
        self.model.eval()

        accs0 = AverageMeter()
        accs1 = AverageMeter()

        cnt = 0
        pred0_l = []
        pred1_l = []
        gt_l = []
        bbox_accs = []
        IOUs = []
        mAPs = []
        bbox_dic = {}

        loss_l = [ [] for _ in range(self.args.threshold) ]
        acc1_l = [ [] for _ in range(self.args.threshold) ]
        acc5_l = [ [] for _ in range(self.args.threshold) ]
        cons_l = [ [] for _ in range(self.args.threshold) ]

        acc_a = [ [] for _ in range(self.args.adapt_threshold) ]
        pxl_a = [ [] for _ in range(self.args.adapt_threshold) ]

        for i, (batch, UR) in tqdm.tqdm(enumerate(zip(self.valid_loader, self.uniform_loader))):
            inp = Variable( batch[0], volatile=True ).cuda()
            gt  = Variable( batch[1], volatile=True ).cuda()
            if self.args.dataset == 'cub200':
                gt = gt[:, 0]

            pred0, pred1, mask = self.model( inp )
            pred00, pred10 = pred0, pred1
            if self.args.dataset not in ['chestx', 'pascalvoc', 'object_discover']:
                score0, pred0 = torch.max( pred0, 1 )
                score1, pred1 = torch.max( pred1, 1 )
                acc0 = (pred0 == gt.view(-1)).type( torch.FloatTensor ).mean()
                acc1 = (pred1 == gt.view(-1)).type( torch.FloatTensor ).mean()
                accs0.update( acc0.data[0], inp.size(0) )
                accs1.update( acc1.data[0], inp.size(0) )
            elif args.dataset == 'pascalvoc':
                pred0_l.append( pred0 )
                pred1_l.append( pred1 )
                gt_l.append(gt)
                """
                pred0 = (pred0 > 0.5).type( torch.FloatTensor )
                pred1 = (pred1 > 0.5).type( torch.FloatTensor )
                gt = gt.type( torch.FloatTensor )
                acc0 = (pred0 == gt).type( torch.FloatTensor ).mean()
                acc1 = (pred1 == gt).type( torch.FloatTensor ).mean()
                accs0.update( acc0.data[0], inp.size(0) )
                accs1.update( acc1.data[0], inp.size(0) )
                """
            elif args.dataset == 'chestx':
                pred0_l.append( pred0 )
                pred1_l.append( pred1 )
                gt_l.append(gt)

            if self.args.dataset == 'object_discover':
                mask = mask[0, 0].type( torch.FloatTensor ).data.numpy()
                gt = batch[1][0].numpy()[0]
                if self.args.arch != 'B' and self.args.arch != 'VGGC':
                    mask = cv2.blur( mask, (32, 32) )
                else:
                    mask = cv2.resize( mask, (gt.shape[1], gt.shape[0]) )
                if gt.max() > 0:
                    gt /= gt.max()
                if mask.max() > 0:
                    mask /= mask.max()
                img = inp[0].type( torch.FloatTensor ).data.numpy()
                img = self.toRGB( img )
                mask_cv2 = self.toRGB( mask )
                gt_cv2 = self.toRGB( gt )
                mask = (mask > mask.max() * 0.3).astype( np.float32 )
                gt = (gt != 0).astype( np.float32 )
                I = mask * gt
                U = mask + gt - I
                if gt.sum() == 0:
                    IOUs.append(1)
                    continue
                IoU = I.sum() / U.sum()
                print( IoU )
                IOUs.append( IoU )
                if self.args.print_mask:
                    mask = self.toRGB( mask )
                    if not self.args.save_img:
                        cv2.imshow( 'x', img )
                        cv2.imshow( 'y', mask_cv2 )
                        cv2.imshow( 'z', gt_cv2 )
                        cv2.imshow( 'a', mask )
                        cv2.waitKey(0)
                    else:
                        name = os.path.join( self.args.save_img, self.args.category )
                        cv2.imwrite( name + '_{}_img.png'.format(i), img )
                        cv2.imwrite( name + '_{}_mask0.png'.format(i), mask_cv2 )
                        cv2.imwrite( name + '_{}_gt.png'.format(i), gt_cv2 )
                        cv2.imwrite( name + '_{}_mask1.png'.format(i), mask )
                        
                continue

            if self.args.adapt_threshold != 1:
                criterion = nn.CrossEntropyLoss(reduce=False)
                pred0, pred1, mask = self.model( inp )
                _, label0 = torch.max( pred0, 1 )
                label0 = label0.type( torch.LongTensor ).data.numpy()
                mask_np = mask.type( torch.FloatTensor ).data.numpy()
                sale = [ np.sort(mask_np[i].reshape(-1)) for i in range(mask_np.shape[0]) ]
                index = [0 for i in range(inp.size(0))]
                interval = 2 / self.args.adapt_threshold
                gt = gt.type( torch.LongTensor ).data.numpy()
                for i in range( self.args.adapt_threshold ):
                    p = [ sale[j][ int(len(sale[j]) / self.args.adapt_threshold * i) ] for j in range(len(sale)) ]
                    mask_b = [ (mask[None, j] > float(p[j])).type( torch.cuda.FloatTensor ) for j in range(len(sale)) ]
                    #print( mask_b )
                    mask_b = torch.cat( mask_b, 0 )
                    inp_b = inp * mask_b.expand( inp.size() )
                    pred1, pred1, _ = self.model( x=inp_b, stage=0 )
                    loss = self.KL_div( pred0, pred1 )
                    loss = loss.type( torch.FloatTensor ).data.numpy()
                    _, label1 = torch.max( pred1, 1 )
                    label1 = label1.type( torch.LongTensor ).data.numpy()
                    _, label1_5 = pred1.topk( 5, 1, True, True )
                    label1_5 = label1_5.type( torch.LongTensor ).data.numpy()
                    for j in range( inp.size(0) ):
                        if loss[j] < index[j] * interval:
                            continue
                        upp = min(int( loss[j] / interval ) + 1, self.args.adapt_threshold)
                        for k in range( index[j], upp ):
                            acc_a[k].append( gt[j] in label1_5[j] )
                            pxl_a[k].append( 1 - i / self.args.adapt_threshold )
                        index[j] = upp

            if self.args.dataset != 'chestx':
                dtype = torch.LongTensor
            else:
                dtype = torch.FloatTensor
            pred0 = pred0.type( dtype ).data.numpy()
            pred1 = pred1.type( dtype ).data.numpy()
            #gt = gt.type( torch.LongTensor ).data.numpy()
            if self.args.threshold != 1:
                """
                _, _, _, CAM_mask = self.model( inp, CAM=True )
                mask *= CAM_mask
                """
                mask_np = mask.type( torch.FloatTensor ).data.numpy()
                """
                mask_np = [cv2.blur(mask_np[i], (32, 32)) for i in range(len(mask_np))]
                mask_np = np.array(mask_np)
                mask = Variable(torch.from_numpy(mask_np)).cuda()
                """
                sale = [ np.sort(mask_np[i].reshape(-1)) for i in range(len(mask_np)) ]
                """
                inp = inp[0:1]
                gt = gt[0:1]
                mask = mask[0:1]
                label = pred0[0]
                """
                flag = True
                loss_list = []
                acc1_list = []
                acc5_list = []
                pred = pred0
                for i in range( self.args.threshold ):
                    p = [ sale[j][ int(len(sale[j]) / self.args.threshold * i) ] for j in range(len(sale)) ]
                    mask_b = [ (mask[None, j] >= float(p[j])).type( torch.cuda.FloatTensor ) for j in range(len(sale)) ]
                    #print( mask_b )
                    mask_b = torch.cat( mask_b, 0 )
                    inp_b = inp * mask_b.expand( inp.size() )
                    pred0, pred1, _ = self.model( x=inp_b, stage=0 )
                    loss = self.criterion( pred0, gt )
                    loss_list.append( loss.data[0] )

                    prec1, prec5 = utils.accuracy( pred0, gt, topk=(1, 5) )
                    acc1_list.append( prec1[0].data[0] )
                    acc5_list.append( prec5[0].data[0] )
                    score0, pred0 = torch.max( pred0, 1 )
                    pred0 = pred0.type( torch.FloatTensor ).data.numpy()
                    cons_l[i].append( (pred0 == pred).mean() )
                    #acc_list.append( (pred0 == gt).type( torch.FloatTensor ).mean() )
                    """
                    if pred0[0] != label and flag:
                        flag = False
                        #pred0 = pred0.type( torch.LongTensor ).data.numpy()
                        #print("full:{}, pred:{}, gt:{}".format( label, pred0[0], gt[0] ))
                        mask_b = mask[0, 0]
                        mask_b = mask_b.type(torch.FloatTensor).data.numpy()
                        mask_b *= 255
                        mask_b = mask_b.astype( np.uint8 )
                        mask_b = cv2.applyColorMap( mask_b, cv2.COLORMAP_JET )

                        img = self.toRGB( inp[0] )
                        img1 = self.toRGB( inp_b[0] )
                        if not self.args.save_img:
                            cv2.imshow('x', mask_b)
                            cv2.imshow('y', img)
                            cv2.imshow('z', img1)
                            cv2.waitKey(0)
                        else:
                            name = '{}/{}_GT({})_PRED({})'.format(self.args.save_img, cnt, self.labels[int(gt[0])], self.labels[pred0[0]])
                            cv2.imwrite( '{}_inp.png'.format(name), img )
                            cv2.imwrite( '{}_mask.png'.format(name), mask_b )
                            cv2.imwrite( '{}_masked_inp.png'.format(name), img1 )
                    """
                for i in range( len(loss_list) ):
                    loss_l[i].append( loss_list[i] )
                    acc1_l[i].append( acc1_list[i] )
                    acc5_l[i].append( acc5_list[i] )
                """
                acc_list = np.array(acc_list).astype( np.float64 )
                loss_list = np.array(loss_list)
                acc_list *= loss_list.max()
                #plt.figure(cnt)
                plt.plot( range(len(loss_list)), loss_list, label='Loss function' )
                plt.plot( range(len(acc_list)), acc_list, label='Correct' )
                plt.grid(True)
                plt.legend()
                plt.xlabel("Removing Percentage")
                if not self.args.save_img:
                    plt.show()
                else:
                    plt.savefig( 'images_imgnet/{}_plot.png'.format(name) )
                plt.close()
                """


            if self.args.print_mask:
                mask = mask.type( torch.FloatTensor )
                mask = mask.data.numpy()
                inp = inp.type( torch.FloatTensor )
                inp = inp.data.numpy()
                for pic, img, j in zip(mask, inp, range(mask.shape[0])):
                    if cnt == 1000:
                        return
                    print("pred0:{}, pred1:{}, gt:{}".format( pred0[j], pred1[j], gt[j].type(torch.FloatTensor).data.numpy() ))
                    img = self.toRGB( img )
                    pic = pic[0]
                    if self.args.bbox:
                        pic1 = cv2.blur( pic, (32, 32) )
                        x, y, w, h = utils.bbox_generator( pic1, float(pic1.max())*0.2 )
                    pic /= pic.max()
                    pic_mask = (pic >= 0.2)
                    pic *= 255
                    pic = pic.astype( np.uint8 )
                    pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )
                    pic_mask = pic_mask[:, :, None]
                    img1 = pic.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.3
                    img1 = img1.astype(np.uint8)

                    img = img.astype( np.uint8 )

                    if self.args.arch == 'VGGC':
                        pic = cv2.resize( pic, (224, 224) )
                        img = cv2.resize( img, (224, 224) )
                        img1 = cv2.resize( img1, (224, 224) )

                    if self.args.bbox:
                        img, pic, img1 = img.copy(), pic.copy(), img1.copy()
                        cv2.rectangle( img, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        cv2.rectangle( pic, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        cv2.rectangle( img1, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        if self.args.dataset == 'cub200' or self.args.loc:
                            bbox = batch[2][j].numpy()
                            x, y, w, h = bbox
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            cv2.rectangle( img, (x, y), (x+w, y+h), (0, 255, 0), 2 )
                            cv2.rectangle( pic, (x, y), (x+w, y+h), (0, 255, 0), 2 )
                            cv2.rectangle( img1, (x, y), (x+w, y+h), (0, 255, 0), 2 )

                    if not self.args.save_img:
                        cv2.imshow( 'x', pic )
                        cv2.imshow( 'y', img )
                        cv2.imshow( 'z', img1 )
                        cv2.waitKey(0)
                    else:
                        name = '{}/{}_GT({})_PRED({})'.format(self.args.save_img, cnt, self.labels[int(gt[j])], self.labels[int(pred1[j])])
                        cv2.imwrite( '{}_inp.png'.format(name), img )
                        cv2.imwrite( '{}_mask.png'.format(name), pic )
                        cv2.imwrite( '{}_masked_inp.png'.format(name), img1 )
                    cnt += 1

            if self.args.bbox and self.args.dataset == 'cub200' and not self.args.print_mask:
                mask = mask.type( torch.FloatTensor )
                mask = mask.data.numpy()
                for pic, bbox in zip(mask, batch[2]):
                    pic = pic[0]
                    x_p, y_p, w_p, h_p = utils.bbox_generator( pic, float(pic.mean()) )
                    x_g, y_g, w_g, h_g = bbox.numpy()
                    IOU = utils.IOU( (x_p, y_p, x_p+w_p, y_p+h_p), (x_g, y_g, x_g+w_g, y_g+h_g) )
                    if IOU > 0.5:
                        bbox_accs.append(1)
                    else:
                        bbox_accs.append(0)
                    IOUs.append(IOU)

            if self.args.loc and not self.args.print_mask:
                mask = mask.type( torch.FloatTensor )
                mask = mask.data.numpy()
                pred0, pred1 = pred00, pred10
                pred0 = pred0.type( torch.FloatTensor ).data.numpy()
                pred1 = pred1.type( torch.FloatTensor ).data.numpy()
                for img, pic, m, pred, pred_1, gt, index in zip(inp, mask, batch[2], pred0, pred1, batch[1], batch[3]):
                    pic = pic[0]
                    pic = cv2.blur( pic, (32, 32) )
                    x_p, y_p, w_p, h_p = utils.bbox_generator( pic, float(np.max(pic)) * 0.2 )
                    x_g, y_g, w_g, h_g = m.numpy()
                    IoU = utils.IOU( (x_p, y_p, x_p+w_p, y_p+h_p), (x_g, y_g, x_g+w_g, y_g+h_g) )

                    """
                    x_p, y_p, w_p, h_p = int(x_p), int(y_p), int(w_p), int(h_p)
                    x_g, y_g, w_g, h_g = int(x_g), int(y_g), int(w_g), int(h_g)
                    pic0 = (pic >= np.mean(pic)).astype( np.float32 )
                    pic0 = self.toRGB( pic0 )
                    img = self.toRGB( img )
                    pic = self.toRGB( pic )
                    img = img.copy()
                    pic = pic.copy()
                    cv2.rectangle( img, (x_p, y_p), (x_p+w_p, y_p+h_p), (0, 255, 0), 2 )
                    cv2.rectangle( img, (x_g, y_g), (x_g+w_g, y_g+h_g), (0, 0, 255), 2 )
                    cv2.rectangle( pic, (x_p, y_p), (x_p+w_p, y_p+h_p), (0, 255, 0), 2 )
                    cv2.rectangle( pic, (x_g, y_g), (x_g+w_g, y_g+h_g), (0, 0, 255), 2 )
                    print( IoU )
                    cv2.imshow('x', pic0)
                    cv2.imshow('z', img)
                    cv2.imshow('y', pic)
                    cv2.waitKey(0)
                    continue
                    """

                    gt = int(gt[0])
                    index = int(index[0])
                    bbox_dic[index] = (x_p, y_p, w_p, h_p)
                    pred = pred
                    s = list(reversed(sorted(pred)))
                    if pred[gt] < s[4]:
                        IoU = 0
                    if IoU >= 0.5:
                        bbox_accs.append(1)
                    else:
                        bbox_accs.append(0)
                    if pred[gt] < s[0]:
                        IoU = 0
                    if IoU >= 0.5:
                        IOUs.append( 1 )
                    else:
                        IOUs.append( 0 )

        if self.args.dataset == 'object_discover':
            IOUs = np.array(IOUs).mean()
            print("IoU = {}".format(IOUs))
            exit()
        if self.args.adapt_threshold != 1:
            acc_a = [ np.array(_).mean() for _ in acc_a ]
            pxl_a = [ np.array(_).mean() for _ in pxl_a ]
            logger.info( "Acc list {}".format( acc_a ) )
            logger.info( "Pxl remaining list {}".format( pxl_a ) )
        if self.args.threshold != 1:
            loss_l = [ np.array(_).mean() for _ in loss_l ]
            acc1_l = [ np.array(_).mean() for _ in acc1_l ]
            acc5_l = [ np.array(_).mean() for _ in acc5_l ]
            cons_l = [ np.array(_).mean() for _ in cons_l ]
            logger.info( "Loss list {}".format( loss_l ) )
            logger.info( "Acc1 list {}".format( acc1_l ) )
            logger.info( "Acc5 list {}".format( acc5_l ) )
            logger.info( "Consist list {}".format( cons_l ) )
            with open("curve/{}".format(self.args.save_img), "wb") as f:
                pickle.dump( cons_l, f )
        if self.args.dataset not in ['chestx', 'pascalvoc']:
            log_str = "VAL FINAL -> Accuracy0: {}, Accuracy1: {}".format( accs0.avg, accs1.avg )
        elif self.args.dataset == 'pascalvoc':
            log_str = "VAL FINAL -> Accuracy0: {}, Accuracy1: {}, mAP: {}".format( accs0.avg, accs1.avg, np.array(mAPs).mean() )
            pred0 = torch.cat( pred0_l, 0 )
            pred1 = torch.cat( pred1_l, 0 )
            gt = torch.cat( gt_l, 0 )
            AUC0 = np.array(self.compute_APs( gt, pred0 ))
            AUC1 = np.array(self.compute_APs( gt, pred1 ))
            print(AUC0, AUC1)
            log_str += " mAP0: {}, mAP1: {}".format( AUC0.mean(), AUC1.mean() )
        else:
            pred0 = torch.cat( pred0_l, 0 )
            pred1 = torch.cat( pred1_l, 0 )
            gt = torch.cat( gt_l, 0 )
            AUC0 = np.array(self.compute_AUCs( gt, pred0 ))
            AUC1 = np.array(self.compute_AUCs( gt, pred1 ))
            print(AUC0, AUC1)
            log_str = "VAL FINAL -> AUC0: {}, AUC1: {}".format( AUC0.mean(), AUC1.mean() )
            accs1.avg = AUC1.mean()
        logger.info( log_str )
        if (self.args.bbox and self.args.dataset == 'cub200') or self.args.loc:
            log_str = "IOU top 1 acc: {}, IOU top 5 acc: {}".format( np.array(IOUs).mean(), np.array(bbox_accs).mean())
            logger.info( log_str )
            pickle.dump( bbox_dic, open('1.data', 'wb') )
        self.save( accs1.avg )

if __name__ == '__main__':
    args = parser.parse_args()
    Env( args )
