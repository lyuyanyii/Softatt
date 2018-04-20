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

model_names = ['A', 'B', 'C', 'CDNet', 'PRNet']

parser = argparse.ArgumentParser( description='Low Supervised Semantic Segmentation' )

parser.add_argument( '--arch', metavar='ARCH', choices=model_names )
parser.add_argument( '--save-folder', type=str, metavar='PATH' )
parser.add_argument( '--lr', type=float, help='initial learning rate' )
#parser.add_argument( '--lr-step', type=float, help='lr will be decayed at these steps' )
#parser.add_argument( '--lr-decay', type=float, help='lr decayed rate' )
parser.add_argument( '--data', type=str, help='the directory of data' )
parser.add_argument( '--dataset', type=str, choices=['mnist', 'cifar10', 'imgnet', 'chestx', 'place365', 'cub200'] )
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
parser.add_argument( '--KL', type=float, default=0, action='store_true', help='using KL-divergence loss' )
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
                                            new_fc=(args.dataset=='cub200'),)

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
            self.optimizer_cls = optim.SGD( model.module.cls.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
            self.optimizer_reg = optim.SGD( model.module.reg.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        else:
            self.optimizer_cls = optim.Adam( model.module.cls.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08 )
            self.optimizer_reg = optim.Adam( model.module.reg.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08 )
        if args.dataset != 'chestx':
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = utils.WeightedBCELoss()
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
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif args.dataset == 'place356':
            self.labels = labels
            args.data = '/scratch/data/place365/'        
            traindir = os.path.join(args.data, 'data_large')
            valdir = os.path.join(args.data, 'test_large')
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

        else:
            raise NotImplementedError('Dataset has not been implemented')

        self.train_loader = data.DataLoader( train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )
        #torch.save(torch.random.get_rng_state(), 'rng_state.data')
        if self.args.evaluation:
            torch.random.set_rng_state( torch.load('rng_state.data') )
        self.valid_loader = data.DataLoader( valid_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, 
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
        tot_epoch1 = (args.def_iter - self.it) * args.batch_size // len(train_dataset) + 1
        tot_epoch2 = (args.tot_iter - self.reg_it) * args.batch_size // len(train_dataset) + 1
        if args.trained_cls:
            tot_epoch1 = 0
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
                if args.dataset != 'cub200' or i % 5 == 4:
                    self.valid()
                if self.it >= args.def_iter:
                    break
            for i in range(tot_epoch2):
                self.train_reg()
                if args.dataset != 'cub200' or i % 5 == 4:
                    self.valid()
                if self.reg_it >= args.tot_iter:
                    break

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
            if len(gt.size()) == 2:
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
            if self.it >= self.args.def_iter:
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
            if len(gt.size()) == 2:
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
                pred0, pred1, mask = self.model( inp, 1, self.args.binary, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training, sharp=self.args.sharp_noise )
                pred2 = None
            else:
                pred0, pred1, pred2, mask = self.model( inp, 1, self.args.binary, single=False, noise=self.args.noise, gauss=self.args.gauss, R=noise, UR=UR, noise_rate=self.args.noise_rate, hard_threshold=self.args.hard_threshold_training , sharp=self.args.sharp_noise)

            loss1 = self.criterion( pred0, gt )
            if self.args.KL == 0:
                loss2 = self.criterion( pred1, gt )
            else:
                loss2 = (torch.nn.Softmax()(pred0) * (torch.log(torch.nn.Softmax()(pred0 + 1e-5)) - torch.log(torch.nn.Softmax()(pred1 + 1e-5)))).sum(1).mean(0) * self.args.KL + self.criterion( pred1, gt ) * (1 - self.args.KL)
            loss = loss1 + loss2
            if pred2 is not None:
                #loss -= self.criterion( pred2, gt ) * 0.01
                loss -= self.entropy( pred2 ) * 0.1
            loss += mask.mean(0).sum() * self.args.L1
            loss += (mask * (1-mask)).mean(0).sum() * self.args.quad
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
        elif self.args.dataset == 'imgnet' or self.args.dataset == 'cub200':
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
        for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
            inp = Variable( batch[0], requires_grad=True ).cuda()
            gt  = Variable( batch[1] ).cuda()

            pred0, pred1, mask = self.model( inp )
            mask_ori = mask
            loss = self.criterion( pred0, gt )
            score0, pred0 = torch.max( pred0, 1 )
            score1, pred1 = torch.max( pred1, 1 )

            if int(pred0[0]) != int(gt[0]):
                continue

            #loss.backward()
            grad = torch.autograd.grad( outputs=loss, inputs=inp, grad_outputs=torch.ones(1).cuda(), create_graph = True, retain_graph = True, only_inputs = True )[0]
            adv_noise = torch.sign( grad )
            print(adv_noise)
            max_c = 0.1
            num_its = 20
            flag0, flag1 = False, False
            for j in range(num_its):
                adv_inp = inp + adv_noise * max_c * (j / num_its)
                pred0, pred1, mask = self.model( adv_inp )
                score0, pred0 = torch.max( pred0, 1 )
                score1, pred1 = torch.max( pred1, 1 )

                if not flag0 and int(pred0[0]) != int(gt[0]):
                    flag0 = True
                    print('Ori attack succeeds with lambda = {:.3f}.'.format(max_c * (j / num_its)))
                    print( self.labels[int(pred0[0])], "***", self.labels[int(gt[0])] )
                    img = self.toRGB( adv_inp[0] )
                    mask_b = self.toRGB( mask[0, 0] )
                    mask_c = self.toRGB( mask_ori[0, 0] )
                    cv2.imshow('x', img)
                    cv2.imshow('y', mask_b)
                    cv2.imshow('z', mask_c)
                    cv2.waitKey(0)
                if flag0 and not flag1 and int(pred1[0]) != int(gt[0]):
                    flag1 = True
                    print('Mask attack succeeds with lambda = {:.3f}.'.format(max_c * (j / num_its)))
                    print( self.labels[int(pred0[0])], "***", self.labels[int(gt[0])] )
                    img = self.toRGB( adv_inp[0] )
                    mask_b = self.toRGB( mask[0, 0] )
                    mask_c = self.toRGB( mask_ori[0, 0] )
                    cv2.imshow('x', img)
                    cv2.imshow('y', mask_b)
                    cv2.imshow('z', mask_c)
                    cv2.waitKey(0)
            if not flag0 or not flag1:
                print('Adv attack fails')

    def compute_AUCs(self, gt, pred):
        from sklearn.metrics import roc_auc_score
        AUROCs = []
        gt_np = gt.cpu().data.numpy()
        pred_np = pred.cpu().data.numpy()
        for i in range(14):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return AUROCs

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
        for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
            inp = Variable( batch[0], volatile=True ).cuda()
            gt  = Variable( batch[1], volatile=True ).cuda()
            if self.args.dataset == 'cub200':
                gt = gt[:, 0]

            pred0, pred1, mask = self.model( inp )
            if self.args.dataset != 'chestx':
                score0, pred0 = torch.max( pred0, 1 )
                score1, pred1 = torch.max( pred1, 1 )
                acc0 = (pred0 == gt).type( torch.FloatTensor ).mean()
                acc1 = (pred1 == gt).type( torch.FloatTensor ).mean()
                accs0.update( acc0.data[0], inp.size(0) )
                accs1.update( acc1.data[0], inp.size(0) )
            else:
                pred0_l.append( pred0 )
                pred1_l.append( pred1 )
                gt_l.append(gt)

            if self.args.dataset != 'chestx':
                dtype = torch.LongTensor
            else:
                dtype = torch.FloatTensor
            pred0 = pred0.type( dtype ).data.numpy()
            pred1 = pred1.type( dtype ).data.numpy()
            #gt = gt.type( torch.LongTensor ).data.numpy()
            if self.args.threshold != 1 and cnt <= 300:
                cnt += 1
                sale = mask[0].type( torch.FloatTensor )
                sale = sale.data.numpy().reshape(-1)
                sale = sorted(sale)
                inp = inp[0:1]
                gt = gt[0:1]
                mask = mask[0:1]
                label = pred0[0]
                flag = True
                loss_list = []
                acc_list = []
                for i in range( self.args.threshold ):
                    p = sale[ int(len(sale) / self.args.threshold * i) ]
                    mask_b = (mask > float(p)).type( torch.cuda.FloatTensor )
                    #print( mask_b )
                    inp_b = inp * mask_b.expand( inp.size() )
                    pred0, pred1, _ = self.model( x=inp_b, stage=0 )
                    loss = self.criterion( pred0, gt )
                    loss_list.append( loss.data[0] )
                    score0, pred0 = torch.max( pred0, 1 )
                    pred0 = pred0.type( torch.LongTensor ).data.numpy()
                    acc_list.append( pred0[0] == int(gt[0]) )
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


            if self.args.print_mask:
                mask = mask.type( torch.FloatTensor )
                mask = mask.data.numpy()
                inp = inp.type( torch.FloatTensor )
                inp = inp.data.numpy()
                for pic, img, j in zip(mask, inp, range(mask.shape[0])):
                    if cnt == 300:
                        return
                    print("pred0:{}, pred1:{}, gt:{}".format( pred0[j], pred1[j], gt[j].type(torch.FloatTensor).data.numpy() ))
                    img = self.toRGB( img )
                    pic = pic[0]
                    if self.args.bbox:
                        x, y, w, h = utils.bbox_generator( pic, float(pic.max()) * 0.5 )
                    pic *= 255
                    pic = pic.astype( np.uint8 )
                    pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )
                    img1 = pic.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.3
                    img1 = img1.astype(np.uint8)

                    img = img.astype( np.uint8 )

                    if self.args.bbox:
                        img, pic, img1 = img.copy(), pic.copy(), img1.copy()
                        cv2.rectangle( img, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        cv2.rectangle( pic, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        cv2.rectangle( img1, (x, y), (x+w, y+h), (0, 0, 255), 2 )
                        if self.args.dataset == 'cub200':
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
                    x_p, y_p, w_p, h_p = utils.bbox_generator( pic, float(pic.max()) * 0.5 )
                    x_g, y_g, w_g, h_g = bbox.numpy()
                    IOU = utils.IOU( (x_p, y_p, x_p+w_p, y_p+h_p), (x_g, y_g, x_g+w_g, y_g+h_g) )
                    if IOU > 0.5:
                        bbox_accs.append(1)
                    else:
                        bbox_accs.append(0)
                    IOUs.append(IOU)

                    
        if self.args.dataset != 'chestx':
            log_str = "VAL FINAL -> Accuracy0: {}, Accuracy1: {}".format( accs0.avg, accs1.avg )
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
        if self.args.bbox and self.args.dataset == 'cub200':
            log_str = "IOU: {}, IOU Acc: {}".format( np.array(IOUs).mean(), np.array(bbox_accs).mean())
            logger.info( log_str )
        self.save( accs1.avg )

if __name__ == '__main__':
    args = parser.parse_args()
    Env( args )
