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

model_names = ['A', 'B', 'C']

parser = argparse.ArgumentParser( description='Low Supervised Semantic Segmentation' )

parser.add_argument( '--arch', metavar='ARCH', choices=model_names )
parser.add_argument( '--save-folder', type=str, metavar='PATH' )
parser.add_argument( '--lr', type=float, help='initial learning rate' )
#parser.add_argument( '--lr-step', type=float, help='lr will be decayed at these steps' )
#parser.add_argument( '--lr-decay', type=float, help='lr decayed rate' )
parser.add_argument( '--data', type=str, help='the directory of data' )
parser.add_argument( '--dataset', type=str, choices=['mnist', 'cifar10', 'imgnet'] )
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
parser.add_argument( '--save-img', dest='save_img', action='store_true' )
parser.add_argument( '--double', dest='double', action='store_true', help='using mask & 1-mask to compute 2 losses' )

class Env():
    def __init__(self, args):
        self.best_acc = 0
        self.args = args

        logger = utils.setup_logger( os.path.join( args.save_folder, 'log.log' ) )
        self.logger = logger

        for key, value in sorted( vars(args).items() ):
            logger.info( str(key) + ': ' + str(value) )

        model = getattr(models, args.arch)()

        model = torch.nn.DataParallel( model ).cuda()

        if args.pretrained:
            logger.info( '=> using a pre-trained model from {}'.format(args.pretrained) )
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict( checkpoint['model'] )
        else:
            logger.info( '=> initailizing the model, {}, with random weights.'.format(args.arch) )
        self.model = model

        logger.info( 'Dims: {}'.format( sum([m.data.nelement() if m.requires_grad else 0
            for m in model.parameters()] ) ) )

        self.optimizer_cls = optim.SGD( model.module.cls.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        self.optimizer_reg = optim.SGD( model.module.reg.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        self.criterion = nn.CrossEntropyLoss().cuda()
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
            args.data = '/scratch/datasets/imagenet/'
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
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
        else:
            raise NotImplementedError('Dataset has not been implemented')

        self.train_loader = data.DataLoader( train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )
        self.valid_loader = data.DataLoader( valid_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )

        self.args = args
        self.save( self.best_acc )

        tot_epoch1 = (args.tot_iter - self.it) * args.batch_size // len(train_dataset) + 1
        tot_epoch2 = (args.tot_iter - self.reg_it) * args.batch_size // len(train_dataset) + 1
        if args.trained_cls:
            tot_epoch1 = 0
        if args.evaluation:
            self.valid()
        elif args.grad_check:
            self.grad_check()
        elif args.single_batch_exp:
            self.single_batch_exp()
        elif args.joint or args.advtrain:
            for i in range(tot_epoch1):
                if args.advtrain:
                    self.advtrain()
                else:
                    self.train_cls()
                    self.train_reg()
                self.valid()
                if self.it >= args.tot_iter:
                    exit()
        else:
            for i in range(tot_epoch1):
                self.train_cls()
                self.valid()
                if self.it >= args.tot_iter:
                    break
            for i in range(tot_epoch2):
                self.train_reg()
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
        
        for i, batch in enumerate(self.train_loader):
            self.it += 1

            if self.it % (self.args.tot_iter // 3) == 0:
                for group in self.optimizer_cls.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_cls.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

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

    def train_reg( self ):
        model.module.reg.apply( weight_init )
        logger = self.logger
        logger.info("Mask Training Epoch")
        losses = AverageMeter()
        diff = AverageMeter()
        self.model.train()
        self.model.module.cls.eval()
        for param in self.model.module.cls.parameters():
            param.requires_grad = False
        for i, batch in enumerate(self.train_loader):
            self.reg_it += 1
            if self.reg_it % (self.args.tot_iter // 3) == 0:
                for group in self.optimizer_reg.param_groups:
                    group['lr'] *= 0.1

            self.optimizer_reg.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            if not self.args.double:
                pred0, pred1, mask = self.model( inp, 1, self.args.binary )
                pred2 = None
            else:
                pred0, pred1, pred2, mask = self.model( inp, 1, self.args.binary, single=False )

            loss1 = self.criterion( pred0, gt )
            loss2 = self.criterion( pred1, gt )
            loss = loss1 + loss2
            if pred2 is not None:
                #loss -= self.criterion( pred2, gt ) * 0.01
                loss -= self.entropy( pred2 ) * 0.1
            loss += mask.mean(0).sum() * self.args.L1
            diff0 = loss2 - loss1
            losses.update( loss.data[0], inp.size(0) )
            diff.update( diff0.data[0], inp.size(0) )
            loss.backward()
            self.optimizer_reg.step()
            if self.reg_it % self.args.print_freq == 0:
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

    def valid( self ):
        logger = self.logger
        self.model.eval()

        accs0 = AverageMeter()
        accs1 = AverageMeter()

        cnt = 0
        for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
            inp = Variable( batch[0], volatile=True ).cuda()
            gt  = Variable( batch[1], volatile=True ).cuda()

            #print("AAAAA")
            if not self.args.binary:
                pred0, pred1, mask = self.model( inp )
            else:
                pred0, pred1, mask = self.model( inp, stage=1, binary=self.args.binary )
            score0, pred0 = torch.max( pred0, 1 )
            score1, pred1 = torch.max( pred1, 1 )
            acc0 = (pred0 == gt).type( torch.FloatTensor ).mean()
            acc1 = (pred1 == gt).type( torch.FloatTensor ).mean()
            accs0.update( acc0.data[0], inp.size(0) )
            accs1.update( acc1.data[0], inp.size(0) )

            pred0 = pred0.type( torch.LongTensor ).data.numpy()
            pred1 = pred1.type( torch.LongTensor ).data.numpy()
            gt = gt.type( torch.LongTensor ).data.numpy()
            if self.args.print_mask:
                mask = mask.type( torch.FloatTensor )
                mask = mask.data.numpy()
                inp = inp.type( torch.FloatTensor )
                inp = inp.data.numpy()
                for pic, img, j in zip(mask, inp, range(mask.shape[0])):
                    if cnt == 300:
                        break
                    #if gt[j] == pred0[j]:
                        continue
                    print("pred0:{}, pred1:{}, gt:{}".format( pred0[j], pred1[j], gt[j] ))
                    if self.args.dataset == 'mnist':
                        img = (img[0] + 0.5) * 255
                    elif self.args.dataset == 'cifar10':
                        img1 = img * pic[0]
                        img = img.transpose( 1, 2, 0 )
                        img1 = img1.transpose( 1, 2, 0 )
                        mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
                        std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
                        img = (img * std + mean) * 255
                        img1 = (img1 * std + mean) * 255
                        img1 = img1.astype( np.uint8 )
                    elif self.args.dataset == 'imgnet':
                        img1 = img * pic[0]
                        img = img.transpose( 1, 2, 0 )
                        img1 = img1.transpose( 1, 2, 0 )
                        mean = np.array([0.485, 0.456, 0.406])
                        std  = np.array([0.229, 0.224, 0.225])
                        img = (img * std + mean) * 255
                        img1 = (img1 * std + mean) * 255
                        img1 = img1.astype( np.uint8 )
                    pic = pic[0]
                    print(pic)
                    #pic /= pic.max()
                    #pic = (pic > pic.mean()).astype( np.float32 )
                    pic *= 255
                    pic = pic.astype( np.uint8 )
                    pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )

                    img = img.astype( np.uint8 )
                    if not self.args.save_img:
                        cv2.imshow( 'x', pic )
                        cv2.imshow( 'y', img )
                        cv2.imshow( 'z', img1 )
                        cv2.waitKey(0)
                    else:
                        cv2.imwrite( 'images_wrong/{}_gt{}_pred{}_inp.png'.format(cnt, gt[j], pred0[j]), img )
                        cv2.imwrite( 'images_wrong/{}_gt{}_pred{}_mask.png'.format(cnt, gt[j], pred0[j]), pic )
                        cv2.imwrite( 'images_wrong/{}_gt{}_pred{}_masked_inp.png'.format(cnt, gt[j], pred0[j]), img1 )
                    cnt += 1
                    
        log_str = "VAL FINAL -> Accuracy0: {}, Accuracy1: {}".format( accs0.avg, accs1.avg )
        logger.info( log_str )
        self.save( accs1.avg )

if __name__ == '__main__':
    args = parser.parse_args()
    Env( args )
