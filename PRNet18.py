import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized
from torchvision import models
import torch.utils.model_zoo as model_zoo

class Cls( nn.Module ):
    def __init__( self, pretrained = True ):
        super().__init__()
        resnet = models.resnet18()
        if pretrained:
            #resnet.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            model_file = 'pretrained_models/whole_resnet18_places365_python36.pth.tar'
            resnet = torch.load(model_file)
        self.conv1 = nn.Sequential( resnet.conv1, resnet.bn1, resnet.relu )
        self.layer0 = nn.Sequential( resnet.maxpool, resnet.layer1 )
        self.layer1, self.layer2, self.layer3 = resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.pre_chls = [64, 64, 128, 256, 512]

    def forward( self, x ):
        x0 = self.conv1( x )
        x1 = self.layer0( x0 )
        x2 = self.layer1( x1 )
        x3 = self.layer2( x2 )
        x4 = self.layer3( x3 )
        f = self.avgpool( x4 )
        f = f.view( f.size(0), -1 )
        pred = self.fc( f )
        return x0, x1, x2, x3, x4, pred

class Reg( nn.Module ):
    def __init__( self, pre_chls ):
        super().__init__()
        chls = [64//4, 256//4, 512//4, 1024//4, 2048//4]
        self.t0 = conv( pre_chls[0], chls[0] )
        self.t1 = conv( pre_chls[1], chls[1] )
        self.t2 = conv( pre_chls[2], chls[2] )
        self.t3 = conv( pre_chls[3], chls[3] )
        self.t4 = conv( pre_chls[4], chls[4] )
        BasicBlock = conv
        self.conv1 = nn.Sequential( *[ BasicBlock(chls[4], chls[4]) for _ in range(2) ] )
        self.conv2 = tconv( chls[4], chls[3] )
        self.conv3_0 = conv( chls[3] * 2, chls[3] )
        self.conv3 = nn.Sequential( *[ BasicBlock(chls[3], chls[3]) for _ in range(2) ] )
        self.conv4 = tconv( chls[3], chls[2] )
        self.conv5_0 = conv( chls[2] * 2, chls[2] )
        self.conv5 = nn.Sequential( *[ BasicBlock(chls[2], chls[2]) for _ in range(2) ] )
        self.conv6 = tconv( chls[2], chls[1] )
        self.conv7_0 = conv( chls[1] * 2, chls[1] )
        self.conv7 = nn.Sequential( *[ BasicBlock(chls[1], chls[1]) for _ in range(2) ] )
        self.conv8 = tconv( chls[1], chls[0] )
        self.conv9 = tconv( chls[0] * 2, chls[0] )
        self.conv10_0 = conv( chls[0], 10 )
        self.conv10 = nn.Conv2d( 10, 1, 3, padding = 1 )
        self.apply( weight_init )
    def forward( self, x0, x1, x2, x3, x4 ):
        x0 = self.t0( x0 )
        x1 = self.t1( x1 )
        x2 = self.t2( x2 )
        x3 = self.t3( x3 )
        x4 = self.t4( x4 )
        x4 = self.conv2( self.conv1( x4 ) )
        x3 = torch.cat([x4, x3], 1)
        x3 = self.conv3( self.conv3_0(x3) )
        x3 = self.conv4( x3 )
        x2 = torch.cat([x2, x3], 1)
        x2 = self.conv5( self.conv5_0(x2) )
        x2 = self.conv6( x2 )
        x1 = torch.cat([x1, x2], 1)
        x1 = self.conv7( self.conv7_0(x1) )
        x1 = self.conv8( x1 )
        x0 = torch.cat([x0, x1], 1)
        x0 = self.conv9( x0 )
        x = self.conv10( self.conv10_0(x0) )
        x = nn.Sigmoid()(x)
        return x

class Net( nn.Module ):
    def __init__( self, large_reg=False ):
        super().__init__()

        self.cls = Cls()
        if not large_reg:
            self.reg = Reg( pre_chls=self.cls.pre_chls )
        else:
            self.reg = LReg()

    def forward( self, x, stage = 1, binary=False, single=True, noise=False, gauss=False, R=None, UR=None, noise_rate=None ):
        x0, x1, x2, x3, x4, pred0 = self.cls(x)

        if stage == 0:
            return pred0, pred0, None

        mask = self.reg( x0, x1, x2, x3, x4)

        if noise and not gauss:
            y = x
            perm0 = torch.randperm( y.size(0) ).cuda()
            perm2 = torch.randperm( y.size(2) ).cuda()
            perm3 = torch.randperm( y.size(3) ).cuda()
            #y = y[perm0, :, perm2, perm3]
            y = y[perm0, :, :, :]
            y = y[:, :, perm2, :]
            y = y[:, :, :, perm3]
            y = y.view( x.size() )
        if gauss:
            y = R * noise_rate
            y *= torch.abs(x)

        if binary:
            bi = Binarized()
            mask = bi( mask, UR )
            
        if not gauss:
            x = x * mask.expand( x.size() )

        if noise:
            x = x + y * (1 - mask.expand( x.size() ) )

        x0, x1, x2, x3, x4, pred1 = self.cls(x)

        return pred0, pred1, mask
