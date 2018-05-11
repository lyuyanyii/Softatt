import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized, ThresholdBinarized, sharp_t
from torchvision import models
import torch.utils.model_zoo as model_zoo
import utils

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
	}

class Cls( nn.Module ):
    def __init__( self, pretrained = True ):
        super().__init__()
        resnet = models.resnet18()
        if pretrained:
            resnet.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
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
        #f = self.avgpool( x4 )
        f = x4.mean(3).mean(2)
        f = f.view( f.size(0), -1 )
        pred = self.fc( f )
        return x0, x1, x2, x3, x4, pred

def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1, new=False ):
    if new:
        return nn.Sequential(
            nn.Upsample( scale_factor=2 ),
            conv( inp_chl, out_chl ),
            )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
            nn.BatchNorm2d( out_chl ),
            nn.ReLU( True ),
            )

class Reg( nn.Module ):
    def __init__( self, pre_chls, new_tconv=False ):
        super().__init__()
        chls = [64//4, 256//4, 512//4, 1024//4, 2048//4]
        self.t0 = conv( pre_chls[0], chls[0] )
        self.t1 = conv( pre_chls[1], chls[1] )
        self.t2 = conv( pre_chls[2], chls[2] )
        self.t3 = conv( pre_chls[3], chls[3] )
        self.t4 = conv( pre_chls[4], chls[4] )
        self.conv1 = nn.Sequential( *[ conv(chls[4], chls[4]) for _ in range(2) ] )
        self.conv2 = tconv( chls[4], chls[3], new=new_tconv )
        self.conv3 = nn.Sequential( *[ conv(chls[3], chls[3]) for _ in range(2) ] )
        self.conv4 = tconv( chls[3] * 2, chls[2], new=new_tconv )
        self.conv5 = nn.Sequential( *[ conv(chls[2], chls[2]) for _ in range(2) ] )
        self.conv6 = tconv( chls[2] * 2, chls[1], new=new_tconv )
        self.conv7 = nn.Sequential( *[ conv(chls[1], chls[1]) for _ in range(2) ] )
        self.conv8 = tconv( chls[1] * 2, chls[0], new=new_tconv )
        self.conv9 = tconv( chls[0] * 2, chls[0], new=new_tconv )
        self.conv10 = nn.Conv2d( chls[0], 1, 3, padding = 1 )
        self.apply( weight_init )

    def forward( self, x0, x1, x2, x3, x4 ):
        x0 = self.t0( x0 )
        x1 = self.t1( x1 )
        x2 = self.t2( x2 )
        x3 = self.t3( x3 )
        x4 = self.t4( x4 )
        x4 = self.conv2( self.conv1( x4 ) )
        x3 = self.conv3( x3 )
        x4 = F.upsample( x4, (x3.size(2), x3.size(3)), mode = 'bilinear' )
        x = torch.cat([x4, x3], 1)
        x = self.conv4( x )
        x2 = self.conv5( x2 )
        x = F.upsample( x, (x2.size(2), x2.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x2], 1)
        x = self.conv6( x )
        x1 = self.conv7( x1 )
        x = F.upsample( x, (x1.size(2), x1.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x1], 1)
        x = self.conv8( x )
        x = F.upsample( x, (x0.size(2), x0.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x0], 1)
        x = self.conv9( x )
        x = self.conv10( x )
        x = nn.Sigmoid()(x)
        return x

class CAM_Reg( nn.Module ):
    def __init__( self, pre_chls ):
        super().__init__()
        chls = [64//4, 256//4, 512//4, 1024//4, 2048//4]
        self.t0 = conv( pre_chls[0], chls[0] )
        self.t1 = conv( pre_chls[1], chls[1] )
        self.t2 = conv( pre_chls[2], chls[2] )
        self.t3 = conv( pre_chls[3], chls[3] )
        self.t4 = conv( pre_chls[4], chls[4] )
        self.conv1 = nn.Sequential( *[ conv(chls[4], chls[4]) for _ in range(2) ] )
        self.conv2 = tconv( chls[4], chls[3] )
        self.conv3 = nn.Sequential( *[ conv(chls[3], chls[3]) for _ in range(2) ] )
        self.conv4 = tconv( chls[3] * 2, chls[2] )
        self.conv5 = nn.Sequential( *[ conv(chls[2], chls[2]) for _ in range(2) ] )
        self.conv6 = tconv( chls[2] * 2, chls[1] )
        self.conv7 = nn.Sequential( *[ conv(chls[1], chls[1]) for _ in range(2) ] )
        self.conv8 = tconv( chls[1] * 2, chls[0] )
        self.conv9 = tconv( chls[0] * 2, chls[0] )
        self.conv9_1 = conv( chls[0] + 1, chls[0] )
        self.conv9_2 = conv( chls[0], chls[0] )
        self.conv10 = nn.Conv2d( chls[0], 1, 3, padding = 1 )
        self.apply( weight_init )

    def forward( self, x0, x1, x2, x3, x4, CAM ):
        x0 = self.t0( x0 )
        x1 = self.t1( x1 )
        x2 = self.t2( x2 )
        x3 = self.t3( x3 )
        x4 = self.t4( x4 )
        x4 = self.conv2( self.conv1( x4 ) )
        x3 = self.conv3( x3 )
        x4 = F.upsample( x4, (x3.size(2), x3.size(3)), mode = 'bilinear' )
        x = torch.cat([x4, x3], 1)
        x = self.conv4( x )
        x2 = self.conv5( x2 )
        x = F.upsample( x, (x2.size(2), x2.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x2], 1)
        x = self.conv6( x )
        x1 = self.conv7( x1 )
        x = F.upsample( x, (x1.size(2), x1.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x1], 1)
        x = self.conv8( x )
        x = F.upsample( x, (x0.size(2), x0.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x0], 1)
        x = self.conv9( x )
        x = torch.cat([x, CAM], 1)
        x = self.conv9_1(x)
        x = self.conv9_2(x)
        x = self.conv10( x )
        x = nn.Sigmoid()(x)
        return x

class new_CAM_Reg( nn.Module ):
    def __init__( self, pre_chls ):
        super().__init__()
        chls = [64//4, 256//4, 512//4, 1024//4, 2048//4]
        self.t0 = conv( pre_chls[0], chls[0] )
        self.t1 = conv( pre_chls[1], chls[1] )
        self.t2 = conv( pre_chls[2], chls[2] )
        self.t3 = conv( pre_chls[3], chls[3] )
        self.t4 = conv( pre_chls[4], chls[4] )
        self.conv1 = nn.Sequential( *[ conv(chls[4], chls[4]) for _ in range(2) ] )
        self.conv2 = tconv( chls[4], chls[3] )
        self.conv3 = nn.Sequential( *[ conv(chls[3], chls[3]) for _ in range(2) ] )
        self.conv4 = tconv( chls[3] * 2, chls[2] )
        self.conv5 = nn.Sequential( *[ conv(chls[2], chls[2]) for _ in range(2) ] )
        self.conv6 = tconv( chls[2] * 2, chls[1] )
        self.conv7 = nn.Sequential( *[ conv(chls[1], chls[1]) for _ in range(2) ] )
        self.conv8 = tconv( chls[1] * 2, chls[0] )
        self.conv9 = tconv( chls[0] * 2, chls[0] )
        self.conv9_1 = conv( 1, chls[0] )
        self.conv10 = nn.Conv2d( chls[0] * 2, 1, 3, padding = 1 )
        self.apply( weight_init )

    def forward( self, x0, x1, x2, x3, x4, CAM ):
        x0 = self.t0( x0 )
        x1 = self.t1( x1 )
        x2 = self.t2( x2 )
        x3 = self.t3( x3 )
        x4 = self.t4( x4 )
        x4 = self.conv2( self.conv1( x4 ) )
        x3 = self.conv3( x3 )
        x4 = F.upsample( x4, (x3.size(2), x3.size(3)), mode = 'bilinear' )
        x = torch.cat([x4, x3], 1)
        x = self.conv4( x )
        x2 = self.conv5( x2 )
        x = F.upsample( x, (x2.size(2), x2.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x2], 1)
        x = self.conv6( x )
        x1 = self.conv7( x1 )
        x = F.upsample( x, (x1.size(2), x1.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x1], 1)
        x = self.conv8( x )
        x = F.upsample( x, (x0.size(2), x0.size(3)), mode = 'bilinear' )
        x = torch.cat([x, x0], 1)
        x = self.conv9( x )
        CAM = self.conv9_1(CAM)
        x = self.conv10( torch.cat([x, CAM], 1) )
        x = nn.Sigmoid()(x)
        return x

class LReg( nn.Module ):
    def __init__( self, pre_chls ):
        super().__init__()
        chls = [64//4, 256//4, 512//4, 1024//4, 2048//4]
        self.t0 = conv( pre_chls[0], chls[0] )
        self.t1 = conv( pre_chls[1], chls[1] )
        self.t2 = conv( pre_chls[2], chls[2] )
        self.t3 = conv( pre_chls[3], chls[3] )
        self.t4 = conv( pre_chls[4], chls[4] )
        BasicBlock = models.resnet.BasicBlock
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

class Quarter_Reg( nn.Module ):
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
        self.conv8 = conv( chls[1], chls[0] )
        self.conv9_0 = conv( chls[0], chls[0], stride = 2 )
        self.conv9 = conv( chls[0] * 2, chls[0] )
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
        x0 = self.conv9_0( x0 )
        x0 = torch.cat([x0, x1], 1)
        x0 = self.conv9( x0 )
        x = self.conv10( self.conv10_0(x0) )
        x = nn.Sigmoid()(x)
        x = F.upsample( x, (x.size(2) * 2, x.size(3) * 2), mode = 'bilinear' )
        return x

class Net( nn.Module ):
    def __init__( self, large_reg=False, quarter_reg=False, new_fc=False, cls=None, CAM=False, new_CAM=False, new_tconv=False ):
        super().__init__()

        if cls is None:
            self.cls = Cls()
        else:
            self.cls = cls
        if CAM or new_CAM:
            if CAM:
                self.reg = CAM_Reg( pre_chls=self.cls.pre_chls )
            elif new_CAM:
                self.reg = new_CAM_Reg( pre_chls=self.cls.pre_chls )
            self.CAM = True
        elif not large_reg:
            self.reg = Reg(pre_chls=self.cls.pre_chls, new_tconv=new_tconv)
        elif quarter_reg:
            self.reg = Quarter_Reg(pre_chls=self.cls.pre_chls)
        else:
            self.reg = LReg(pre_chls=self.cls.pre_chls)
        if new_fc:
            self.cls.fc = nn.Linear( self.cls.pre_chls[-1], 200 )

    def forward( self, x, stage = 1, binary=False, single=True, noise=False, gauss=False, R=None, UR=None, noise_rate=None, hard_threshold=False, sharp=False, quantiled=False, CAM=False, **kwargs ):
        x0, x1, x2, x3, x4, pred0 = self.cls(x)

        if stage == 0:
            return pred0, pred0, None

        if hasattr(self, 'CAM') and self.CAM:
            score0, pred = torch.max( pred0, 1 )
            fc = list(self.cls.fc.parameters())[0]
            CAM_mask = torch.cat( [self._get_mask(x4[i], int(pred[i]), fc, pred0[i]) for i in range(x.size(0))], 0 )
            CAM_mask = F.upsample( CAM_mask, (x.size(2), x.size(3)), mode = 'bilinear' )
            mask = self.reg( x0, x1, x2, x3, x4, CAM_mask )
        else:
            mask = self.reg( x0, x1, x2, x3, x4)
        mask = F.upsample( mask, (x.size(2), x.size(3)), mode = 'bilinear' )

        y = None
        if noise and not gauss:
            y = x
            y = torch.transpose(y, 0, 1)
            y = torch.transpose(y, 1, 2)
            y = y.contiguous()
            y = y.view( x.size(2), x.size(0) * x.size(1), x.size(3) )
            perm0 = torch.randperm( y.size(0) ).cuda()
            perm1 = torch.randperm( y.size(1) ).cuda()
            perm2 = torch.randperm( y.size(2) ).cuda()
            y = y[perm0, :, :]
            y = y[:, perm1, :]
            y = y[:, :, perm2]
            y = y.view( x.size() )
            
            y = torch.transpose(y, 0, 1)
            y = torch.transpose(y, 1, 2)
            y = y.contiguous()
            y = y.view( x.size(2), x.size(0) * x.size(1), x.size(3) )
            perm0 = torch.randperm( y.size(0) ).cuda()
            perm1 = torch.randperm( y.size(1) ).cuda()
            perm2 = torch.randperm( y.size(2) ).cuda()
            y = y[perm0, :, :]
            y = y[:, perm1, :]
            y = y[:, :, perm2]
            y = y.view( x.size() )
        if gauss:
            y = R
            y *= torch.abs(x)
        if noise:
            y *= noise_rate
        if y is None:
            y = 0

        if quantiled:
            q = utils.Quantile()
            mask = q( mask )
        if binary:
            if not hard_threshold:
                bi = Binarized()
                mask = bi( mask, UR )
            else:
                bi = ThresholdBinarized()
                mask = bi( mask )
        
        if sharp:
            #mask_P = (mask > 0.1).type( torch.cuda.FloatTensor )
            #mask = mask * mask_P
            sp = sharp_t()
            mask = sp(mask)
           
        if not gauss:
            x = x * mask.expand( x.size() )

        if noise:
            x = x + y * (1 - mask.expand( x.size() ) )


        if CAM:
            score0, pred = torch.max( pred0, 1 )
            fc = list(self.cls.fc.parameters())[0]
            CAM_mask = torch.cat( [self._get_mask(x4[i], int(pred[i]), fc, pred0[i]) for i in range(x.size(0))], 0 )
            CAM_mask = F.upsample( CAM_mask, (x.size(2), x.size(3)), mode = 'bilinear' )

        x0, x1, x2, x3, x4, pred1 = self.cls(x)

        if CAM:
            return pred0, pred1, mask, CAM_mask
        return pred0, pred1, mask

    def _get_mask( self, x, idx, fc, pred0 ):
        mask = (x[None, :, :, :] * fc[:, :, None, None] * pred0[:, None, None, None]).sum(0).sum(0)
        mask = mask - mask.min()
        mask = mask / mask.max()
        return mask[None, None, :, :]


