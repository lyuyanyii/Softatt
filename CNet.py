import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init
from torchvision import models
import torch.utils.model_zoo as model_zoo

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

def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1 ):
    return nn.Sequential(
        nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )


class Reg( nn.Module ):
    def __init__( self ):
        super().__init__()
        pre_chls = [64, 64, 128, 256, 512]
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
        x = torch.cat([x4, x3], 1)
        x = self.conv4( x )
        x2 = self.conv5( x2 )
        x = torch.cat([x, x2], 1)
        x = self.conv6( x )
        x1 = self.conv7( x1 )
        x = torch.cat([x, x1], 1)
        x = self.conv8( x )
        x = torch.cat([x, x0], 1)
        x = self.conv9( x )
        x = self.conv10( x )
        x = nn.Sigmoid()(x)
        return x

class Net( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.cls = Cls()
        self.reg = Reg()

    def forward( self, x, stage = 1, binary=False, single=True ):
        x0, x1, x2, x3, x4, pred0 = self.cls(x)

        if stage == 0:
            return pred0, pred0, None

        mask = self.reg( x0, x1, x2, x3, x4)

        x = x * mask.expand( x.size() )

        x0, x1, x2, x3, x4, pred1 = self.cls(x)

        return pred0, pred1, mask
