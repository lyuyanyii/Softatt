import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init

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

class Cls( nn.Module ):
    def __init__(self):
        super().__init__()
        
        self.conv1 = conv( 1, 32 )
        self.conv2 = conv( 32, 32 )
        self.conv3 = conv( 32, 64, stride = 2 )
        self.conv4 = conv( 64, 64 )
        self.conv5 = conv( 64, 128, stride = 2 )
        self.conv6 = nn.Conv2d( 128, 128, 3, padding = 1 )
        self.fc1 = nn.Linear( 128, 10 )

    def forward( self, x ):
        x0 = self.conv2( self.conv1( x  ) )
        x1 = self.conv4( self.conv3( x0 ) )
        x2 = self.conv6( self.conv5( x1 ) )

        f = x2.mean(3).mean(2)
        pred0 = self.fc1( f )

        return x0, x1, x2, pred0

class Reg( nn.Module ):
    def __init__(self):
        super().__init__()
        self.conv7 = conv( 128, 128 )
        self.conv8 = tconv( 128, 64 )
        self.conv9 = conv( 64 + 64, 64 )
        self.conv10 = tconv( 64, 32 )
        self.conv11 = conv( 32 + 32, 32 )
        self.conv12 = nn.Conv2d( 32, 1, 3, padding = 1 )
    def forward( self, x0, x1, x2 ):
        y0 = self.conv8( self.conv7( x2 ) )
        y0 = torch.cat([y0, x1], 1)
        y1 = self.conv10( self.conv9( y0 ) )
        y1 = torch.cat([y1, x0], 1)
        mask = nn.Sigmoid() (self.conv12( self.conv11( y1 ) ))

        return mask

class Net( nn.Module ):
    def __init__( self ):
        super(Net, self).__init__()
        
        self.cls = Cls()
        self.reg = Reg()

        for module in self.children():
            module.apply( weight_init )

    def forward( self, x, stage = 1 ):
        x0, x1, x2, pred0 = self.cls( x )

        if stage == 0:
            return pred0, pred0, None

        mask = self.reg( x0, x1, x2 )

        x = x * mask.expand( x.size() )

        x0, x1, x2, pred1 = self.cls( x )

        return pred0, pred1, mask

