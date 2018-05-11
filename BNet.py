import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized

class DenseLayer(nn.Module):
    def __init__(self, inp_chl, growth_rate, bn_size = 4):
        super().__init__()
        self.bn1   = nn.BatchNorm2d( inp_chl )
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d( inp_chl, bn_size * growth_rate, 1, stride = 1, bias = False )
        self.bn2   = nn.BatchNorm2d( bn_size * growth_rate )
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d( bn_size * growth_rate, growth_rate, 3, stride = 1, padding = 1, bias = False )

    def forward( self, x ):
        y = x
        y = self.bn1  ( y )
        y = self.relu1( y )
        y = self.conv1( y )
        y = self.bn2  ( y )
        y = self.relu2( y )
        y = self.conv2( y )
        return torch.cat([x, y], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, inp_chl, growth_rate, bn_size = 4):
        super().__init__()
        self.layers = nn.Sequential( *[ DenseLayer( inp_chl + i*growth_rate, growth_rate, bn_size ) for i in range(num_layers) ] )

    def forward( self, x ):
        x = self.layers( x )
        return x

class Transition(nn.Module):
    def __init__(self, inp_chl, out_chl):
        super().__init__()
        self.bn = nn.BatchNorm2d( inp_chl, out_chl )
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d( inp_chl, out_chl, 1, stride = 1, bias = False )
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward( self, x ):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class Cls(nn.Module):
    def __init__( self, growth_rate = 8, layers = [10, 10, 10], bn_size = 4 ):
        super().__init__()

        self.conv1 = nn.Conv2d( 3, growth_rate * 2, 3, stride = 1, padding = 1 )
        self.encoder1 = DenseBlock( layers[0], growth_rate * 2, growth_rate, bn_size )
        self.chl1 = growth_rate * (2 + layers[0])
        self.t1 = Transition( self.chl1, self.chl1 )
        self.encoder2 = DenseBlock( layers[1], self.chl1, growth_rate, bn_size )
        self.chl2 = self.chl1 + growth_rate * layers[1]
        self.t2 = Transition( self.chl2, self.chl2 )
        self.encoder3 = DenseBlock( layers[2], self.chl2, growth_rate, bn_size )
        self.chl3 = self.chl2 + growth_rate * layers[2]
        self.fc = nn.Linear( self.chl3, 10 )
        self.apply( weight_init )

    def forward( self, x ):
        x = self.conv1( x )
        x0 = self.encoder1( x )
        x1 = self.encoder2( self.t1( x0 ) )
        x2 = self.encoder3( self.t2( x1 ) )
        f = x2.mean(3).mean(2)
        pred = self.fc( f )
        return x0, x1, x2, pred

def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.BatchNorm2d( inp_chl ),
        nn.ReLU( True ),
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        )

def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1 ):
    return nn.Sequential(
        nn.BatchNorm2d( inp_chl ),
        nn.ReLU( True ),
        nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        )

class Reg(nn.Module):
    def __init__( self, chl0, chl1, chl2 ):
        super().__init__()
        self.conv1 = conv( chl2, 128 )
        self.conv2 = nn.Sequential( *[conv(128, 128) for _ in range(2)] )
        self.conv3 = tconv( 128, 64 )
        self.conv4 = conv( chl1, 64 )
        self.conv5 = nn.Sequential( *[conv(64, 64) for _ in range(2)] )
        self.conv6 = tconv( 64+64, 32 )
        self.conv7 = conv( chl0, 32 )
        self.conv8 = nn.Sequential( *[conv(32, 32) for _ in range(2)] )
        self.conv9 = conv( 32+32, 1 )
        self.apply( weight_init )

    def forward( self, x0, x1, x2 ):
        x2 = self.conv3( self.conv2( self.conv1( x2 ) ) )
        x1 = self.conv5( self.conv4( x1 ) )
        x2 = F.upsample( x2, (x1.size(2), x1.size(3)), mode = 'bilinear' )
        x = self.conv6( torch.cat([x2, x1], 1) )
        x0 = self.conv8( self.conv7( x0 ) )
        x = F.upsample( x, (x0.size(2), x0.size(3)), mode = 'bilinear' )
        x = self.conv9( torch.cat([x0, x], 1) )
        mask = nn.Sigmoid()( x )
        return mask

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.cls = Cls()
        self.reg = Reg( self.cls.chl1, self.cls.chl2, self.cls.chl3 )

    def forward( self, x, stage = 1, binary = False, single = True, **kwargs ):
        x0, x1, x2, pred0 = self.cls(x)
        
        if stage == 0:
            return pred0, pred0, None

        mask = self.reg( x0, x1, x2 )

        if binary:
            b = Binarized()
            mask = b( mask )
        x_pos = x * mask.expand( x.size() )

        x0, x1, x2, pred1 = self.cls(x_pos)
        
        if single:
            return pred0, pred1, mask

        x_neg = x * (1 - mask.expand(x.size()))

        x0, x1, x2, pred2 = self.cls(x_neg)

        return pred0, pred1, pred2, mask
