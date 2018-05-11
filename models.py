import ANet
import BNet
import CNet
import CDNet121
import PRNet18
import SegNet
import Res18CAM
import VGG19
import DenseNet121
import VGG_CIFAR

def A( **kwargs ):
    return ANet.Net()

def B( **kwargs ):
    return BNet.Net()

def C( large_reg, quarter_reg, new_fc, CAM, new_CAM, new_tconv, **kwargs ):
    return CNet.Net(large_reg=large_reg, quarter_reg=quarter_reg, new_fc=new_fc, CAM=CAM, new_CAM=new_CAM, new_tconv=new_tconv)

def CDNet( **kwargs ):
    return CDNet121.Net()

def PRNet( **kwarges ):
    return PRNet18.Net()

def SNet( **kwargs ):
    return SegNet.Net()

def R18CAM( **kwargs ):
    return Res18CAM.Net()

def VGG( **kwargs ):
    return VGG19.Net()

def DsNet121( **kwargs ):
    return DenseNet121.Net()

def VGGC( **kwargs ):
    return VGG_CIFAR.Net()
