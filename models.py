import ANet
import BNet
import CNet

def A( **kwargs ):
    return ANet.Net()

def B( **kwargs ):
    return BNet.Net()

def C( large_reg, **kwargs ):
    return CNet.Net(large_reg=large_reg)
