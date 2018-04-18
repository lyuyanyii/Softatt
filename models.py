import ANet
import BNet
import CNet
import CDNet121
import PRNet18

def A( **kwargs ):
    return ANet.Net()

def B( **kwargs ):
    return BNet.Net()

def C( large_reg, **kwargs ):
    return CNet.Net(large_reg=large_reg)

def CDNet( **kwargs ):
    return CDNet121.Net()

def PRNet( **kwarges ):
    return PRNet18.Net()
