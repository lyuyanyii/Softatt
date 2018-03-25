import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

def cls_zero_grad( m ):
    if hasattr(m, 'cls'):
        m.zero_grad()

def weight_init( m ):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal( m.weight )
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, save_folder, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    torch.save(state, save_folder + '/' + filename)
    if is_best:
        shutil.copyfile(save_folder + '/' + filename,
                        save_folder + '/' + 'model_best.pth.tar')


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger

class WeightsCheck():
    def __init__(self, model):
        self.params_mean = []
        dtype = torch.FloatTensor
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                self.params_mean.append(float(param.mean().type(dtype)))

    def check(self, model):
        dtype = torch.FloatTensor
        cnt = 0
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                if param.grad is None:
                    print("Warning: param with shape {} has no grad".format(param.size()))
                mean = float(param.mean().type(dtype))
                if mean == self.params_mean[cnt]:
                    print("Warning: param with shape {} has not been updated".format(param.size()))
                self.params_mean[cnt] = mean
                cnt += 1


