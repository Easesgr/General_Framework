import re
import ast
import yaml
import torch.optim as optim
import cv2
import logging
import os,sys
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR
import time
from scipy.signal import convolve2d
import math

import numpy as np

try:
    import accimage
except ImportError:
    accimage = None



class PrettySafeLoader(yaml.SafeLoader):
    '''
    Allows yaml to load tuples. Credits to Matt Anderson. See:
    https://stackoverflow.com/questions/9169025/how-can-i-add-a-python-tuple-to-a-yaml-file-using-pyyaml
    '''

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

# 计时器
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class AttrDict(dict):
    def __init__(self, d={}):
        super(AttrDict, self).__init__()
        for k, v in d.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = AttrDict(v)
        super(AttrDict, self).__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return self.__getitem__(k)
        except KeyError:
            raise AttributeError(k)

    __setattr__ = __setitem__


def attr_to_dict(attr):
    '''
        Transforms attr string to nested dict
    '''
    nested_k, v = attr.split('=')
    ks = nested_k.split('.')
    d = {}
    ref = d
    while len(ks) > 1:
        k = ks.pop(0)
        ref[k] = {}
        ref = ref[k]

    ref[ks.pop()] = assign_numeric_type(v)

    return d


def assign_numeric_type(v):
    if re.match(r'^-?\d+(?:\.\d+)$', v) is not None:
        return float(v)
    elif re.match(r'^-?\d+$', v) is not None:
        return int(v)
    elif re.match(r'^range\(-?\d+,-?\d+,-?\d+\)$', v) is not None:
        r_nos = v.split('range(')[-1][:-1].split(',')
        return list(range(int(r_nos[0]), int(r_nos[1]), int(r_nos[2])))
    elif v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    elif v.lower() == 'null':
        return None
    else:
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError) as e:
            return v

def merge_dict(a, b):
    '''
        merge dictionary b into dictionary a
    '''
    assert isinstance(a, dict) and isinstance(b, dict)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def read_yaml(filepath):
    with open(filepath, 'r',encoding='utf-8') as stream:
        try:
            return yaml.load(stream, Loader=PrettySafeLoader)
        except yaml.YAMLError as exc:
            logger.error(exc)
            return {}

def get_args():
    args = {}
    basepath = os.path.dirname(__file__)
    args = merge_dict(args, read_yaml(os.path.join(basepath, 'configs', 'default.yaml')))
    if len(sys.argv) > 1: #
        for arg in sys.argv[1:]:
            if arg.endswith('.yaml'):
                args = merge_dict(args, read_yaml(arg))
            elif len(arg.split('=')) == 2:
                args = merge_dict(args, attr_to_dict(arg))
            else:
                logger.warning(f'unrecognizable argument: {arg}')
    return args


def initialize_optimizer(args,model,model_name):

    optim_g_params = args.model[model_name].optim_args

    # 根据优化器类型选择不同的优化器
    if optim_g_params['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            betas=tuple(optim_g_params['betas'])
        )
    elif optim_g_params['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            momentum=optim_g_params['momentum']
        )
    elif optim_g_params['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            betas=tuple(optim_g_params['betas'])
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_g_params['type']}")

    return optimizer



def initialize_scheduler(args,optimizer,model_name):

    scheduler_params = args.model[model_name].scheduler

    # 根据调度器类型选择不同的调度器
    if scheduler_params['type'] == 'MultiStepLR':
        scheduler = MultiStepLR(
            optimizer,
            milestones=scheduler_params['milestones'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_params['type'] == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_params['type'] == 'ExponentialLR':
        scheduler = ExponentialLR(
            optimizer,
            gamma=scheduler_params['gamma']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_params['type']}")

    return scheduler




def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 计算总参数
def no_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 转换为Y通道
def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    :parame img: uint8 or float ndarray
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                      [128.553, -74.203, -93.786],
                                      [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_im_type)

def tensor2img(tensor):
    """
    将一个 PyTorch tensor 转换为浮点 NumPy 图像数组，像素值范围 [0,255]，不截断溢出值。

    输入:
        tensor: Tensor，形状为 [B, C, H, W]，数值范围通常为 [0,1]
    返回:
        arr: NumPy 数组，形状为 [H, W, C]，类型为 float64，未经截断，直接乘以255
    """
    if tensor.dim() != 4:
        raise ValueError("输入 tensor 必须是四维的：batch_size, channels, height, width")

    # 取第一个样本，变换为 H×W×C，并转换为 numpy
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # 不使用 clip，直接乘255并转 float64（允许超过[0,255]的值）
    arr = (arr * 255.0).astype(np.float64)

    # 转换为 YCbCr 色彩空间
    res = rgb2ycbcr(arr)
    return res


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


