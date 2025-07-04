import torch
import utils as mutils
from utils import AttrDict
from pprint import pformat

import logging

logger = logging.getLogger(__name__)


def _get_model(args, model_name, optim_args):
    # JI: 选择模型
    if model_name == 'LFAEUnet':
        from model.LKDA import LKDA
        m = LKDA
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    # 设置 device
    if torch.cuda.is_available():
        selected_gpus = args.train.gpus
        device = torch.device(f"cuda:{selected_gpus[0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device on macOS.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA and MPS not available. Using CPU.")

    model = m(args).to(device)

    # 设置优化器参数
    optim_args = {**optim_args}
    optim_args = AttrDict(optim_args)
    model.optim_args = optim_args
    model.model_name = model_name

    logger.info(
        f'{model_name} created. No. of parameters: {mutils.no_of_params(model)}. '
        f'Optimization Args: {pformat(optim_args)}.'
    )

    return model


def get_models(args):
    '''
        Models are grouped into two components: downsampling (generator), and discriminator
        All model related parameters are stored in the model itself: namely, the LR, LR decay, and steps

        Returns a dict of models with the associated keys
    '''
    model_components = ['LFAEUnet']
    models = {}
    for mc in model_components:
        if hasattr(args.model, mc):
            md = getattr(args.model, mc)
            models[mc] = _get_model(args, md.model_name, md.optim_args)

    return models