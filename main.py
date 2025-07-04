import torch
import sys
from log import Checkpoint
from utils import AttrDict,get_args
from data import get_dataloader
from model import get_models
from pprint import pformat

import logging
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
from trainer.train import Trainer

if __name__ == '__main__':
    # 获取参数列表
    dargs = get_args()

    # 转换参数格式
    args = AttrDict(dargs)
    # 记录参数日志
    logger.info(f'Command ran: {" ".join(sys.argv)}')
    logger.info(pformat(args))
    # 初始化日志
    if dargs['sys']['log']:
        ckp = Checkpoint(dargs)
    else:
        ckp = None
    if args.run:
        # 随机种子
        torch.manual_seed(args.sys.seed)
        # 初始化模型
        models = get_models(args)
        # 初始化数据集器
        dataloaders = get_dataloader(args)
        # 初始化训练器
        trainer = Trainer(args, models, dataloaders, ckp)

        if not args.train.test_only:
            # 训练
            trainer.train()
        else:
            # 仅测试
            trainer.testall()
            pass

