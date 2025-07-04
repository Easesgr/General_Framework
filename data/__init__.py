from data import dataloder
from torch.utils.data import  DataLoader
import logging
import os
logger = logging.getLogger(__name__)

def _get_val_dataloaders(args):
    # 加载测试集
    val_set = dataloder.TestDataSet(os.path.join(args['data']['val_data_root'], 'input'),
                                           os.path.join(args['data']['val_data_root'], 'target'))
    loader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return loader_val

def get_dataloader(args):
    train_data_roots = args.data.train_data_root.split(',')
    train_input_roots = [os.path.join(root, 'input') for root in train_data_roots]
    train_target_roots = [os.path.join(root, 'target') for root in train_data_roots]

    train_set = dataloder.TrainDataSet(train_input_roots, train_target_roots, args.data.patch_size)

    val_set = dataloder.TestDataSet(
        os.path.join(args.data.val_data_root, 'input'),
        os.path.join(args.data.val_data_root, 'target')
    )

    # 这里不再使用 DistributedSampler，保持 shuffle=True
    loader_train = DataLoader(
        train_set, batch_size=args.data.batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )

    loader_test = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    # 可加入验证集

    return loader_train, loader_test

