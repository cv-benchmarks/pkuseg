"""
    Distribute Training Code
"""

import argparse
import os
import os.path as osp

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils import data

from libs.datasets.cityscapes import Cityscapes
from libs.utils.logger import Logger
from libs.utils.loss import CriterionOhemDSN, CriterionDSN
from libs.networks.builder import ModelBuilder


try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


def train(rank, world_size, pth_dir, save_per_iter, criterion, train_loader,
          model, optimizer, scheduler):
    for iter, [images, labels, _] in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduce_loss = all_reduce_tensor(loss, world_size)
        if local_rank == 0:
            log_iter()
            if iter % save_per_iter == save_per_iter - 1:
                Log.info('Save checkpoint at step {}'.format(iter))
                latest_path = osp.join(pth_dir, 'latest.pth')
                save_states(latest_path, model.module, optimizer, scheduler)
                iter_path = osp.join(pth_dir, 'step_{}.pth'.format(iter+1))
                shutil.copy(latest_path, iter_path)

    if local_rank == 0:
        Log.info('Save checkpoint at step {}'.format(iter))
        latest_path = osp.join(pth_dir, 'latest.pth')
        save_checkpoint(model_path, model.module, optimizer, scheduler)
        final_path = osp.join(pth_dir, 'final.pth')
        shutil.copy(latest_path, final_path)


def main(cfgs):
    local_rank = cfgs['local_rank']
    world_size = int(os.environ['WORLD_SIZE'])
    Log.info('rank: {}, world_size: {}'.format(local_rank, world_size))

    log_dir = cfgs['log_dir']
    pth_dir = cfgs['pth_dir']
    if local_rank == 0:
        assure_dir(log_dir)
        assure_dir(pth_dir)

    Logger.init(**cfgs['logger'])
    network = ModuleBuilder(cfgs['network'])
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, network.parameters()),
        **cfgs['optimizer'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    model = DistributedDataParallel(network).cuda()
    model = apex.parallel.convert_syncbn_model(model)

    # set loss function
    use_ce_weight = cfgs.get('use_ce_weight', False)
    ohem_config = cfgs.get('ohem', dict())
    ohem_config['aux_weight'] = cfgs.get('aux_loss', dict()).get('aux_weight', 0.)
    criterion = CriterionOhemDSN(use_weight=use_ce_weight, **ohem_config)
    criterion.cuda()

    cudnn.benchmark = True

    if args.world_size == 1:
        print(model)

    # this is a little different from mul-gpu traning setting in distributed training
    # because each train_loader is a process that sample from the dataset class.
    batch_size = args.gpu_num * args.batch_size_per_gpu
    max_iters = args.num_steps * batch_size / args.gpu_num
    # set data loader
    data_set = Cityscapes(args.data_dir, args.data_list, max_iters=max_iters, crop_size=input_size,
                  scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,vars=IMG_VARS, RGB= args.rgb)

    train_loader = data.DataLoader(
        data_set,
        batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print("train loader", len(train_loader))
    # empty cuda cache
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparser.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.confg, Loader=yaml.Loader) as fp:
        cfgs = yaml.load(fp)
    cfgs['local_rank'] = args.local_rank
    main(cfgs)
