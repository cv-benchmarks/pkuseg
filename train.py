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

from libs.datasets.builder import build_dataset
from libs.networks.builder import ModelBuilder
from libs.utils.trainer import all_reduce, PolyLRScheduler
from libs.utils.trainer import DistributedSampler4Iter
from libs.utils.logger import Logger
from libs.utils.loss import build_criterion


try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


def train(rank, world_size, pth_dir, freq_config, criterion, train_loader,
          model, optimizer, scheduler):
    log_freq = freq_config['log_per_iter']
    tsb_freq = freq_config['tsb_per_iter']
    save_freq = freq_config['save_per_iter']

    for iter, [images, labels, _] in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        preds = model(images)
        loss, losses = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduce_loss = all_reduce_tensor(loss, world_size)
        if local_rank == 0:
            log_iter(losses)
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
    Logger.init(**cfgs['logger'])

    local_rank = cfgs['local_rank']
    world_size = int(os.environ['WORLD_SIZE'])
    Log.info('rank: {}, world_size: {}'.format(local_rank, world_size))

    log_dir = cfgs['log_dir']
    pth_dir = cfgs['pth_dir']
    if local_rank == 0:
        assure_dir(log_dir)
        assure_dir(pth_dir)

    aux_config = cfgs.get('auxiliary', None)
    network = ModuleBuilder(cfgs['network'], aux_config).cuda()
    criterion = build_criterion(cfgs['criterion'], aux_config).cuda()
    optimizer = optim.SGD(network.parameters(), **cfgs['optimizer'])
    scheduler = PolyLRScheduler(optimizer, **cfgs['scheduler'])

    dataset = build_dataset(**cfgs['dataset'], **cfgs['transforms'])
    sampler = DistributedSampler4Iter(dataset, world_size=world_size, 
                                      rank=local_rank, **cfgs['sampler'])
    train_loader = DataLoader(dataset, sampler=sampler, **cfgs['loader'])

    cudnn.benchmark = True
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    model = DistributedDataParallel(network)
    model = apex.parallel.convert_syncbn_model(model)

    torch.cuda.empty_cache()
    train(local_rank, world_size, pth_dir, cfgs['frequency'], criterion, 
          train_loader, model, optimizer, scheduler)


if __name__ == '__main__':
    parser = argparser.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.confg, Loader=yaml.Loader) as fp:
        cfgs = yaml.load(fp)
    cfgs['local_rank'] = args.local_rank

    main(cfgs)
