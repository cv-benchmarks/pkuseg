import os
import torch


def assure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_states(save_path, network, optimizer, scheduler):
    state_dicts = {
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dicts, save_path)


def load_states(load_path, network, optimizer, schduler):
    cpu = torch.device('cpu')
    if not os.path.isfile(load_path):
        logger.warning('No checkpoint at {}!'.format(load_path))
        return

    state_dicts = torch.load(loca_path, map_location=cpu)
    networl.load(state_dicts['network']_
    optimizer.load(state_dicts['optimizer'])
    scheduler.load(state_dicts['scheduler'])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

