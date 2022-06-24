import torch
import visdom
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

# for dataset
from pcanoice import PCANoisePIL
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import ImageNetDataset

import os

# for model
from alexnet import AlexNet
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import StepLR

import time

# training
from train import train_one_epoch
# test
from test import test_and_evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='/home/cvmlserver7/Sungmin/data/imagenet')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='alexnet')

    # usage : --gpu_ids 0, 1, 2, 3
    return parser


def main_worker(rank, opts):

    # 2. init dist
    local_gpu_id = init_for_distributed(rank, opts)

    # # 3. visdom
    vis = visdom.Visdom(port=opts.port)

    # 4. data set
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        PCANoisePIL(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = ImageNetDataset(root=opts.root,
                                transform=transform_train,
                                split='train',
                                visualization=False)

    test_set = ImageNetDataset(root=opts.root,
                               transform=transform_test,
                               split='val',
                               visualization=False)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=int(opts.batch_size / opts.world_size),
                              shuffle=False,
                              num_workers=int(opts.num_workers / opts.world_size),
                              sampler=train_sampler,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=int(opts.batch_size / opts.world_size),
                             shuffle=False,
                             num_workers=int(opts.num_workers / opts.world_size),
                             sampler=test_sampler,
                             pin_memory=True)

    # 5. model
    model = AlexNet()
    model = model.cuda(local_gpu_id)
    model = DDP(module=model,
                device_ids=[local_gpu_id])

    # 6. criterion
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                weight_decay=opts.weight_decay,
                                momentum=opts.momentum)

    # 8. scheduler
    scheduler = StepLR(optimizer=optimizer,
                       step_size=30,
                       gamma=0.1)

    if opts.start_epoch != 0:
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1),
                                map_location=torch.device('cuda:{}'.format(local_gpu_id)))
        # 하나 적은걸 가져와서 train
        model.load_state_dict(checkpoint['model_state_dict'])          # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load sched state dict

        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    for epoch in range(opts.start_epoch, opts.epoch):

        # 9. train
        train_sampler.set_epoch(epoch)
        train_one_epoch(epoch, vis, train_loader, model, optimizer, criterion, scheduler, opts)

        # 10. test
        test_and_evaluate(epoch, vis, test_loader, model, criterion, opts)
        scheduler.step()

    return 0


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # # 2. init_process_group
    # print(os.environ['MASTER_ADDR'])
    # print(os.environ['MASTER_PORT'])
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Alexnet training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    # main_worker(opts.rank, opts)
    mp.spawn(main_worker,
             args=(opts, ),
             nprocs=opts.world_size,
             join=True)


