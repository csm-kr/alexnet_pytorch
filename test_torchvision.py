import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from thop import profile

import os
import torch

# for test
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from test import test_and_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/cvmlserver7/Sungmin/data/imagenet')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='alexnet')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    opts = parser.parse_args()
    print(opts)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
         ])

    test_set = torchvision.datasets.ImageNet(root="/home/cvmlserver7/Sungmin/data/imagenet", transform=transform, split='val')
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=8)

    model = torchvision.models.alexnet(pretrained=True).to(opts.rank)

    # for
    batch_size_ = 1
    input = torch.randn([batch_size_, 3, 224, 224]).to(opts.rank)
    macs, params = profile(model, inputs=(input,))
    print("#params", sum([x.numel() for x in model.parameters()]))
    print("flops :", macs / batch_size_)
    print("params : ", params)

    vis = None
    criterion = CrossEntropyLoss().to(opts.rank)
    test_and_evaluate(epoch=opts.epoch,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      criterion=criterion,
                      opts=opts,
                      is_load=False)
