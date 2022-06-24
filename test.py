import os
import torch

# for test
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from alexnet import AlexNet


def test_and_evaluate(epoch, vis, test_loader, model, criterion, opts):
    """
    evaluate imagenet test data
    :param epoch: epoch for evaluating test dataset
    :param vis: visdom
    :param data_loader: test loader (torch.utils.DataLoader)
    :param model: model
    :param criterion: loss
    :param opts: options from config
    :return: avg_loss and accuracy

    function flow
    1. load .pth file
    2. forward the whole test dataset
    3. calculate loss and accuracy
    """
    # 1. load pth.tar
    if isinstance(model, (torch.nn.parallel.distributed.DistributedDataParallel, torch.nn.DataParallel)):
        checkpoint = torch.load(f=os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                                map_location=torch.device('cuda:{}'.format(opts.rank)))
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

    else:
        checkpoint = torch.load(f=os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                                map_location=torch.device('cuda:{}'.format(opts.rank)))
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('module.', ''): v for (k, v) in state_dict.items()}
        model.load_state_dict(state_dict)

    # 2. forward the whole test dataset & calculate performance
    model.eval()

    val_avg_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image = data[0].to(int(opts.gpu_ids[opts.rank]))
            label = data[1].to(int(opts.gpu_ids[opts.rank]))

            output = model(image)
            loss = criterion(output, label)
            val_avg_loss += loss.item()

            # rank 1
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct_top1 += (pred == label).sum().item()

            # ------------------------------------------------------------------------------
            # rank 5
            _, rank5 = output.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))

            # ------------------------------------------------------------------------------
            for k in range(5):
                correct_k = correct5[:k+1].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()

            print("step : {} / {}".format(idx + 1, len(test_loader.dataset) / int(label.size(0))))
            print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
            print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))
            print("--------------------------------------------------------------")
            print("top-1 error    :  {0:0.3f}%".format((1 - correct_top1 / total) * 100))
            print("top-5 error    :  {0:0.3f}%".format((1 - correct_top5 / total) * 100))
            print("")

        accuracy_top1 = correct_top1 / total
        accuracy_top5 = correct_top5 / total
        val_avg_loss = val_avg_loss / len(test_loader)  # make mean loss

        if vis is not None:
            vis.line(X=torch.ones((1, 3)) * epoch,
                     Y=torch.Tensor([accuracy_top1, accuracy_top5, val_avg_loss]).unsqueeze(0),
                     update='append',
                     win='test_loss_acc',
                     opts=dict(x_label='epoch',
                               y_label='test_loss and acc',
                               title='test_loss and accuracy',
                               legend=['accuracy_top1', 'accuracy_top5', 'avg_loss']))
        print("")
        print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
        print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/cvmlserver7/Sungmin/data/imagenet')
    parser.add_argument('--epoch', type=int, default=72)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='alexnet')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_ids', nargs="+", default=[0])
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

    test_set = ImageNet(root=opts.root, transform=transform, split='val')
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)

    vis = None
    criterion = CrossEntropyLoss().to(opts.rank)
    model = AlexNet().to(opts.rank)

    test_and_evaluate(epoch=opts.epoch,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      criterion=criterion,
                      opts=opts)









