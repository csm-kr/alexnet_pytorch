import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class PCANoisePIL(object):
    def __init__(self,
                 alphastd=0.1,
                 eigval=np.array([1.148, 4.794, 55.46]),
                 eigvec=np.array([[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203],])
                 ):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.set_alpha()

    def __call__(self, img):

        # 1. pil to numpy
        img_np = np.array(img)                                   # [H, W, C]
        offset = np.dot(self.eigvec * self.alpha, self.eigval)
        img_np = img_np + offset
        img_np = np.maximum(np.minimum(img_np, 255.0), 0.0)
        ret = Image.fromarray(np.uint8(img_np))
        return ret

    def set_alpha(self, ):
        # change per each epoch
        self.alpha = np.random.normal(0, self.alphastd, size=(3,))


class ImageNetDataset(torchvision.datasets.ImageNet):
    def __init__(self, root, split, transform, visualization=False):
        super().__init__(root, split, transform)
        self.transform = transform
        self.visualization = visualization

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # add visualization to DatasetFolder
        if self.visualization:

            # 1. device
            device = sample.device
            sample = sample.permute(1, 2, 0)

            # 2. un-normalization
            mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
            sample = sample * std + mean

            # 3. clamp
            vis_sample = sample.clamp(0, 1)

            # 4. show
            plt.figure('input')
            plt.imshow(vis_sample)
            plt.show()

        return sample, target


if __name__ == '__main__':

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

    data_root = 'D:\data\ILSVRC_classification'
    test_set = ImageNetDataset(root=data_root,
                               transform=transform_test,
                               split='val',
                               visualization=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=2,
                             shuffle=False,
                             num_workers=0)

    for (images, labels) in test_loader:
        print(images.size())
        print(labels)
