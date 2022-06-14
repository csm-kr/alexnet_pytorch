import torch.nn as nn
import torch
from thop import profile


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    #     # init
    #     self.init_convs()
    #     self.classifier.apply(self.init_layers)
    #
    # def init_layers(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 1.)
    #
    # def init_convs(self):
    #     conv_cnt = 0
    #     for m in self.features.children():
    #         if isinstance(m, nn.Conv2d):
    #             conv_cnt += 1
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if conv_cnt in [2, 4, 5]:
    #                 nn.init.constant_(m.bias, 1.)
    #             else:
    #                 nn.init.constant_(m.bias, 0.)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    input = torch.randn([batch_size, 3, 224, 224]).to(device)
    model = AlexNet().to(device)
    macs, params = profile(model, inputs=(input,))
    print("#params", sum([x.numel() for x in model.parameters()]))
    print("flops :", macs / batch_size)
    print("params : ", params)
