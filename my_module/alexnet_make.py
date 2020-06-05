import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url


num_classes = 50


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=num_classes):
        size_check = torch.FloatTensor(1, 3, 585, 414)
        features = nn.Sequential(
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

        # バッチサイズ10, 6×6のフィルターが256枚
        # 10バッチは残して、6×6×256を１次元に落とす=>6×6×256=9216
        print("features(size_check).size()", features(size_check).size())
        features_2 = features(size_check).size()[2]
        print('features_2', features_2)
        features_3 = features(size_check).size()[3]
        # バッチ１０の値を軸にして残りの次元を１次元へ落とした場合のTensorの形状をチェックすると9216。
        print("features(size_check).view(size_check.size(0), -1).size()",
              features(size_check).view(size_check.size(0), -1).size())
        # fc_sizeを全結合の形状として保持しておく
        fc_size = features(size_check).view(size_check.size(0), -1).size()[1]
        print('fc_size', fc_size)

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
        print("features(size_check).size()", features(size_check).size())
        features_2 = features(size_check).size()[2]
        print('features_2', features_2)
        features_3 = features(size_check).size()[3]
        # バッチ１０の値を軸にして残りの次元を１次元へ落とした場合のTensorの形状をチェックすると9216。
        print("features(size_check).view(size_check.size(0), -1).size()",
              features(size_check).view(size_check.size(0), -1).size())
        # fc_sizeを全結合の形状として保持しておく
        fc_size = features(size_check).view(size_check.size(0), -1).size()[1]
        print('fc_size', fc_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((17, 11))

        # self.avgpool = nn.AvgPool2d(kernel_size=(
        #     features_2, features_3), padding=0)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=(
            features_2, features_3), padding=0),
            nn.Flatten(),
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.Linear(256 * 8 * 5, 4096),
            # nn.Linear(256 * 17 * 11, 4096),
            nn.Linear(256, 4096),

            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['alexnet'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model
