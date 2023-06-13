import torch
from torch import nn

from .darknet import Darknet, load_darknet_from_url, DarknetURL


class NavNet(nn.Module):
    def __init__(self, pretrained: nn.Module, backbone_out_ch: int, n_outputs: int):
        super(NavNet, self).__init__()
        self.features = pretrained
        self.extra_features = nn.Sequential(
            nn.Conv2d(backbone_out_ch, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, n_outputs)
        )
        self._init(self.extra_features, self.regressor)

    @staticmethod
    def _init(*modules):
        for m in modules:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode="fan_out")
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
                elif isinstance(mm, nn.Linear):
                    nn.init.normal_(mm.weight, 0, 0.01)
                    nn.init.zeros_(mm.bias)

    def forward(self, inputs):
        inputs = self.features(inputs)
        inputs = self.extra_features(inputs)
        inputs = torch.flatten(inputs, 1)
        inputs = self.regressor(inputs)
        return inputs


def get_net_by_name(net_name: str, n_outputs: int):
    import torchvision.models as models

    def process_resnet(net_: models.ResNet):
        backbone = nn.Sequential(
            net_.conv1,
            net_.bn1,
            net_.relu,
            net_.maxpool,
            net_.layer1,
            net_.layer2,
            net_.layer3,
            net_.layer4,
        )
        return backbone, 512

    def process_squeezenet(net_: models.SqueezeNet):
        fire: models.squeezenet.Fire = net_.features[-1]
        nout = fire.expand1x1.out_channels + fire.expand3x3.out_channels
        return net_.features, nout

    def process_darknet(net_: Darknet, pretrained: bool):
        if pretrained:
            load_darknet_from_url(net_, DarknetURL)
        return net_.layers[0], 1024

    networks = {
        "resnet18": (models.resnet18, process_resnet),
        "resnet34": (models.resnet34, process_resnet),
        "resnet50": (models.resnet50, process_resnet),
        "resnext50": (models.resnext50_32x4d, process_resnet),
        "squeezenet1_1": (models.squeezenet1_1, process_squeezenet),
        "darknet": (Darknet, process_darknet),
    }
    getter, processor = networks[net_name]
    try:
        full_network = getter(pretrained=True)
        backbone, backbone_out_ch = processor(full_network)
        our_network = NavNet(backbone, backbone_out_ch, n_outputs)
    except TypeError:  # if getter doesn't support pretrained
        full_network = getter()
        backbone, backbone_out_ch = processor(full_network, pretrained=True)
        our_network = NavNet(backbone, backbone_out_ch, n_outputs)
    return our_network
