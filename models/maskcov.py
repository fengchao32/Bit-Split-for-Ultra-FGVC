import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Parameter

from .quan import Quantization
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_quan import resnet18_quan, resnet50_quan, resnet101_quan


__all__ = [
    'MaskCOV',
    'maskcov_resnet18',
    'maskcov_resnet50',
    'maskcov_resnet50_quan',
    'maskcov_resnet101',
    'maskcov_resnet101_quan',
]


def _myphi(x, m):
    x = x * m
    return (
        1 - x ** 2 / math.factorial(2)
        + x ** 4 / math.factorial(4)
        - x ** 6 / math.factorial(6)
        + x ** 8 / math.factorial(8)
        - x ** 9 / math.factorial(9)
    )


class AngleLinear(nn.Module):
    """A-Softmax classifier used by the original MaskCOV code when enabled."""

    def __init__(self, in_features, out_features, m=4, phiflag=True, bias=False):
        super(AngleLinear, self).__init__()
        if bias:
            raise ValueError('AngleLinear in MaskCOV is defined without bias')
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, input):
        x = input
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.detach().acos()
            k = (self.m * theta / math.pi).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = _myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        return cos_theta, phi_theta


def _resnet_feature_extractor(backbone):
    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )


_BACKBONES = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet18_quan': resnet18_quan,
    'resnet50_quan': resnet50_quan,
    'resnet101_quan': resnet101_quan,
}


class MaskCOV(nn.Module):
    """MaskCOV classification head on top of a ResNet feature extractor.

    The module intentionally keeps the original attribute names:
    ``model``, ``avgpool``, ``classifier``, ``classifier_swap`` and
    ``classifier_cova``. That makes checkpoints produced by PR_MaskCOV load
    cleanly after stripping an optional ``module.`` prefix.
    """

    def __init__(
        self,
        num_classes=1110,
        backbone='resnet50',
        use_cdrm=True,
        cls_2=True,
        cls_2xmul=False,
        swap_num=(2, 2),
        use_Asoftmax=False,
        quantize_features=False,
    ):
        super(MaskCOV, self).__init__()
        if backbone not in _BACKBONES:
            raise ValueError('unsupported MaskCOV backbone: {}'.format(backbone))
        if cls_2 and cls_2xmul:
            raise ValueError('cls_2 and cls_2xmul are mutually exclusive')
        if use_cdrm and not (cls_2 or cls_2xmul):
            cls_2 = True

        self.use_cdrm = use_cdrm
        self.num_classes = num_classes
        self.backbone_arch = backbone
        self.use_Asoftmax = use_Asoftmax
        self.cls_2 = cls_2
        self.cls_2xmul = cls_2xmul
        self.swap_num = tuple(swap_num)

        backbone_model = _BACKBONES[backbone]()
        self.model = _resnet_feature_extractor(backbone_model)
        feature_dim = backbone_model.fc.in_features

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.feature_quant = Quantization() if quantize_features else nn.Identity()
        self.classifier = nn.Linear(feature_dim, self.num_classes, bias=False)

        if self.use_cdrm:
            if self.cls_2:
                self.classifier_swap = nn.Linear(feature_dim, 2, bias=False)
            elif self.cls_2xmul:
                self.classifier_swap = nn.Linear(feature_dim, 2 * self.num_classes, bias=False)

            self.blockN = self.swap_num[0] * self.swap_num[1]
            self.classifier_cova = nn.Linear(feature_dim, self.blockN * 9, bias=False)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(feature_dim, self.num_classes, bias=False)

    def extract_features(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_quant(x)
        return x

    def forward(self, x, last_cont=None):
        x = self.extract_features(x)

        out = [self.classifier(x)]

        if self.use_cdrm:
            out.append(self.classifier_swap(x))
            out.append(self.classifier_cova(x))

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.extract_features(last_cont)
                out.append(self.Aclassifier(last_x))

        return out


def _clean_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return cleaned


def load_maskcov_state_dict(model, state_dict, strict=False):
    """Load a MaskCOV checkpoint, accepting DataParallel ``module.`` prefixes."""

    cleaned = _clean_state_dict(state_dict)
    model_state = model.state_dict()
    compatible = OrderedDict()
    skipped = []
    for key, value in cleaned.items():
        if hasattr(value, 'shape') and key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if strict and (missing or unexpected or skipped):
        raise RuntimeError(
            'MaskCOV checkpoint mismatch. missing={}, unexpected={}, skipped={}'.format(
                missing, unexpected, skipped
            )
        )
    return missing, unexpected, skipped


def _normalize_factory_kwargs(kwargs):
    if 'numcls' in kwargs and 'num_classes' not in kwargs:
        kwargs['num_classes'] = kwargs.pop('numcls')
    return kwargs


def maskcov_resnet18(pretrained=False, progress=True, **kwargs):
    kwargs.pop('progress', None)
    kwargs = _normalize_factory_kwargs(kwargs)
    if pretrained:
        raise ValueError('pass MaskCOV checkpoints with --pretrained instead')
    return MaskCOV(backbone='resnet18', **kwargs)


def maskcov_resnet50(pretrained=False, progress=True, **kwargs):
    kwargs.pop('progress', None)
    kwargs = _normalize_factory_kwargs(kwargs)
    if pretrained:
        raise ValueError('pass MaskCOV checkpoints with --pretrained instead')
    return MaskCOV(backbone='resnet50', **kwargs)


def maskcov_resnet50_quan(pretrained=False, progress=True, **kwargs):
    kwargs.pop('progress', None)
    kwargs = _normalize_factory_kwargs(kwargs)
    if pretrained:
        raise ValueError('pass MaskCOV checkpoints with --pretrained instead')
    return MaskCOV(backbone='resnet50_quan', quantize_features=True, **kwargs)


def maskcov_resnet101(pretrained=False, progress=True, **kwargs):
    kwargs.pop('progress', None)
    kwargs = _normalize_factory_kwargs(kwargs)
    if pretrained:
        raise ValueError('pass MaskCOV checkpoints with --pretrained instead')
    return MaskCOV(backbone='resnet101', **kwargs)


def maskcov_resnet101_quan(pretrained=False, progress=True, **kwargs):
    kwargs.pop('progress', None)
    kwargs = _normalize_factory_kwargs(kwargs)
    if pretrained:
        raise ValueError('pass MaskCOV checkpoints with --pretrained instead')
    return MaskCOV(backbone='resnet101_quan', quantize_features=True, **kwargs)
