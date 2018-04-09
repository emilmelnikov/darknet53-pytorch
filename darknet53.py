import functools

import torch
from torch import nn
import torch.nn.functional as F


def applyseq(x, *fs):
    """Consecutively apply functions in a list to a value (see the examples).

    >>> applyseq(1)
    1
    >>> applyseq(1, lambda x: x+1)
    2
    >>> applyseq(1, lambda x: x+1, lambda x: bin(x))
    '0b10'
    """
    return functools.reduce(lambda a, f: f(a), fs, x)


def global_avg_pool2d(x):
    """Global average pooling."""
    return F.adaptive_avg_pool2d(x.view(x.size(0), -1), 1)


class SameConvBlock(nn.Module):
    """Size-preserving convolution, followed by BatchNorm and activation."""

    def __init__(self, cin, cout, *, size, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, size, stride=stride, padding=size//2, bias=False)
        self.bn = nn.BatchNorm1d(cout, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


class ConvBlock(SameConvBlock):
    """Ordinary 3x3 convolution block."""
    def __init__(self, cin, cout):
        super().__init__(cin, cout, size=3, stride=1)


class ReduceConvBlock(SameConvBlock):
    """Dimensionality-reducing 1x1 convolution block."""
    def __init__(self, cin, cout):
        super().__init__(cin, cout, size=1, stride=1)


class DownConvBlock(SameConvBlock):
    """Downsampling 3x3 convolution, divides the size by 2."""
    def __init__(self, cin, cout):
        super().__init__(cin, cout, size=3, stride=2)


class ResBlock(nn.Module):
    """Residual block. The residual path has an 1x1 and 3x3 convolution blocks."""

    def __init__(self, cin):
        super().__init__()
        self.convreduce = ReduceConvBlock(cin, cin//2)
        self.conv = ConvBlock(cin//2, cin)

    def forward(self, x):
        return x + self.conv(self.convreduce(x))


class DarkNetBlock(nn.Module):
    """Basic block for the DarkNet: downsampling followed by multiple residual blocks."""

    def __init__(self, cin, n):
        super().__init__()
        self.convdown = DownConvBlock(cin, 2*cin)
        self.reslist = nn.ModuleList([ResBlock(2*cin) for i in range(n)])

    def forward(self, x):
        return applyseq(x, self.convdown, *self.reslist)


class DarkNet(nn.Module):
    """DarkNet network.

    - First, expand channels from `cin` to `c1`.
    - Then, apply a sequence of DarkNetBlock modules; each element in the `rbs` list adds
      a single such block with the number of residual blocks equal to the element's value.
    - Finally, apply adaptive average pooling to the `nf` features.
    """

    def __init__(self, cin, c1, rbs, nf):
        super().__init__()
        self.conv1 = ConvBlock(cin, c1)
        self.dns = nn.ModuleList([DarkNetBlock(2**i * c1, n) for i, n in enumerate(rbs)])
        self.fc = nn.Linear(2**len(rbs) * c1, nf)

    def forward(self, x):
        return applyseq(x, self.conv1, *self.dns, global_avg_pool2d, self.fc)


def darknet53():
    """Darknet-53 with parameters defined in the paper [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)."""
    return DarkNet(cin=3, c1=32, rbs=(1, 2, 8, 8, 4), nf=1000)
