import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import math
# from CrossAttention import *

class Conv2dBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dBN, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=True)),
            ('bn', nn.BatchNorm2d(out_channels))
        ]))

class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dBNReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class Conv2dINReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dINReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('in', nn.InstanceNorm2d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class ConvTrans1dBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans1dBN, self).__init__(OrderedDict([
            ('convtrans1d', nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm1d(out_channels))                        
        ]))

class ConvTrans1dBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans1dBNReLu, self).__init__(OrderedDict([
            ('convtrans1d', nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm1d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class ConvTrans2dBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans2dBNReLu, self).__init__(OrderedDict([
            ('convtrans2d', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv1dBNReLU, self).__init__(OrderedDict([
            ('conv1d', nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                groups)),
            ('bn', nn.BatchNorm1d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlock1D, self).__init__()
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.k = kernel_size
        self.stride = stride
        self.padding = int(padding)

        self.conv1 = Conv1dBNReLU(self.in_ch, self.out_ch, self.k, 1, is_padding=self.padding)
        self.conv2 = Conv1dBNReLU(self.out_ch, self.out_ch, self.k, self.stride, is_padding=self.padding)

        self.relu = nn.LeakyReLU(0.3)
        if use_1x1conv:
            self.conv3 = Conv1dBNReLU(self.in_ch, self.out_ch, 1, self.stride, is_padding=self.padding)
        else:
            self.conv3 = None
    def forward(self, x):
        # x1 = self.conv2(F.relu(self.conv1(x)))
        x1 = self.conv2(self.relu(self.conv1(x)))
        if self.conv3:
            x = self.conv3(x)
        out = x+x1
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.k = kernel_size
        self.stride = stride
        self.padding = int(padding)

        self.conv1 = Conv2dBN(self.in_ch, self.out_ch, self.k, 1, is_padding=self.padding)
        self.conv2 = Conv2dBN(self.out_ch, self.out_ch, self.k, self.stride, is_padding=self.padding)

        self.relu = nn.LeakyReLU(0.3)
        if use_1x1conv:
            self.conv3 = Conv2dBN(self.in_ch, self.out_ch, 1, self.stride, is_padding=self.padding)
        else:
            self.conv3 = None
    def forward(self, x):
        # x1 = self.conv2(F.relu(self.conv1(x)))
        x1 = self.conv2(self.relu(self.conv1(x)))
        if self.conv3:
            x = self.conv3(x)
        out = x+x1
        return out

class ResBlockBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlockBNReLu, self).__init__(OrderedDict([
            ('conv2d', ResBlock(in_channels, out_channels, kernel_size, stride,
                                padding=padding, use_1x1conv=use_1x1conv)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))    
        ]))

class ConvAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.query = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.key = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.value = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.lepe = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
    def forward(self, x):
        q = self.query(x)   # [B, c, 1, 1024]
        k = self.key(x)     # # [B, c, 1, 1024]
        attn_matrix = q.transpose(-2, -1) @ k
        attn_matrix = nn.functional.softmax(attn_matrix, dim=-1, dtype=attn_matrix.dtype)
        v = self.value(x)
        pe = self.lepe(x)
        x = (v @ attn_matrix) + pe
        return x
    
class Conv1dAttention(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, is_padding=1):
        super().__init__()
        self.query = Conv1dBNReLU(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, is_padding=is_padding)
        self.key = Conv1dBNReLU(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, is_padding=is_padding)
        self.value = Conv1dBNReLU(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, is_padding=is_padding)
        self.lepe = Conv1dBNReLU(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, is_padding=is_padding)
    def forward(self, x):
        q = self.query(x)   # [B, c, 1, 1024]
        k = self.key(x)     # # [B, c, 1, 1024]
        attn_matrix = q.transpose(-2, -1) @ k
        attn_matrix = nn.functional.softmax(attn_matrix, dim=-1, dtype=attn_matrix.dtype)
        v = self.value(x)
        pe = self.lepe(x)
        x = (v @ attn_matrix) + pe
        return x


class LinearAttention(nn.Module):
    def __init__ (self, in_feats, out_feats):
        self.query = nn.Linear(in_feats, out_feats)
        self.key = nn.Linear(in_feats, out_feats)
        self.value = nn.Linear(in_feats, out_feats)
        self.relu = nn.LeakyReLU(0.3)
    
    def forward(self, x):
        q = self.query(x) # [B, 75, out_feats]
        k = self.key(x) # [B, 75, out_feats]
        attn_matrix = q.transpose(-2, -1) @ k # [B, out_feats, out_feats]
        attn_matrix = nn.functional.softmax(attn_matrix, dim=-1, dtype=attn_matrix.dtype)
        v = self.value(x)
        x = v @ attn_matrix
        return x

class Conv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False))
        ]))


class UNet_Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, F_norm=nn.BatchNorm2d, dropout=0.0, is_padding=True, last_conv=False):
        super().__init__()
        if is_padding and not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        if last_conv == False:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True),

                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                F_norm(C_out),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                # nn.Dropout(0.3),
                nn.LeakyReLU(0.3, inplace=False),

                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),

            )

    def forward(self, x):
        return self.layer(x)

class Down_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, F_norm=nn.BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, kernel_size, stride=stride, padding=padding, bias=False),
            F_norm(C),
            nn.LeakyReLU(0.3, inplace=False)
        )

    def forward(self, x):
        return self.Down(x)


class Up_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, output_padding, F_norm=nn.BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(kernel_size[i] - stride[i] + output_padding[i]
                        ) // 2 for i in range(len(kernel_size))]
        else:
            padding = (kernel_size - stride + output_padding) // 2
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(C, C//2, kernel_size, stride,
                               padding, output_padding, bias=False),
            F_norm(C//2),
            nn.LeakyReLU(0.3, inplace=False)
        )

    def forward(self, x, r):
        return torch.cat((self.Up(x), r), 1)
    

class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_ratio=0):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        return x
    
class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 is_padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__() 
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        out_vanilla = self.conv(x)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        return out_vanilla - self.theta * out_diff
        # [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        # tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0) # .cuda()
        # conv_weight = torch.cat((tensor_zeros, self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], tensor_zeros), 2)
        # conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        # out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        # if math.fabs(self.theta - 0.0) < 1e-8:
        #     return out_normal 
        # else:
        #     #pdb.set_trace()
        #     [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
        #     kernel_diff = self.conv.weight.sum(2).sum(2)
        #     kernel_diff = kernel_diff[:, :, None, None]
        #     out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        #     return out_normal - self.theta * out_diff
        
class ResBlock_CDC(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlock_CDC, self).__init__()
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.k = kernel_size
        self.stride = stride
        self.padding = int(padding)

        self.conv1 = nn.Sequential(Conv2d_Hori_Veri_Cross(self.in_ch, self.out_ch, self.k, 1, is_padding=self.padding),
                                   nn.BatchNorm2d(self.out_ch),
                                   nn.LeakyReLU(0.3))
        self.conv2 = nn.Sequential(Conv2d_Hori_Veri_Cross(self.out_ch, self.out_ch, self.k, self.stride, is_padding=self.padding),
                                   nn.BatchNorm2d(self.out_ch),
                                   nn.LeakyReLU(0.3))

        self.relu = nn.LeakyReLU(0.3)
        if use_1x1conv:
            self.conv3 = nn.Sequential(Conv2d_Hori_Veri_Cross(self.in_ch, self.out_ch, 1, self.stride, is_padding=self.padding),
                                   nn.BatchNorm2d(self.out_ch),
                                   nn.LeakyReLU(0.3))
        else:
            self.conv3 = None
    def forward(self, x):
        # x1 = self.conv2(F.relu(self.conv1(x)))
        x1 = self.conv2(self.relu(self.conv1(x)))
        if self.conv3:
            x = self.conv3(x)
        out = x+x1
        return out
    
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant=1):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant=1):
        return GradReverse.apply(x, constant)
    
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out
    
class Conv2Plus1D(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, mid_planes: int, spatial_stride: int = 1, temporal_stride: int = 1, is_padding: int = 1, 
                 spatial_kernel_size: int = 5, temporal_kernel_size: int = 11) -> None:
        super(Conv2Plus1D, self).__init__()
        if not isinstance(spatial_kernel_size, int):
            spatial_padding = [(i - 1) // 2 for i in spatial_kernel_size]
        else:
            spatial_padding = [(spatial_kernel_size - 1) // 2, (spatial_kernel_size - 1) // 2]    
            spatial_kernel_size = [spatial_kernel_size, spatial_kernel_size]
        
        if isinstance(spatial_stride, int):
            spatial_stride = [spatial_stride, spatial_stride]
        if is_padding:
            # spatial_padding = 2
            temporal_padding = (temporal_kernel_size - 1) // 2
        self.spatial_conv = nn.Conv3d(
                in_channels=in_planes,
                out_channels=mid_planes,
                kernel_size=(1, spatial_kernel_size[0], spatial_kernel_size[1]),
                stride=(1, spatial_stride[0], spatial_stride[1]),
                padding=(0, spatial_padding[0], spatial_padding[1]),
                bias=False,
            )
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.relu = nn.LeakyReLU(0.3, inplace=False)
        self.temporal_conv = nn.Conv3d(
                in_channels=mid_planes, 
                out_channels=out_planes, 
                kernel_size=(temporal_kernel_size, 1, 1), 
                stride=(temporal_stride, 1, 1), 
                padding=(temporal_padding, 0, 0), 
                bias=False
            )
        self.bn2 = nn.BatchNorm3d(out_planes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x
    
class ResBlock_Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size, 
                 temporal_kernel_size, spatial_stride, temporal_stride, 
                 padding, use_1x1conv=True):
        super(ResBlock_Conv2Plus1D, self).__init__()
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)

        self.padding = int(padding)

        self.conv1 = Conv2Plus1D(in_planes=in_channels, 
                                 out_planes=out_channels,
                                 mid_planes=out_channels,
                                 spatial_kernel_size=spatial_kernel_size,
                                 temporal_kernel_size=temporal_kernel_size,
                                 spatial_stride=spatial_stride,
                                 temporal_stride=temporal_stride,
                                 is_padding=self.padding)
                                 
        self.conv2 = Conv2Plus1D(in_planes=out_channels, 
                                 out_planes=out_channels,
                                 mid_planes=out_channels,
                                 spatial_kernel_size=spatial_kernel_size,
                                 temporal_kernel_size=temporal_kernel_size,
                                 spatial_stride=1,
                                 temporal_stride=temporal_stride,
                                 is_padding=self.padding)

        self.relu = nn.LeakyReLU(0.3)
        if use_1x1conv:
            self.conv3 = Conv2Plus1D(in_planes=in_channels, 
                                 out_planes=out_channels,
                                 mid_planes=out_channels,
                                 spatial_kernel_size=1,
                                 temporal_kernel_size=temporal_kernel_size,
                                 spatial_stride=spatial_stride,
                                 temporal_stride=temporal_stride,
                                 is_padding=self.padding)
        else:
            self.conv3 = None
    def forward(self, x):
        # x1 = self.conv2(F.relu(self.conv1(x)))
        x1 = self.conv2(self.relu(self.conv1(x)))
        if self.conv3:
            x = self.conv3(x)
        out = x+x1
        return out


class Transposed_Conv2Plus1D(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, spatial_stride: int = 1, temporal_stride: int = 1, padding: int = 1) -> None:
        super(Transposed_Conv2Plus1D, self).__init__()
        self.spatial_conv = nn.ConvTranspose3d(
                in_channels=in_planes,
                out_channels=midplanes,
                kernel_size=(1, 5, 5),
                stride=(1, spatial_stride, spatial_stride),
                padding=(0, padding, padding),
                bias=False,
                output_padding=(0, 0, 0)
            )
        self.bn1 = nn.BatchNorm3d(midplanes)
        self.relu = nn.LeakyReLU(0.3, inplace=False)
        self.temporal_conv = nn.ConvTranspose3d(
                in_channels=midplanes, 
                out_channels=out_planes, 
                kernel_size=(11, 1, 1), 
                stride=(temporal_stride, 1, 1), 
                padding=(5, 0, 0), 
                bias=False,
                output_padding=(1, 0, 0)
            )
        self.bn2 = nn.BatchNorm3d(out_planes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x    
    


class I3DHead_m(nn.Module):
    """Classification head for I3D. modified to 1D feats

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.avg_pool = None

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, L]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


class VirtualSoftmax(nn.Module):
    def __init__(self, linear_layer):
        super(VirtualSoftmax, self).__init__()
        self.linear = linear_layer

    def forward(self, inputs, labels=None, mode='train'):
        # inputs: [B, feat_dim]
        # labels: [B, 1]
        WX = self.linear(inputs)
        if mode == 'train' and labels is not None:
            W_yi = self.linear.weight[:, labels.long()] # [feat_dim, B, 1]
            W_yi_norm = torch.norm(W_yi, dim=0) # [B, 1]
            X_i_norm = torch.norm(inputs, dim=1) # [B]
            WX_virt = W_yi_norm.squeeze(-1) * X_i_norm
            # WX_virt = torch.clamp(WX_virt, 1e-10, 15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class Domain_Regressor(nn.Module):
    def __init__(self, dim):
        super(Domain_Regressor, self).__init__()
        self.conv = ResBlock(dim, 1, (1, 11), 1, 1)
        self.fc = FFN(150, 64, 32)
    
    def forward(self, x, constant=1):
        input = GradReverse.grad_reverse(x, constant)
        x = self.conv(input).squeeze()
        x = self.fc(x)
        return x

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out
    
def load_pretrained_model_to_custom_config(custom_config):
    # pretrained_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")
    pretrained_model = VideoMAEModel.from_pretrained("huggingface/")
    custom_model = VideoMAEModel(custom_config)
    custom_model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=False)
    return custom_model


def load_pretrained_model_to_existing_model(custom_model, use_option='use_pretrained'):
    print(f'VideoMAE backbone:{use_option}')
    if use_option == 'use_pretrained':
        pretrained_model = VideoMAEModel.from_pretrained("huggingface/")
        custom_model.load_state_dict(pretrained_model.encoder.state_dict(), strict=False)
    elif use_option == 'freeze':
        pretrained_model = VideoMAEModel.from_pretrained("huggingface/")
        custom_model.load_state_dict(pretrained_model.encoder.state_dict(), strict=False)
        for k, v in custom_model.named_parameters():
            v.requires_grad = False
    elif use_option == 'no_use':
        custom_model = custom_model
    return custom_model


class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim, in_dim//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.leakyrelu = nn.LeakyReLU(0.3)
        self.fc3 = nn.Linear(in_dim//2, out_dim)

    def forward(self, X, Y):
        # X, Y:  [B, 150, dim]
        f = torch.cat([X, Y], dim=-1)
        gX = self.sigmoid(self.dropout1(self.fc1(f)))
        gY = self.sigmoid(self.dropout2(self.fc2(f)))
        out = gX * X + gY * Y
        out = F.normalize(self.fc3(out), dim=-1)
        return out
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, use_batch_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None

    def return_hidden_feats(self, x):
        hidden = self.input_fc(x)
        
        if self.use_batch_norm:
            hidden = self.batch_norm(hidden)

        hidden = F.leaky_relu(hidden, 0.3)
        return hidden

    def forward(self, x):    
        hidden = self.input_fc(x)
        
        if self.use_batch_norm:
            hidden = self.batch_norm(hidden)

        hidden = F.leaky_relu(hidden, 0.3)
        
        y_pred = self.output_fc(hidden)
        
        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred

class GraphConvolution(nn.Module):
    """
    Simple GCN layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x



class TaskAttentionAggregator(nn.Module):
    """
    使用 Transformer Encoder Layer 和 [CLS] Token 聚合任务内图片 Embedding 序列。
    处理变长序列输入。
    输入:
        - 一个批次的任务图片 Embeddings (填充后)
        - 对应的 Padding Mask
    输出: 一个聚合后的 Task Embedding
    """
    def __init__(self, embed_dim=256, nhead=8, dim_feedforward=1024, dropout=0.1, max_images=10): # max_images 用于定义位置编码大小
        """
        初始化函数

        Args:
            embed_dim (int): Embedding 维度。
            nhead (int): 多头注意力头数。
            dim_feedforward (int): 前馈网络维度。
            dropout (float): Dropout 概率。
            max_images (int): 数据集中任务包含的最大图片数，用于确定位置编码大小。
        """
        super().__init__()
        self.embed_dim = embed_dim
        # 注意：位置编码需要覆盖最大可能的长度 + 1 ([CLS])
        self.seq_len_with_cls = max_images + 1

        # 1. [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # 2. 位置编码 (定义为最大长度)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len_with_cls, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. (可选) LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, intra_task_embeddings, src_key_padding_mask):
        """
        前向传播函数

        Args:
            image_embeddings (torch.Tensor): 填充后的一个批次的任务图片 Embeddings。
                                            Shape: [batch_size, max_len, embed_dim]
            src_key_padding_mask (torch.Tensor): 指示哪些位置是 Padding 的布尔掩码。
                                                 True 表示该位置是 Padding，False 表示是真实数据。
                                                 Shape: [batch_size, max_len]

        Returns:
            torch.Tensor: 聚合后的 Task Embeddings。
                          Shape: [batch_size, embed_dim]
        """
        if len(intra_task_embeddings.shape) == 3:
            B, S, E = intra_task_embeddings.shape # S 是填充后的长度 max_len
        else: 
            intra_task_embeddings = intra_task_embeddings.unsqueeze(0)
            B, S, E = intra_task_embeddings.shape
        
        if len(src_key_padding_mask.shape) == 1:
            src_key_padding_mask = src_key_padding_mask.unsqueeze(0)
        
        # 1. 准备 [CLS] Token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(intra_task_embeddings.device) # Shape: [B, 1, E]

        # 2. 拼接 [CLS] Token 到序列开头
        x = torch.cat((cls_tokens, intra_task_embeddings), dim=1) # Shape: [B, S + 1, E]

        # 3. 准备 Padding Mask for [CLS] token
        # [CLS] token is never padded, so add a 'False' column at the beginning
        # src_key_padding_mask shape: [B, S]
        # cls_mask shape: [B, 1] (all False)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=intra_task_embeddings.device)
        # full_padding_mask shape: [B, S + 1]
        full_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # 4. 添加位置编码 (只添加到序列长度 S+1)
        # pos_embed shape: [1, max_len + 1, E]
        # x shape: [B, S + 1, E]
        # 我们需要截取或确保 pos_embed 匹配当前输入的 S+1
        # 如果 S (max_len in batch) <= self.seq_len_with_cls - 1
        x = x + self.pos_embed[:, :S + 1, :]

        # 5. 通过 Transformer Encoder Layer，传入 Padding Mask
        # transformer_output shape: [B, S + 1, E]
        transformer_output = self.transformer_encoder(
            x,
            src_key_padding_mask=full_padding_mask # 关键！传入掩码
        )

        # 6. 提取 [CLS] Token 输出 (仍然是第一个位置)
        cls_output = transformer_output[:, 0] # Shape: [B, E]

        # 7. (可选) LayerNorm
        task_embedding = self.norm(cls_output)

        return task_embedding

class FusionModule(nn.Module):
    # ... (选择 FiLM 或 Cross-Attention 实现) ...
    def __init__(self, exg_emb_dim, vlm_projected_dim, fused_emb_dim, fusion_type='film'):
      super().__init__()
      self.fusion_type = fusion_type
      self.fused_emb_dim = fused_emb_dim
      if fusion_type == 'film':
          self.gamma_mlp = nn.Linear(vlm_projected_dim, exg_emb_dim)
          self.beta_mlp = nn.Linear(vlm_projected_dim, exg_emb_dim)
          if exg_emb_dim != fused_emb_dim:
               self.output_proj = nn.Linear(exg_emb_dim, fused_emb_dim)
          else:
               self.output_proj = nn.Identity()
      elif fusion_type == 'cross_attention':
            # 设置 B: Q=VLM, K=V=ExG
            self.vlm_proj_dim = vlm_projected_dim
            self.exg_dim = exg_emb_dim
            self.cross_attn = nn.MultiheadAttention(embed_dim=self.vlm_proj_dim,
                                                    kdim=self.exg_dim,
                                                    vdim=self.exg_dim,
                                                    num_heads=8, # 示例头数
                                                    batch_first=True)
            self.ff = nn.Linear(self.vlm_proj_dim, fused_emb_dim) # 调整输出维度
            self.norm = nn.LayerNorm(self.vlm_proj_dim) # Norm 维度也要匹配Q
      else:
          raise ValueError("Unsupported fusion_type")

    def forward(self, exg_emb, vlm_proj_emb):
      if self.fusion_type == 'film':
          gamma = self.gamma_mlp(vlm_proj_emb)
          beta = self.beta_mlp(vlm_proj_emb)
          fused = gamma * exg_emb + beta
          fused = self.output_proj(fused)
      elif self.fusion_type == 'cross_attention':
          vlm_q = vlm_proj_emb.unsqueeze(1)
          exg_kv = exg_emb.unsqueeze(1)
          attn_output, _ = self.cross_attn(query=vlm_q, key=exg_kv, value=exg_kv)
          # 残差连接可选，这里简化为直接处理输出
          fused = self.norm(attn_output.squeeze(1))
          fused = self.ff(fused)
      return fused