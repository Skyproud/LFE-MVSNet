import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.deform_conv import DeformConv2dPack

from modules.SCConv import ScConv
from wtconv.wtconv2d import WTConv2d
from torch.autograd import Function
import pywt
from modules.HWD import Down_wt

class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, in_affine=True, num_groups=8, **kwargs):
        super(Conv2dUnit, self).__init__()
        bias = False
        if norm is None:
            bias = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm2d(out_channels, momentum=in_momentum, affine=in_affine)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class WTD_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(WTD_Conv2d, self).__init__()
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=7, wt_levels=3)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                              bias=False)
        self.down = Down_wt(out_channels, out_channels)
        self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm_layer(x)
        x = F.relu(x)
        x = self.down(x)
        return x

class WTD_Conv2d_WOdown(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(WTD_Conv2d_WOdown, self).__init__()
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=7, wt_levels=3)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                              bias=False)
        self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm_layer(x)
        x = F.relu(x)
        return x


class Res_Unit(nn.Module):
    """Res block"""
    def __init__(self, in_channels, out_channels):
        super(Res_Unit, self).__init__()
        self.conv = WTD_Conv2d(in_channels, out_channels)
        self.SCconv = ScConv(op_channel=out_channels)
        self.LeackyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        y = self.SCconv(x)
        x = x + y
        x = self.LeackyReLU(x)
        return x


class Res_Unit2(nn.Module):
    """Res block"""
    def __init__(self, in_channels, out_channels):
        super(Res_Unit2, self).__init__()
        self.conv = WTD_Conv2d_WOdown(in_channels, out_channels)
        self.SCconv = ScConv(op_channel=out_channels)
        self.LeackyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        y = self.SCconv(x)
        x = x + y
        x = self.LeackyReLU(x)
        return x


class Res_Unit1(nn.Module):
    """Res block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bn_momentum=0.1):
        super(Res_Unit1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.SCconv = ScConv(op_channel=out_channels)
        self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.LeackyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.LeackyReLU(x)
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.LeackyReLU(x)
        # print(x.shape)
        y = self.SCconv(x)
        x = x + y
        x = self.LeackyReLU(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # Residual connection
        return out2

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class Fusion(nn.Module):
    """"x1:(B,8,64,64) X2:(B,8,32,32)"""
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = ResNet(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = ResNet(in_channels)


        self.idwt = IDWT_2D(wave)


    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, 1)
        high = torch.cat([lh, hl, hh], 1)
        high1=self.convh1(high)
        high2= self.high(high1)
        highf=self.convh2(high2)
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape

        #
        if(h1!=h2):
            x2 =F.pad(x2, (0, 0, 1, 0), "constant", 0)


        low=torch.cat([ll, x2], 1)
        low = self.convl(low)
        lowf=self.low(low)

        out = torch.cat((lowf, highf), 1)
        out_idwt = self.idwt(out)

        return out_idwt

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=bias, **kwargs)
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm2d(out_channels, momentum=in_momentum, affine=in_affine)
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.norm_layer is not None:
            y = self.norm_layer(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

class Conv3dUnit(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, **kwargs)
        
        
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm3d(out_channels, momentum=in_momentum, affine=in_affine)

        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv3dUnit(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        # assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=bias, **kwargs)

        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm3d(out_channels, momentum=in_momentum, affine=in_affine)

        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.norm_layer is not None:
            y = self.norm_layer(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, 
        norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
        in_affine=True, num_groups=8):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
            relu=relu, norm=norm, bn_momentum=bn_momentum, in_momentum=in_momentum, 
            in_affine=in_affine, num_groups=num_groups)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
            relu=relu, norm=norm, bn_momentum=bn_momentum, in_momentum=in_momentum, 
            in_affine=in_affine, num_groups=num_groups)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class StageFeatExtNet(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(StageFeatExtNet, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [32, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        # [B,32,H/4,W/4]
        self.conv2 = nn.Sequential(
                Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = nn.Sequential(
            Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            )

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 8, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 8, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 4, self.output_channels[2], 1, bias=False)
            self.conv_out["3"] = nn.Conv2d(base_channels * 2, self.output_channels[3], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)
        
        if stage_num == 3:
            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 4, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 4, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 2, self.output_channels[2], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)

    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)
            output_feature["0"] = self.conv_out["0"](conv3)
            intra_feat = F.interpolate(conv3, scale_factor=2, mode="bilinear") + self.conv_inner["1"](conv2)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner["2"](conv1)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner["3"](conv0)
            output_feature["3"] = self.conv_out["3"](intra_feat)

        if self.stage_num == 3:
            output_feature["0"] = self.conv_out["0"](conv2)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner["1"](conv1)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner["2"](conv0)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            
        return output_feature




class SC_StageNet(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(SC_StageNet, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [32, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        # [B,8,H,W]
        self.conv0 = Res_Unit2(3, base_channels)
        # [B,16,H/2,W/2]
        self.conv1 = Res_Unit(base_channels, base_channels * 2)
        # [B,32,H/4,W/4]
        self.conv2 = Res_Unit(base_channels * 2, base_channels * 4)

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = Res_Unit(base_channels * 4, base_channels * 8)

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 8, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 8, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 4, self.output_channels[2], 1, bias=False)
            self.conv_out["3"] = nn.Conv2d(base_channels * 2, self.output_channels[3], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)
            self.fusion1 = Fusion(base_channels * 8, wave='haar')
            self.fusion2 = Fusion(base_channels * 4, wave='haar')
            self.fusion3 = Fusion(base_channels * 2, wave='haar')


    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        up_channel1 = self.conv_inner["1"](conv2)
        up_channel2 = self.conv_inner["2"](conv1)
        up_channel3 = self.conv_inner["3"](conv0)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)
            output_feature["0"] = self.conv_out["0"](conv3)
            fusion_feat1 = self.fusion1(up_channel1, conv3)
            output_feature["1"] = self.conv_out["1"](fusion_feat1)
            fusion_feat2 = self.fusion2(up_channel2, conv2)
            output_feature["2"] = self.conv_out["2"](fusion_feat2)
            fusion_feat3 = self.fusion3(up_channel3, conv1)
            output_feature["3"] = self.conv_out["3"](fusion_feat3)

        return output_feature

class SC_StageNet1(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(SC_StageNet1, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [64, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num


        # [B,8,H,W]
        self.conv0 = Res_Unit2(3, base_channels)
        # [B,16,H/2,W/2]
        self.conv1 = Res_Unit(base_channels, base_channels * 2)
        # [B,32,H/4,W/4]
        self.conv2 = Res_Unit(base_channels * 2, base_channels * 4)

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = Res_Unit(base_channels * 4, base_channels * 8)

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = DeformConv2dPack(in_channels=base_channels * 8, out_channels=self.output_channels[0],
                                                  kernel_size=3, stride=1, padding=1, deform_groups=1)
            self.conv_out["1"] = DeformConv2dPack(base_channels * 8, self.output_channels[1], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)
            self.conv_out["2"] = DeformConv2dPack(base_channels * 4, self.output_channels[2], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)
            self.conv_out["3"] = DeformConv2dPack(base_channels * 2, self.output_channels[3], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)
            self.fusion1 = Fusion(base_channels * 8, wave='haar')
            self.fusion2 = Fusion(base_channels * 4, wave='haar')
            self.fusion3 = Fusion(base_channels * 2, wave='haar')


    def forward(self, x):

        output_feature = {}
        conv0 = self.conv0(x) # [B,8,H,W]
        conv1 = self.conv1(conv0) # [B,16,H/2,W/2]
        conv2 = self.conv2(conv1) # [B,32,H/4,W/4]
        up_channel1 = self.conv_inner["1"](conv2)
        up_channel2 = self.conv_inner["2"](conv1)
        up_channel3 = self.conv_inner["3"](conv0)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)  # 64,H/8,W/8
            output_feature["0"] = self.conv_out["0"](conv3)
            fusion_feat1 = self.fusion1(up_channel1, conv3)
            output_feature["1"] = self.conv_out["1"](fusion_feat1) # 32,H/4,W/4
            fusion_feat2 = self.fusion2(up_channel2, conv2)
            output_feature["2"] = self.conv_out["2"](fusion_feat2)
            fusion_feat3 = self.fusion3(up_channel3, conv1)
            output_feature["3"] = self.conv_out["3"](fusion_feat3)

        return output_feature


class SC_StageNet2(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(SC_StageNet2, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [64, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num


        # [B,8,H,W]
        self.conv0 = Res_Unit2(3, base_channels)
        # [B,16,H/2,W/2]
        self.conv1 = Res_Unit(base_channels, base_channels * 2)
        # [B,32,H/4,W/4]
        self.conv2 = Res_Unit(base_channels * 2, base_channels * 4)

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = Res_Unit(base_channels * 4, base_channels * 8)

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = DeformConv2dPack(in_channels=base_channels * 8, out_channels=self.output_channels[0],
                                                  kernel_size=3, stride=1, padding=1, deform_groups=1)
            self.conv_out["1"] = DeformConv2dPack(base_channels * 8, self.output_channels[1], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)
            self.conv_out["2"] = DeformConv2dPack(base_channels * 4, self.output_channels[2], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)
            self.conv_out["3"] = DeformConv2dPack(base_channels * 2, self.output_channels[3], kernel_size=3, stride=1,
                                                  padding=1, deform_groups=1)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)



    def forward(self, x):

        output_feature = {}
        conv0 = self.conv0(x) # [B,8,H,W]
        conv1 = self.conv1(conv0) # [B,16,H/2,W/2]
        conv2 = self.conv2(conv1) # [B,32,H/4,W/4]
        up_channel1 = self.conv_inner["1"](conv2)
        up_channel2 = self.conv_inner["2"](conv1)
        up_channel3 = self.conv_inner["3"](conv0)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)  # 64,H/8,W/8
            output_feature["0"] = self.conv_out["0"](conv3)
            fusion_feat1 = F.interpolate(conv3, scale_factor=2, mode="bilinear") + up_channel1 # 64,H/4,W/4
            output_feature["1"] = self.conv_out["1"](fusion_feat1) # 32,H/4,W/4
            fusion_feat2 = F.interpolate(conv2, scale_factor=2, mode="bilinear") + up_channel2
            output_feature["2"] = self.conv_out["2"](fusion_feat2)
            fusion_feat3 = F.interpolate(conv1, scale_factor=2, mode="bilinear") + up_channel3
            output_feature["3"] = self.conv_out["3"](fusion_feat3)

        return output_feature




class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels=1):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1, norm="groupnorm")

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1, norm="groupnorm")

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1, norm="groupnorm")

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1, norm="groupnorm")

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.prob = nn.Conv3d(base_channels, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class CostRegNetBN(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels=1):
        super(CostRegNetBN, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1, norm="batchnorm")

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1, norm="batchnorm")

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1, norm="batchnorm")

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1, norm="batchnorm")

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.prob = nn.Conv3d(base_channels, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class FeatureFetcher(nn.Module):
    def __init__(self, mode="bilinear", align_corners=True):
        super(FeatureFetcher, self).__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """adapted from 
        https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/feature_fetcher.py

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            # hanlding z == 0.0
            # following https://github.com/kwea123/CasMVSNet_pl/blob/5813306b451b22226a321d347e5f6020a20ae8c9/models/modules.py#L75
            z_zero_mask = (z == 0.0)
            z[z_zero_mask] = 1.0

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            # hanlding z == 0.0
            uv[:, :, 0][z_zero_mask] = 2 * width
            uv[:, :, 1][z_zero_mask] = 2 * height

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode, align_corners=self.align_corners)
        pts_feature = pts_feature.squeeze(3)

        pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)

        return pts_feature


def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0.5, width - 0.5, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0.5, height - 0.5, height).view(height, 1).expand(height, width)
        # y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, in_channels):
        super(PixelwiseNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, 16, kernel_size=1, stride=1, padding=0, norm="batchnorm")
        self.conv1 = Conv3dUnit(16, 8, kernel_size=1, stride=1, padding=0, norm="batchnorm")
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        

    def forward(self, x1):
        # x1: [B, G, Ndepth, H, W]
        
        # [B, Ndepth, H, W]
        x1 =self.conv2(self.conv1(self.conv0(x1))).squeeze(1)
        
        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]
        
        return output.unsqueeze(1) # [B,1,H,W]


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 512, 640)
    x2 = torch.randn(1, 16, 64, 80) # 输入 B C H W
    model = WTD_Conv2d(3, 8, 8)
    model1 = Res_Unit(3, 8)
    model3 = SC_StageNet1(8)
    model2 = Fusion(64, wave='haar')
    x = model3(x1)
