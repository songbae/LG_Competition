import torch.nn as nn 
import torch 
import torch.nn.functional as F
import timm 
from fastai.vision.all import *


class AddCoordinates(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, y_dim, x_dim = input_tensor.size()

        xx_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)

        xx_range = torch.arange(y_dim, dtype=torch.int32)
        yy_range = torch.arange(x_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (y_dim - 1)
        yy_channel = yy_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        if torch.cuda.is_available:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                            torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=1,
                 stride=1, padding=0, with_r=False):
        super(CoordConvNet, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.coord_adder = AddCoordinates(with_r=with_r)
        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x
class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
                           nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2**(len(self.convs)-i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4)
             for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1,
                                                   stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        #encoder
        # m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
        #                    'resnext50_32x4d_ssl')
        m = timm.create_model("swsl_resnext50_32x4d", pretrained=True,in_chans=32)
        self.coord_net=CoordConvNet()
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        #aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[
                         stride*1, stride*2, stride*3, stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        #decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.dec_last=UnetBlock(32,32,16)
        self.fpn = FPN([512, 256, 128, 64,32], [16]*5)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(
            16+16*5, 3, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc_start=self.coord_net(x)
        enc0 = self.enc0(enc_start)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        dec_last=self.dec_last(dec0,enc_start)
        x = self.fpn([enc5, dec3, dec2, dec1,dec0], dec_last)
        x = self.final_conv(self.drop(x))
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

if __name__=='__main__':
  device='cuda' if torch.cuda.is_available() else 'cpu'
  img=np.zeros((2,3,224,224),dtype=np.float32)
  img=torch.from_numpy(img).to(device)
  model=UneXt50().to(device)
  output=model(img)
  print(output.shape)
