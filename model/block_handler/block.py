import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Block():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, in_feat, out_feat, normalize=True, last = False):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.normalize = normalize
        self.last = last
        self.layers = None

############################################################################################
########################################   BASIC  ##########################################
############################################################################################

class Basic(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
    
        self.layers = [nn.Linear(self.in_feat, self.out_feat)]
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        if self.last == False:
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

############################################################################################
######################################   Residual  #########################################
############################################################################################

class Residual(Block, nn.Module):
    def __init__(self, in_feat, out_feat, normalize=True):
        Block.__init__(self, in_feat, out_feat, normalize)
        nn.Module.__init__(self)
        
        half_feat = in_feat // 2

        # 차원 축소 및 복원
        self.reduce_expand = nn.Sequential(
            nn.Linear(in_feat, half_feat),
            nn.BatchNorm1d(half_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(half_feat, half_feat),
            nn.BatchNorm1d(half_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(half_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 최종 출력 레이어
        self.final_layer = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 차원 축소 및 복원 후 입력 텐서와 합산
        reduced = self.reduce_expand(x)
        residual = reduced + x  # x와 reduce_expand(x)의 결과를 합산
        # 합산된 텐서를 out_feat 크기로 변환
        out = self.final_layer(residual)
        return out

############################################################################################
#####################################   Transformer  #######################################
############################################################################################

class TransformerBlock(Block, nn.Module):
    def __init__(self, in_feat, out_feat=None, normalize=True, last=False, nhead=8, dropout=0.1):
        Block.__init__(self, in_feat, out_feat, normalize)
        nn.Module.__init__(self)

        self.attention = nn.MultiheadAttention(embed_dim = self.in_feat, num_heads=nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.in_feat, 2*self.in_feat),
            nn.ReLU(),
            nn.Linear(2*self.in_feat, self.in_feat)
        )
        
        if self.normalize:
            self.norm1 = nn.LayerNorm(self.in_feat)
            self.norm2 = nn.LayerNorm(self.out_feat)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if not self.last:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.output_fc = nn.Linear(self.in_feat, self.out_feat)


    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)
        if self.normalize:
            x = self.norm1(x)

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        
        x = self.output_fc(x)
        if self.normalize:
            x = self.norm2(x)
        
        if not self.last:
            x = self.activation(x)

        return x
    
############################################################################################
########################################   Resnet   ########################################
############################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(Block, nn.Module):
    def __init__(self, in_feat, out_feat, normalize=True, stride=1, num_blocks=2):
        Block.__init__(self, in_feat, out_feat, normalize)
        nn.Module.__init__(self)
        
        self.in_planes = in_feat

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            downsample = None
            if i == 0 and (stride != 1 or self.in_planes != out_feat * BasicBlock.expansion):
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, out_feat * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_feat * BasicBlock.expansion),
                )
            self.layers.append(BasicBlock(self.in_planes, out_feat, stride if i == 0 else 1, downsample))
            self.in_planes = out_feat * BasicBlock.expansion

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_planes, out_feat, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

############################################################################################
########################################    VGG   ##########################################
############################################################################################

class VGG(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)

        self.layers = [
            nn.Conv2d(self.in_feat, self.out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat, self.out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat, self.out_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat),
            nn.ReLU(inplace=True)
        ]

############################################################################################
######################################    Dense   ##########################################
############################################################################################

class DenseLayer(Block, nn.Module):
    def __init__(self, in_feat, growth_rate, normalize=True):
        Block.__init__(self, in_feat, growth_rate, normalize)
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_feat, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)

class Dense(Block, nn.Module):
    def __init__(self, num_layers, in_feat, growth_rate, out_feat, normalize=True):
        Block.__init__(self, in_feat, out_feat, normalize)
        nn.Module.__init__(self)
        
        self.layers = nn.ModuleList()
        current_channels = in_feat

        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate, normalize=normalize))
            current_channels += growth_rate

        # 마지막 레이어에서 원하는 출력 채널 수에 맞추기 위한 1x1 컨볼루션 트랜스포즈 레이어 추가
        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(current_channels, out_feat, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
    
############################################################################################
#####################################   Inception   ########################################
############################################################################################
    
class InceptionV1(Block, nn.Module):  # Ensure InceptionV1 inherits from Block and nn.Module
    def __init__(self, in_feat, out_feat, normalize, last):
        Block.__init__(self, in_feat, out_feat, normalize, last)
        nn.Module.__init__(self)
        
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(in_feat, out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(in_feat, out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_feat // 4, out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(in_feat, out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_feat // 4, out_feat // 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_feat, out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_feat // 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
class InceptionV2(Block, nn.Module):  # Ensure InceptionV2 inherits from Block and nn.Module
    def __init__(self, in_feat, out_feat, normalize=True, last=False):
        Block.__init__(self, in_feat, out_feat, normalize, last)
        nn.Module.__init__(self)
        
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
class InceptionV3(Block, nn.Module):
    def __init__(self, in_feat, out_feat, normalize, last):
        Block.__init__(self, in_feat, out_feat, normalize, last)
        nn.Module.__init__(self)
        
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionV4(Block, nn.Module):
    def __init__(self, in_feat, out_feat, normalize, last):
        Block.__init__(self, in_feat, out_feat, normalize, last)
        nn.Module.__init__(self)

        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

############################################################################################
######################################    DCMv3   ##########################################
############################################################################################
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
    
class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
    
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = torch.nn.functional.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale
    
    
class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=1024,
            out_channels=512,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0, 
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
            "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
            "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.out_channels=out_channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        
        self.offset = nn.Linear( 
            channels,
            group * kernel_size * kernel_size * 2)
        
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, self.out_channels)
        self._reset_parameters()

        # creates a trainable parameter center_feature_scale_proj_weight with shape 
        # (group, channels) and parameter center_feature_scale_proj_bias with shape 
        # (group, ) initialized with zeros.
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        # Linear projection of the input feature map
        x = self.input_proj(input)
        x_proj = x # Preserve a reference for later use

        # Permute input dimensions for depthwise convolution
        x1 = input.permute(0, 3, 1, 2) # (N, C, H, W)
        x1 = self.dw_conv(x1) # Apply depthwise convolution
        offset = self.offset(x1) # Compute offsets for deformable convolution

        # self.mask(x1) outputs (N, H, W, self.group * kernel_size * kernel_size)
        # The output is reshaped to have dimensions (N, H, W, self.group, -1). 
        # This reshaping is performed to create a set of masks for each group and position 
        # in the output feature map.
        mask = self.mask(x1).reshape(N, H, W, self.group, -1) 
        # Applies the softmax function along the last dimension
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        # Perform deformable convolution using the core operation
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)

        # Optionally, apply center feature scaling
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                # linearly projects x1
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            
            # reshapes and repeats 'center_feature_scale' to be compatible with 'x'
            # N, H, W, groups -> N, H, W, groups, 1 -> 
            # N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)

            # Apply center feature scaling to the output
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
            #  modulate the importance of features at different spatial locations 
        
        # Linear projection of the output
        x = self.output_proj(x)
      
        return x

    
def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, 
                          pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes

    # Calculate the output dimensions after convolution
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    # Generate grid of reference points in the input space
    # ref_x and ref_y represent ~ HxW
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))

    # Flattens and normalizes the reference points to the range [0, 1]
    ref_y = ref_y.reshape(-1)[None] / H_ 
    ref_x = ref_x.reshape(-1)[None] / W_

    # Stack the reference points and reshape for compatibility with deformable convolution
    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref

def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []

    # Generate a meshgrid of coordinates based on kernel size and dilation
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))

    # Normalize the coordinates to the range [-1, 1]
    points_list.extend([x / W_, y / H_])

    # Stack the normalized coordinates and reshape for deformable convolution compatibility
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid

def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    
    # Pad input feature map
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])

    # Extract input dimensions
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    # Compute reference points and dilation grids
    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, 
        stride_h, stride_w) # (1, H_out, W_out, 1, 2)
    
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, 
        group, input.device) # (1, 1, 1, group * kernel_h * kernel_w, 2).
    
    # Compute spatial normalization factors
    # (1, 1, 1, group * kernel_h * kernel_w * 2) 
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device)
    
    # Compute sampling locations
    # (N, H_out, W_out, group * kernel_h * kernel_w * 2)
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) \
    + offset * offset_scale / spatial_norm
    
    # Calculate constants
    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1

    # Reshape input 
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> 
    # N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # (group*N, group_channels, H_out + padding, W_out + padding)
    
    # Reshape sampling grid 
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> 
    # N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # (group*N, H_out * W_out, kernel_size * kernel_size, 2)

    # Reshape sampling_grid, perform bilinear interpretation if points are not available
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', 
        align_corners=False) # (group*N, group_channels, H_out * W_out, kernel_size * kernel_size)
    
    # Reshape mask
    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> 
    # (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    # (group*N, 1, H_out * W_out, kernel_size * kernel_size)

    # for all {i,j}, sum over groups: x_g(p_{i,j} + location-aware offsets) * m_{g,k}(i,j)
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, H_out*W_out) 
                                # (N, channels, H_out * W_out)
    
    # Transpose and reshape the output
    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous() # (N, H_out, W_out, channels)

class DCNv3(Block, nn.Module):
    def __init__(self, in_feat, out_feat, normalize, last):
        Block.__init__(self, in_feat, out_feat, normalize, last)
        nn.Module.__init__(self)

        self.layers =DCNv3_pytorch(channels=self.in_feat,out_channels=self.out_feat, kernel_size=3)

    def forward(self, x):
        # x: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
    
        new_features = self.layers(x)

        # x: (N, H, W, C) -> (N, C, H, W)
        new_features = new_features.permute(0, 3, 1, 2)
        return new_features
