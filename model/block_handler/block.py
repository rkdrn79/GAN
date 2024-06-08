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

class TransformerBlock(Block):
    def __init__(self, in_feat, out_feat,  nhead, num_layers, dim_feedforward=2048):
        super().__init__(in_feat, out_feat)
        self.layers = []
        encoder_layers = nn.TransformerEncoderLayer(in_feat, nhead, dim_feedforward)

        self.layers.append(PositionalEncoding(in_feat))
        self.layers.append(nn.TransformerEncoder(encoder_layers, num_layers))
        self.layers.append(nn.Linear(self.in_feat, self.out_feat))
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
############################################################################################
########################################    CNN   ##########################################
############################################################################################

class CNN(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
    
        self.layers = [nn.Conv2d(self.in_feat, self.out_feat, kernel_size=3, stride=1, padding=1, bias=False)]
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        if self.last == False:
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

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
    
class InceptionV1(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
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
    
class InceptionV2(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
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
    
class InceptionV3(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
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

class InceptionV4(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_feat // 4, self.out_feat // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_feat // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_feat, self.out_feat // 4, kernel_size=1, stride=1),
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