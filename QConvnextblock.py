from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from quant_fn import Linear_Q,Conv2d_Q,activation_quant
Conv2d=Conv2d_Q
BatchNorm2d = nn.BatchNorm2d


class ConvnextBlock(nn.Module):
    '''
    Original ConvNext: DW7x7 (96-96)+ LayerNorm 
                       -> Conv1x1(96-384) + GELU -> Conv1x1(384-96)
    Modified ConvNext: DW7x7 (cin-cout)+ BatchNorm 
                       -> Conv1x1(cin-cin*e) + RELU 
                       -> Conv1x1(cin*e-cout)
    '''
    def __init__(self,a_bit,w_bit,C_in,C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvnextBlock,self).__init__()
        self.C_in = C_in
        self.C_out = C_out


        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        
        # when stride =2, c_in must be different form c_out, the opposite is not true,but one norm layer seems no matter
        if stride == 2 or C_in != C_out:
            self.norm = BatchNorm2d(C_in)
            if w_bit ==32:
                self.act_quant1 = activation_quant(32)# don't quantize
            else:
                self.act_quant1 = activation_quant(16)#! Act_quant layer after the BN layer is ok?
            self.downsample = Conv2d(w_bit,C_in,C_out,kernel_size=stride,stride=stride,padding=0,dilation=1,groups=1,bias=False)
        
        if padding is None:
            # assume h_out = h_in / s, p =( k-s) /2
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - 1) / 2.))
        else:
            self.padding = padding

        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv1 = Conv2d(w_bit,C_out, C_out, kernel_size=self.kernel_size, stride=1, padding=self.padding, dilation=1, groups=C_out, bias=bias)
        self.bn1 = BatchNorm2d(C_out)

        self.conv2 = Conv2d(w_bit,C_out,C_out*expansion,kernel_size=1,stride=1,padding=0,dilation=1,groups=self.groups,bias=bias)
        self.act2=nn.ReLU(inplace=True)

        self.conv3 = Conv2d(w_bit,C_out*expansion,C_out,kernel_size=1,stride=1,padding=0,dilation=1,groups=self.groups,bias=bias)
        self.act_quant2 = activation_quant(a_bit)
        
    def forward(self,x):
        #default the x is quantized
        if self.stride == 2 or self.C_in != self.C_out:
            x = self.norm(x)
            x = self.act_quant1(x)
            x = self.downsample(x) # output the q_x
        identity = x
        x = self.bn1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        x += identity
        x = self.act_quant2(x)

        return x


class ConvNorm(nn.Module):
    def __init__(self, a_bit,w_bit,C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,):
        super(ConvNorm,self).__init__()

        assert stride in [1, 2]
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int

        self.conv = Conv2d(w_bit,C_in, C_out, kernel_size=kernel_size, stride=stride, padding=self.padding, 
                            dilation=self.dilation, groups=groups, bias=bias)
        self.bn = BatchNorm2d(C_out)
        self.act_quant=activation_quant(a_bit=a_bit)
    
    def forward(self,x):
        q_x = self.conv(x)
        x = self.bn(q_x)
        q_x = self.act_quant(x)
        return q_x

if __name__ == "__main__":
    abit,wbit=8,32
    layer = ConvnextBlock(a_bit=abit,w_bit=wbit,C_in=6,C_out=16,kernel_size=3,expansion=4,stride=1,padding=1)
    stem = ConvNorm(abit,wbit,C_in=3,C_out=6,kernel_size=3,padding=None)
    print(layer)
    print(stem)
    a = torch.rand(2, 3, 7, 7)
    stem_a = stem(a)
    layer_a = layer(stem_a)

    d = torch.mean(layer_a)

    d.backward()

    
    
    
    