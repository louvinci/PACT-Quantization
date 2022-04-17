import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_fn import act_pactq,activation_quant,Conv2d_Q,Linear_Q
Conv2d=Conv2d_Q
BatchNorm2d = nn.BatchNorm2d

class Block(nn.Module):
    def __init__(self, ain_bit,aout_bit,w_bit,in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes     = expansion * in_planes
        self.conv1 = Conv2d(w_bit,in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(planes)
        self.act_quant1 = activation_quant(a_bit=ain_bit)
        #self.act_quant1 = act_pactq(a_bit=ain_bit,fixed_rescale=6.0)

        self.conv2 = Conv2d(w_bit,planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2   = BatchNorm2d(planes)
        self.act_quant2 = activation_quant(a_bit=ain_bit)
        #self.act_quant2 = act_pactq(a_bit=ain_bit,fixed_rescale=6.0)

        self.conv3 = Conv2d(w_bit,planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = BatchNorm2d(out_planes)
        self.act_quant3 = activation_quant(a_bit=aout_bit)
        #self.act_quant3 = act_pactq(a_bit=ain_bit,fixed_rescale=6.0)

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.act_quant4=activation_quant(a_bit=aout_bit)
            self.shortcut = nn.Sequential(
                Conv2d(w_bit,in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.act_quant1(out)

        out = self.bn2(self.conv2(out))
        out = self.act_quant2(out)

        out = self.bn3(self.conv3(out))
        out = out + self.act_quant4(self.shortcut(x)) if self.stride==1 else out
        #out = out + self.shortcut(x) if self.stride==1 else out # this version we don't quantize the output of the branch
        out = self.act_quant3(out)
        return out


class MobileNetV2_CF(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  #stride=1 for cifar10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, config):
        super(MobileNetV2_CF, self).__init__()
        self.layer_abit = config.layer_abit
        self.layer_wbit = config.layer_wbit

        self.actq_input = activation_quant(self.layer_abit[0]) #32
        self.stem = Conv2d(self.layer_wbit[0],3, 32, kernel_size=3, stride=1, padding=1, bias=False) # stride=1 for cifar10,default w_bit=32
        self.bn1 = BatchNorm2d(32)
        self.actq_first = activation_quant(self.layer_abit[1])

        self.layers = self._make_layers(in_planes=32)

        self.head = Conv2d(self.layer_wbit[-2],320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(1280)
        self.actq_last = activation_quant(self.layer_abit[-1])

        self.linear = Linear_Q(self.layer_wbit[-1],1280, config.num_classes)
        self._criterion = nn.CrossEntropyLoss()

    def _make_layers(self,in_planes):
        layers = []
        layer_id = 1
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                # ain_bit,aout_bit,w_bit,in_planes, out_planes, expansion, stride
                layers.append(Block(self.layer_abit[layer_id],self.layer_abit[layer_id+1],self.layer_wbit[layer_id],in_planes=in_planes,out_planes= out_planes, expansion=expansion, stride=stride)) 
                in_planes = out_planes
                layer_id+=1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.actq_input(x)
        out = self.bn1(self.stem(x))
        out = self.actq_first(out)

        out = self.layers(out)
        out = self.bn2(self.head(out))
        out = F.avg_pool2d(out, 4)
        out = self.actq_last(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
if __name__=="__main__":
    from easydict import EasyDict
    C = EasyDict()
    C.num_classes=10
    C.layer_abit = [32,  1,  2,3, 4,5,6, 7,8,9,10, 10,11,12, 13,14,15, 16, 32,32]# last 16 s
    C.layer_wbit = [32,  10, 2,3, 4,5,6, 7,8,9,10, 10,11,12, 13,14,15, 16, 16,16]
    net = MobileNetV2_CF(C)
    print(net)
