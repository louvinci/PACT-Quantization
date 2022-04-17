import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_fn import act_pactq,activation_quant,Conv2d_Q,Linear_Q
Conv2d=Conv2d_Q
BatchNorm2d = nn.BatchNorm2d


class ResBlock_Q(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 wbit,
                 abit,
                 stride):
        super(ResBlock_Q, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.conv1 = Conv2d(
            w_bit=wbit,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.act_q1 = activation_quant(a_bit=abit)

        self.conv2 = Conv2d(
            w_bit = wbit,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.act_q2 = activation_quant(a_bit=abit)

        if self.resize_identity:
            self.identity_conv = Conv2d(
                w_bit=wbit,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride)
            self.bn3 = BatchNorm2d(out_channels)
            self.act_q3 = activation_quant(a_bit =abit)
        

    def forward(self, x):
        if self.resize_identity:
            indentity = self.act_q3(self.bn3(self.identity_conv(x)))
        else:
            indentity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.act_q1(x)
        x = self.bn2(self.conv2(x))
        x+=indentity
        x = self.act_q2(x)

        return x


class CIFARResNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 nbit_wlist,
                 nbit_alist,
                 first_wbit=8,
                 first_abit=8,
                 last_wbit=8,
                 last_abit=8,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.stem = Conv2d(w_bit=first_wbit,in_channels=in_channels,out_channels=init_block_channels,kernel_size=3,padding=1)
        self.bn1 = BatchNorm2d(init_block_channels)
        self.actq_first = activation_quant(a_bit = first_abit)
        
        self.features = nn.Sequential()
        in_channels = init_block_channels
        block_id=0
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResBlock_Q(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    wbit = nbit_wlist[block_id],
                    abit = nbit_alist[block_id],
                    stride=stride
                    ))
                block_id+=1
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.actq_last = activation_quant(a_bit = last_abit)
        self.output = Linear_Q(
            w_bit = last_wbit, 
            in_features=in_channels,
            out_features=num_classes,
            )

        self._init_params()
        self._criterion = nn.CrossEntropyLoss()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.stem(x)))
        x= self.actq_first(x)

        x = self.features(x)
        x = self.actq_last(x)

        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resnet_cifar_Q(num_classes,
                     blocks,
                     wbit_list,
                     abit_list,
                     first_wbit=16,
                     first_abit=32,
                     last_wbit=16,
                     last_abit=32,
                     **kwargs):
    

    assert (num_classes in [10, 100])

  
    assert ((blocks - 2) % 6 == 0)
    layers = [(blocks - 2) // 6] * 3

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]


    net = CIFARResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        nbit_wlist = wbit_list,
        nbit_alist = abit_list,
        first_wbit = first_wbit,
        first_abit = first_abit,
        last_wbit  = last_wbit,
        last_abit  = last_abit,
        num_classes=num_classes,
        **kwargs)
    return net

def resnet20_cifar10_Q(config, **kwargs):

    listw = config.layer_wbit
    lista = config.layer_abit
    num_classes = config.num_classes
    return get_resnet_cifar_Q(num_classes=num_classes, blocks=20,
                              wbit_list =listw,abit_list=lista,
                              first_wbit=32,first_abit=32,last_wbit=16,last_abit=32,
                              **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from easydict import EasyDict
    C= EasyDict()
    C.layer_abit = [  8,8,8, 8,8,8, 8,8,8]
    C.layer_wbit = [  8,8,8, 8,8,8, 8,8,8]
    C.num_classes=10
    pretrained = False

    models = [
        (resnet20_cifar10_Q, 10),
       # (resnet20_cifar100_Q, 100),
    ]
    
    for model, num_classes in models:

        net = model(C)
        print(net)
        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet20_cifar10_Q or weight_count == 272474)
        #assert (model != resnet20_cifar100 or weight_count == 278324)
        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))
        #print(net)

if __name__ == "__main__":
    _test()