#from QConvnextblock import ConvnextBlock,ConvNorm
from HW_QConvnextblock import HW_ConvNorm,HW_ConvnextBlock
from quant_fn import Linear_Q,Conv2d_Q,activation_quant,act_pactq,ActQuant
from quant_fn import fixed_type
import torch
import torch.nn as nn
from torch.nn import init
import sys
class ToyNet_WINT(nn.Module):
    def __init__(self,config):
        super(ToyNet_WINT,self).__init__()
        self.num_classes = config.num_classes
        self.strides = config.strides
        self.num_layer_list   = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel
        self.layer_abit = config.layer_abit
        self.layer_wbit = config.layer_wbit
        self.cells = nn.ModuleList()
        self.act_qinput=activation_quant(self.layer_abit[0])
        #self.act_qinput= act_pactq(self.layer_abit[0],fixed_rescale=10)
        #self.act_qinput= ActQuant(self.layer_abit[0],scale_coef=10.0)
        
        self.stem = HW_ConvNorm(self.layer_abit[0],self.layer_abit[1],self.layer_wbit[0],C_in=3,C_out=self.stem_channel,kernel_size=3,padding=1,stride=1,bias=False)
        
        t_cin, t_cout = 0,0 
        layer_id =1
        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        t_cin,t_cout = self.stem_channel,self.num_channel_list[stage_id]
                    else:
                        t_cin,t_cout = self.num_channel_list[stage_id-1],self.num_channel_list[stage_id]
                    t_stride = self.strides[stage_id]
                else:
                    t_cin,t_cout = self.num_channel_list[stage_id],self.num_channel_list[stage_id]
                    t_stride = 1
                #TODO change1
                #HW_ConvnextBlock(ainbit,aoutbit,wbit,C_in=C_in,C_out=C_out,expansion=2,kernel_size=3,stride=stride)
                block = HW_ConvnextBlock(self.layer_abit[layer_id],self.layer_abit[layer_id+1],self.layer_wbit[layer_id],t_cin,t_cout,expansion = 4, kernel_size=3,stride=t_stride)
                self.cells.append(block)
                layer_id+=1
        #TODO change2
        self.header = HW_ConvNorm(self.layer_abit[-2],self.layer_abit[-1],self.layer_wbit[-2],self.num_channel_list[-1], self.header_channel, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.act_qlast= activation_quant(a_bit = self.layer_abit[-1])
        #self.act_qlast= act_pactq(a_bit = self.layer_abit[-1],fixed_rescale=10)
        self.act_qlast= ActQuant(a_bit = self.layer_abit[-1],scale_coef=6.0)
        
        self.fc = Linear_Q(self.layer_abit[-1],self.layer_wbit[-1],self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,input):
        # the input of the first layer can choose quantize or not,usually not quantize
        q_input = self.act_qinput(input)
        out = self.stem(q_input)
        for i,cell in enumerate(self.cells):
            out = cell(out)
            # print("SW debug")
            # out = out.cpu().numpy()
            # print(out[0][0],file=open("sw_cell{}.txt".format(i),'w+'))
            # sys.exit(0)

        out = self.header(out)
        out = self.avgpool(out)
        out = self.act_qlast(out)
        out = self.fc(out.view(out.size(0),-1))
        #the last avgpool and fc layer be quantized to 16bits
        return out

    #TODO
    def INTforward(self,x):
        #91.366% not clip, trunc:91.05%
        out = self.stem.INTforward(x,alpha_in=1.0,bit=32,intbit=16)
        alpha_in = self.stem.act_quant_out.scale_coef.item()

        for i, cell in enumerate(self.cells):
            out = cell.INTforward(out,alpha_in,bit=32,intbit=10,BNbit=32,BNintbit=10)
            alpha_in = cell.act_quant_out.scale_coef.item()
        alpha_in = self.cells[-1].act_quant_out.scale_coef.item()

        out = self.header.INTforward(out,alpha_in,bit=32,intbit =16)
        out = out.float()
        out = self.avgpool(out)
        out = out.long()
        alpha_in = self.header.act_quant_out.scale_coef.item()
        inbit, outbit = self.header.act_quant_out.bit, self.act_qlast.bit
        alpha_out = self.act_qlast.scale_coef.item()
        scale_in  = (2**(inbit-1)-1)  / alpha_in
        scale_out = (2**(outbit-1)-1) / alpha_out
        recale = scale_out/scale_in
        bit,intbit = 32,16
        q_rescale = fixed_type(recale,bit,intbit)
        out = torch.div( q_rescale*out, 2**(bit-intbit),rounding_mode='floor') #trunc
        out = torch.clamp(out,-2**(outbit-1),2**(outbit-1))
        
        out = out.view(out.size(0),-1)
        out   = self.fc.INTforward(out,alpha_out)
#*############Replece Cells###################

        # for i, cell in enumerate(self.cells):
        #     #block.INTforward(a,alpha_in,32,16,BNbit=32,BNintbit=16)
        #     out = cell.INTforward(out,alpha_in,bit=32,intbit=10,BNbit=32,BNintbit=10)
        #     alpha_in = cell.act_quant_out.scale_coef.item()
        #     # print("HW debug",alpha_in)
        #     # out = out*1.0/((2**(cell.aout_bit-1)-1)/alpha_in)
        #     # out = out.cpu().numpy()
        #     # print(out[0][0],file=open("hw_cell{}_1.txt".format(i),'w+'))
        #     # sys.exit(0)

        # out = out*1.0/((2**(self.cells[-1].aout_bit-1)-1)/alpha_in)
        # out = self.header(out)
        # out = self.avgpool(out)
        # out = self.act_qlast(out)
        # out = self.fc(out.view(out.size(0),-1))
#*################ repalce the first block 91.36%

        # out = self.cells[0].INTforward(out,alpha_in,bit=32,intbit=16,BNbit=32,BNintbit=16)
        # alpha_in = self.cells[0].act_quant_out.scale_coef.item()  # 91.36%
        # out = out*1.0/((2**(self.cells[0].aout_bit-1)-1)/alpha_in)

        # for i in range(len(self.cells)-1):
        #     out = self.cells[i+1](out)
        # out = self.header(out)
        # out = self.avgpool(out)
        # out = self.act_qlast(out)
        # out = self.fc(out.view(out.size(0),-1))

#!##Replace the block with downsample Cell Failed####### 75%

        # ########
        # out = out*1.0/ ( (2**(self.stem.aout_bit-1)-1) / alpha_in)
        # out = self.cells[0](out)

        # alpha_in = self.cells[0].act_quant_out.scale_coef.item()
        # out = out*((2**(self.cells[0].aout_bit-1)-1)/alpha_in)
        # out = self.cells[1].INTforward(out,alpha_in,bit=32,intbit=8,BNbit=32,BNintbit=8)
        # alpha_in = self.cells[1].act_quant_out.scale_coef.item()
        # out = out*1.0/((2**(self.cells[1].aout_bit-1)-1)/alpha_in)

        # for i in range(len(self.cells)-2):
        #     out = self.cells[i+2](out)
        # out = self.header(out)
        # out = self.avgpool(out)
        # out = self.act_qlast(out)
        # out = self.fc(out.view(out.size(0),-1))
        # #sys.exit(0)
#*##Replace the 3-rd block ####### Right # 91.39%
        # out = out*1.0/ ( (2**(self.stem.aout_bit-1)-1) / alpha_in)
        # out = self.cells[0](out)
        # out = self.cells[1](out)

        # alpha_in = self.cells[1].act_quant_out.scale_coef.item()
        # out = out*((2**(self.cells[1].aout_bit-1)-1)/alpha_in)
        # out = self.cells[2].INTforward(out,alpha_in,bit=32,intbit=18,BNbit=32,BNintbit=18)
        # alpha_in = self.cells[2].act_quant_out.scale_coef.item()
        # out = out*1.0/((2**(self.cells[2].aout_bit-1)-1)/alpha_in)

        # for i in range(len(self.cells)-3):
        #     out = self.cells[i+3](out)
        # out = self.header(out)
        # out = self.avgpool(out)
        # out = self.act_qlast(out)
        # out = self.fc(out.view(out.size(0),-1))
        return out


if __name__=="__main__":
    from easydict import EasyDict as edict
    Config = edict()
    #expasion = 4
    Config.num_classes = 10
    Config.strides=[1,2,1,2]
    Config.num_layer_list = [1, 3, 3, 3]
    Config.num_channel_list = [16, 32,32, 64]
    Config.stem_channel = 16
    Config.header_channel = 512
    Config.layer_abit = [1, 2, 3,4,5, 6,7,8, 9,10,11, 12,13]
    Config.layer_wbit = [1, 2, 3,4,5, 6,7,8, 9,10,11, 12,13]
    model = ToyNet_WINT(Config)
    print(model)