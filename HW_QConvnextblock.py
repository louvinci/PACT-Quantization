from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quant_fn import weight_int,get_scale,fixed_type,bn_actquant_int
from quant_fn import Linear_Q,Conv2d_Q,activation_quant,act_pactq,ActQuant
Conv2d=Conv2d_Q
DEBUG = False
if DEBUG:
    logfile = open('rescale_inf.txt','a+')
BatchNorm2d = nn.BatchNorm2d
torch.set_printoptions(precision=7)

class HW_ConvnextBlock(nn.Module):
    '''
    Original ConvNext: DW7x7 (96-96)+ LayerNorm 
                       -> Conv1x1(96-384) + GELU -> Conv1x1(384-96)
    Modified ConvNext: DW7x7 (cin-cout)+ BatchNorm 
                       -> Conv1x1(cin-cin*e) + RELU 
                       -> Conv1x1(cin*e-cout)
    '''
    def __init__(self,ain_bit,aout_bit,w_bit,C_in,C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(HW_ConvnextBlock,self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.ain_bit = ain_bit
        self.aout_bit= aout_bit
        self.w_bit = w_bit

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        
        # when stride =2, c_in must be different form c_out, the opposite is not true,but one norm layer seems no matter
        if stride == 2 or C_in != C_out:
            self.norm = BatchNorm2d(C_in)
            if w_bit ==32:
                self.act_quant_bn = act_pactq(a_bit=32,fixed_rescale=2)# don't quantize
                #self.act_quant_bn = activation_quant(a_bit=32)
            else:
                #self.act_quant_bn = activation_quant(a_bit=32)
                self.act_quant_bn = act_pactq(a_bit=32,fixed_rescale=2)#* Act_quant layer after the BN layer
            self.downsample = Conv2d(w_bit,C_in,C_out,kernel_size=stride,stride=stride,padding=0,dilation=1,groups=1,bias=False)
            self.actq_down = ActQuant(a_bit=ain_bit)
        
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
        self.act_quant_in1 = ActQuant(a_bit=ain_bit)

        self.conv2 = Conv2d(w_bit,C_out,C_out*expansion,kernel_size=1,stride=1,padding=0,dilation=1,groups=self.groups,bias=bias)
        self.relu=nn.ReLU(inplace=True)
        self.act_quant_in2 = ActQuant(a_bit=ain_bit)
        
        self.conv3 = Conv2d(w_bit,C_out*expansion,C_out,kernel_size=1,stride=1,padding=0,dilation=1,groups=self.groups,bias=bias)
        self.act_quant_out = ActQuant(a_bit=aout_bit)
        
    def forward(self,x):
        #default the x is quantized
        if self.stride == 2 or self.C_in != self.C_out:
            x = self.norm(x)
            #x = self.act_quant_bn(x) # must quantize before the conv engine
            x = self.downsample(x)
            # print("SW Conv ")
            # print(x)
            x = self.actq_down(x)
        # print("SW downsample")
        # print(x)

        identity = x
        x = self.bn1(self.conv1(x)) 
        x = self.act_quant_in1(x)
        
        x = self.relu(self.conv2(x)) # this ReLU layer cannot be overlap 
        x = self.act_quant_in2(x)
        x = self.conv3(x)
        x += identity
        x = self.act_quant_out(x)

        return x
    def INTforward(self,x,alpha_in,bit=24,intbit=10,BNbit=16,BNintbit=6):

        if self.stride == 2 or self.C_in != self.C_out:
            gamma0,beta0,mean0,var0,eps = self.norm.weight,self.norm.bias,self.norm.running_mean,self.norm.running_var,self.norm.eps
            bn_wt0,bn_bias0= gamma0.div(torch.sqrt(var0 + eps)),beta0 - gamma0.mul(mean0).div(torch.sqrt(var0 + eps))
            in_scale = (2**(self.ain_bit-1)-1)/alpha_in
            q_bn_wt0    = fixed_type(bn_wt0,           BNbit,BNintbit)
            q_bn_bias0  = fixed_type(bn_bias0*in_scale,BNbit,BNintbit)
            if DEBUG:
                print("id_bnwt is \n", bn_wt0.cpu().numpy(),  "\nfixed scale:\n", (q_bn_wt0  /(2**(BNbit-BNintbit))).cpu().numpy() ,file=logfile)
                print("id_bias is \n", bn_bias0.cpu().numpy(),"\nfixed scale:\n", (q_bn_bias0/(2**(BNbit-BNintbit))).cpu().numpy() ,file=logfile)
            q_bn_wt0    = q_bn_wt0.reshape(1,q_bn_wt0.size(0),1,1)
            q_bn_bias0  = q_bn_bias0.reshape(1,q_bn_bias0.size(0),1,1)
            x = x*q_bn_wt0 + q_bn_bias0
            x =torch.div( x, 2**(BNbit-BNintbit),rounding_mode='trunc')
            #x =  x//2**(BNbit-BNintbit)
            x = torch.clamp(x,-2**(BNbit-1),2**(BNbit-1)-1)
            #x = torch.clamp(x,-2**(self.ain_bit-1),2**(self.ain_bit-1)-1)
            
            q_w0 = weight_int(self.downsample.weight.data,self.w_bit)
            q_w0,x = q_w0.float(),x.float()
            x = F.conv2d(x,q_w0,bias=None,stride=self.stride,padding=0,dilation=1,groups=1) 
            x=x.long()
            # print("HW conv ")
            # print(x*1.0/ ( (2**(self.w_bit-1)-1) * (2**(self.ain_bit-1)-1)/alpha_in ) )
            alpha_first = self.actq_down.scale_coef.item()
            scale_out0,re_scale0,range_out0=get_scale(self.ain_bit,alpha_in,self.w_bit,self.ain_bit,alpha_first)
            q_rescale0 = fixed_type(re_scale0,bit,intbit)
            if DEBUG:
                print("rescale0 is {:.5f}; fixed scale: {:.5f}".format(re_scale0,q_rescale0/(2**(bit-intbit))),file=logfile)
            #x = (q_rescale0*x)*1.0 / (2**(bit-intbit)) # equals shift bit
            x =torch.div( q_rescale0*x, 2**(bit-intbit),rounding_mode='trunc')
            x = torch.clamp(x,-range_out0-1,range_out0)
        else:
            alpha_first = alpha_in
        
        alpha_out1 = self.act_quant_in1.scale_coef.item()
        alpha_out2 = self.act_quant_in2.scale_coef.item()
        alpha_out3 = self.act_quant_out.scale_coef.item()
        # in block,conv layers share same bit config
        scale_out1,re_scale1,range_out1=get_scale(self.ain_bit,alpha_first,self.w_bit,self.ain_bit,alpha_out1) 
        scale_out2,re_scale2,range_out2=get_scale(self.ain_bit,alpha_out1,self.w_bit,self.ain_bit,alpha_out2)
        scale_out3,re_scale3,range_out3=get_scale(self.ain_bit,alpha_out2,self.w_bit,self.aout_bit,alpha_out3)
        #the first conv rescale is absorbed by the bn layer
        q_rescale2 = fixed_type(re_scale2,bit,intbit)
        q_rescale3 = fixed_type(re_scale3,bit,intbit)
        q_w1 = weight_int(self.conv1.weight.data,self.w_bit)
        q_w2 = weight_int(self.conv2.weight.data,self.w_bit)
        q_w3 = weight_int(self.conv3.weight.data,self.w_bit)

        identity = x
        id_rescale =  scale_out3 / ((2**(self.ain_bit-1)-1)/alpha_first)
        p_idrescale = fixed_type(id_rescale,bit,intbit)

        #################   CONV1 ################
        #print("debug1",x.dtype)
        q_w1, x  = q_w1.float(),x.float() #!cuda type not support change long to float
        q_x = F.conv2d(x,q_w1,bias=None,stride=1,padding=self.padding,dilation=1,groups=self.C_out)
        q_x = q_x.long()
        gamma,beta,mean,var,eps = self.bn1.weight,self.bn1.bias,self.bn1.running_mean,self.bn1.running_var,self.bn1.eps
        bn_wt,bn_bias = bn_actquant_int(gamma,beta,mean,var,eps,re_scale1,scale_out1,bit,intbit)
        q_x =  q_x * bn_wt+bn_bias
        #q_x =  q_x*1.0 / (2**(bit-intbit))
        q_x =torch.div( q_x, 2**(bit-intbit),rounding_mode='trunc')
        q_x = torch.clamp(q_x,-range_out1-1,range_out1)

        #################   CONV2 ################
        q_w2, q_x  = q_w2.float(),q_x.float()
        q_x = F.conv2d(q_x,q_w2,bias=None,stride=1,padding=0,dilation=1,groups=self.groups)
        q_x = q_x.long()
        #q_x = (q_rescale2*q_x)*1.0 / (2**(bit-intbit))#rescale and shift bit
        q_x =torch.div( q_rescale2*q_x, 2**(bit-intbit),rounding_mode='trunc')
        q_x = torch.clamp(q_x,0,range_out2) # containing relu

        #################   CONV3 ################
        q_w3, q_x  = q_w3.float(),q_x.float()
        q_x = F.conv2d(q_x,q_w3,bias=None,stride=1,padding=0,dilation=1,groups=self.groups)
        q_x = q_x.long()
        #q_x = (q_rescale3*q_x + p_idrescale*identity)*1.0 /(2**(bit-intbit))
        q_x =torch.div( q_rescale3*q_x + p_idrescale*identity, 2**(bit-intbit),rounding_mode='trunc')
        q_x = torch.clamp(q_x,(-range_out3-1),range_out3)

        if DEBUG:
            print("rescale_id is {:.5f}; fixed scale: {:.5f}".format( id_rescale,p_idrescale/2**(bit-intbit) ),file=logfile)
            print("rescale2   is {:.5f}; fixed scale: {:.5f}".format( re_scale2,q_rescale2 /2**(bit-intbit) ),file=logfile)
            print("rescale3   is {:.5f}; fixed scale: {:.5f}".format( re_scale3,q_rescale3 /2**(bit-intbit) ),file=logfile)

        return q_x

'''
offline prepare: quantized weight, quatized bn_wt*:(re_scale * bn_wt) bn_bias*:(scale_out*bn_bias)
                 re_scale:(scale_out / (scale_a * scale_w))
'''
class HW_ConvNorm(nn.Module):
    def __init__(self, ain_bit,aout_bit,w_bit,C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,):
        super(HW_ConvNorm,self).__init__()

        assert stride in [1, 2]
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.ain_bit = ain_bit
        self.aout_bit= aout_bit
        self.w_bit = w_bit
        self.bias = bias
        self.stride = stride
        self.groups = groups

        self.conv = Conv2d(w_bit,C_in, C_out, kernel_size=kernel_size, stride=stride, padding=self.padding, 
                            dilation=self.dilation, groups=groups, bias=bias)
        self.bn = BatchNorm2d(C_out)
        #self.act_quant_out=activation_quant(a_bit=aout_bit)
        #self.act_quant_out=act_pactq(a_bit=aout_bit,fixed_rescale=10)
        self.act_quant_out=ActQuant(a_bit=aout_bit,scale_coef=6.0)
    
    def forward(self,x):
        q_x = self.conv(x)
        x = self.bn(q_x)
        q_x = self.act_quant_out(x)
        return q_x
        
    def INTforward(self,x,alpha_in,bit=32,intbit=10):
        #! need to know the activation alpha factor of the previous layer.
        #! "get_scale", "weight_int","bn_actquant_int" can be done offline
        alpha_out = self.act_quant_out.scale_coef.item()
        scale_out,re_scale,range_out=get_scale(self.ain_bit,alpha_in,self.w_bit,self.aout_bit,alpha_out)

        q_w = weight_int(self.conv.weight.data,self.w_bit)
        
        x,q_w = x.float(),q_w.float() # if first layer input keep float, here q_w should be float type
        #* 1. online compute
        q_x = F.conv2d(x,q_w,bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        q_x = q_x.long()

        # BN layer with rescale absorb, return Fixed INT type, and the frac bits
        gamma,beta,mean,var,eps = self.bn.weight,self.bn.bias,self.bn.running_mean,self.bn.running_var,self.bn.eps
        bn_wt,bn_bias = bn_actquant_int(gamma,beta,mean,var,eps,re_scale,scale_out,bit,intbit)

        #* 2. online compute
        q_x =  q_x * bn_wt+bn_bias
        #q_x = q_x >>(bit-intbit)#!do not use the bit shift in pytorch,in hardware directly shift bit
        #q_x =  q_x // (2**(bit-intbit)) # this method will be deprecated
        #q_x = q_x / 2**(bit-intbit)
        q_x=torch.div( q_x, 2**(bit-intbit),rounding_mode='trunc')#'trunc'
        q_x = torch.clamp(q_x,-range_out-1,range_out)

        return q_x 

    def BNforward(self,x):
        q_x = self.conv(x)
        
        bn_bias    =  self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(torch.sqrt(self.bn.running_var + self.bn.eps))
        bn_wt     =   self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        #BN layer
        bn_wt = bn_wt.reshape(1,bn_wt.size(0),1,1)
        bn_bias = bn_bias.reshape(1,bn_bias.size(0),1,1)
        x = q_x * bn_wt+bn_bias
        x = self.act_quant_out(x)
        return x

def Test_HWConvNorm():
    print("======================Test ConvNorm======================")
    ainbit,aoutbit,wbit=8,16,8
    range_in = 2**(ainbit-1)-1
    range_out = 2**(aoutbit-1)-1
    
    alpha_in = 0.8030471205711365
    alpha_out = 0.4455549120903015
    torch.set_grad_enabled(False)
    #scale_a = 2**(a_bit-1) -1 / alpha_a, scale_w = 2**(w_bit-1)-1 torch.IntTensor
    torch.manual_seed(1234)
    with torch.no_grad():
        C_in,C_out =3, 4
        stem = HW_ConvNorm(ainbit,aoutbit,wbit,C_in=C_in,C_out=C_out,kernel_size=3,padding=None)
        stem.conv.weight.data = torch.randn((C_out,C_in,3,3))
        stem.bn.weight.data = torch.rand(C_out)
        stem.bn.bias.data = torch.rand(C_out)
        stem.bn.running_mean= torch.rand(C_out)
        stem.bn.running_var= torch.rand(C_out)

        stem.act_quant_out.scale_coef.data = torch.tensor(alpha_out)
        
        stem.eval()#!necesaary
        print("test 1: quantized data in")
        a = torch.randint(-range_in-1,range_in,(1, C_in, 5, 5)) # already quantized data
        float_a = a*alpha_in/range_in
        stem_a = stem(float_a)
        # stem_b = stem.BNforward(float_a)
        # #print(stem_a,'\n',stem_b)
        # d = (stem_a - stem_b).abs().mean().item()
        # print('error:',d)
        stem_c = stem.INTforward(a,alpha_in,32,16) #24,12
        print("result dtype: ", stem_c.dtype, "; shape: ", stem_c.shape)
        #print(stem_c)
        float_stem_c = stem_c*alpha_out / range_out
        #print(stem_a,'\n',float_stem_c)
        d = (stem_a - float_stem_c).abs().mean().item()
        print('error:',d)

        # print("test 2: float data in")
        # float_a = torch.randn((1, C_in, 5, 5)) # first layer test
        # stem.ain_bit = 32
        # stem_a = stem.forward(float_a)
        # stem_c = stem.INTforward(float_a,1,32,16) #24,12
        # print("result dtype: ", stem_c.dtype, "shape: ", stem_c.shape)
        # #print(stem_c)
        # float_stem_c = stem_c*alpha_out / range_out
        # #print(stem_a,'\n',float_stem_c)
        # d = (stem_a - float_stem_c).abs().mean().item()
        # print('error:',d)

def Test_HWConvblock():
    print("======================Test Convblock======================")
    ainbit,aoutbit,wbit=8,16,8
    range_in = 2**(ainbit-1)-1
    range_out = 2**(aoutbit-1)-1
    #!the alpha vale must be initialized,otherwise data range will be cliped.
    alpha_dn  = 0.5#0.501723051071167
    alpha_in  = 0.7#0.7154887318611145
    alpha1    = 0.4#0.4149089753627777
    alpha2    = 0.24#0.24795827269554138
    alpha_out = 0.5#0.5123898386955261 
    scale_in  = range_in/alpha_in
    torch.set_grad_enabled(False)
    torch.manual_seed(13)
    with torch.no_grad():
        C_in,C_out =3, 4
        stride =1
        #self,ain_bit,aout_bit,w_bit,C_in,C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False

        block = HW_ConvnextBlock(ainbit,aoutbit,wbit,C_in=C_in,C_out=C_out,expansion=4,kernel_size=3,stride=stride)
        block.act_quant_in1.scale_coef.data = torch.tensor(alpha1)
        block.act_quant_in2.scale_coef.data = torch.tensor(alpha2)
        block.act_quant_out.scale_coef.data = torch.tensor(alpha_out)
        if C_in != C_out or stride !=1:
            block.actq_down.scale_coef.data = torch.tensor(alpha_dn)
        block.eval()
        a = torch.randint(-range_in-1,range_in,(1, C_in, 5, 5)) # already quantized data
        float_a = a/scale_in
        res_a = block.forward(float_a)
        #res_c = block.INTforward(a,alpha_in,24,10,BNbit=16,BNintbit=5)
        res_c = block.INTforward(a,alpha_in,32,16,BNbit=32,BNintbit=16)
        print("result type: ", res_c.dtype,";  shape: ", res_c.shape)
        #print(res_c)
        float_res_c = res_c*alpha_out / range_out
        #print(res_a,'\n',float_res_c)
        d = (res_a - float_res_c).abs().mean().item()
        print('error:',d)

if __name__ == "__main__":
    Test_HWConvNorm()
    Test_HWConvblock()

    

    
    
    
    