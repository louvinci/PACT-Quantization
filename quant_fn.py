from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn.parameter import Parameter
DEBUG = False
#logfile = open('bnlog.txt','a+')
# Fake node, quant and dequant opereation
def uniform_quantize(k):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k  - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class ActFn(Function):
    @staticmethod
    def forward(ctx, x,alpha,k):
        assert k <= 16 or k == 32
        if k==32:
            return x
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        #y = torch.clamp(x, min =0, max = alpha.item())
        y = torch.clamp(x, min = -alpha.abs().item(), max = alpha.abs().item())
        scale = (2**k - 1) / alpha
        y_q = torch.round( y * scale) / scale
        return y_q
    '''
    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < 0
        upper_bound      = x > alpha
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1) 
        return dLdy_q * x_range.float(), grad_alpha, None
    '''
    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < -alpha.abs()
        upper_bound      = x > alpha.abs()
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum( dLdy_q * (torch.ge(x, alpha.abs()).float()-torch.le(x,-alpha.abs()).float()) ).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None
        

class ActQuant(nn.Module):
    def __init__(self,a_bit,scale_coef=10.0):
        super(ActQuant,self).__init__()
        self.scale_coef = Parameter(torch.tensor(scale_coef))
        self.bit = a_bit
        self.quant = ActFn.apply

    def forward(self,x):
        if self.bit == 32:
            q_x = x
        else:
            #q_x = self.quant(x,self.scale_coef,self.bit)
            q_x = self.quant(x,self.scale_coef,self.bit-1) #-alpha, alpha
        return q_x
    def __repr__(self):
        return '{}( Abit={},Scale_Coef={} )'.format(self.__class__.__name__, self.bit,self.scale_coef.item())


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 16 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit-1) 

    def forward(self, x):
        # print('===================')
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = (self.uniform_q(x / E) + 1) / 2 * E
        else:
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = self.uniform_q(weight)
            ##standard DorefaNet
            # weight = torch.tanh(x)
            # weight = weight / (2*torch.max(torch.abs(weight))) + 0.5
            # weight_q = 2 * self.uniform_q(weight) - 1 # this operation is not hardward-friendly

        return weight_q 


class activation_quant(nn.Module):
    def __init__(self, a_bit):
        super(activation_quant, self).__init__()
        assert a_bit <= 16 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            #activation_q = self.uniform_q(torch.clamp(x, 0, 1))
            activation_q = self.uniform_q(torch.clamp(x*0.1, 0, 1))
            #activation_q = self.uniform_q(torch.clamp(x/8, 0, 1))
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q

    def __repr__(self):
        return '{}( Abit={} )'.format(self.__class__.__name__, self.a_bit)

class PacTFn(Function):
	@staticmethod
	def forward(ctx, x, alpha, k):
		ctx.save_for_backward(x, alpha)
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		x, alpha, = ctx.saved_tensors
		lower_bound      = x < 0
		upper_bound      = x > alpha
		# x_range       = 1.0-lower_bound-upper_bound
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None

class act_pactq(nn.Module):
    def __init__(self, a_bit,fixed_rescale=10.0):# fixed_rescale is not fixed here
        super(act_pactq, self).__init__()
        assert a_bit <= 16 or a_bit == 32
        self.a_bit = a_bit

        self.scale_coef = Parameter(torch.tensor(fixed_rescale))
        self.quant = PacTFn.apply

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.quant(x,self.scale_coef,self.a_bit) #0, alpha
        return activation_q

    def __repr__(self):
        return '{}( Abit={},Scale_Coef={} )'.format(self.__class__.__name__, self.a_bit,self.scale_coef)

class act_pactFixed(nn.Module):
    def __init__(self, a_bit,fixed_rescale=2.0):# fixed_rescale is fixed here
        super(act_pactq, self).__init__()
        assert a_bit <= 16 or a_bit == 32
        self.a_bit = a_bit
        self.scale_coef = fixed_rescale
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            out = 0.5*( x.abs() - (x-self.scale_coef).abs()+self.scale_coef)
            activation_q = self.uniform_q(out / self.scale_coef ) * self.scale_coef 
        return activation_q

    def __repr__(self):
        return '{}( Abit={},Scale_Coef={} )'.format(self.__class__.__name__, self.a_bit,self.scale_coef)

def weight_int(w,k):
    scale_w = 2**(k-1)-1
    
    w=torch.tanh(w) 
    w = w/ torch.max(torch.abs(w))
    q_w = torch.round(w * scale_w).long()
    q_w = torch.clamp(q_w,-scale_w-1,scale_w)
    return q_w

def get_scale(ain_bit,alpha_in,w_bit,aout_bit,alpha_out):
    if ain_bit==32:
        scale_a = torch.tensor(1.0)
    else:
        scale_a = (2**(ain_bit-1)-1)/alpha_in
    scale_w = 2**(w_bit -1) -1
    range_out = 2**(aout_bit-1) -1
    scale_out = range_out/alpha_out
    if aout_bit ==32:
        scale_out = torch.tensor(1.0)
    re_scale = scale_out / (scale_a * scale_w)
    return scale_out,re_scale,range_out

def fixed_type(input,bit,intege_bit):
    assert intege_bit <= bit,'ERROR bit assign'
    if torch.is_tensor(input) == False:
        input = torch.tensor(input)
    intege =torch.round(input) #rounding
    frac = input - intege
    #print(intege,frac)
    frac_scale = 2**(bit-intege_bit)#to use bit shift here, so the scale is not 2**(bit-intege_bit)-1
    inte_scale = 2**(intege_bit-1)-1
    intege = torch.clamp(intege,-inte_scale-1,inte_scale)# cliping
    frac=torch.round(frac* frac_scale)/frac_scale
    res = intege+frac
    q_res = (res*frac_scale).int()
    return q_res

# bit: the bit to keep the bn_wt, intbit:integer part bit
def bn_actquant_int(gamma,beta,mean,var,eps,re_scale,scale_out,bit,intbit):
    # BN layer with rescale absorb, return Fixed INT type, and the frac bits
    bn_wt     =   gamma.div(torch.sqrt(var + eps))
    bn_bias   =   beta - gamma.mul(mean).div(torch.sqrt(var + eps))
    bn_wt     =   re_scale * bn_wt
    bn_bias   =   scale_out*bn_bias #! note that
    
    fbn_wt  =fixed_type(bn_wt,bit,intbit)
    fbn_bias=fixed_type(bn_bias,bit,intbit)
    if DEBUG:
        print(bn_wt.cpu().numpy(),file=logfile)
        print((fbn_wt/(2**(bit-intbit))).cpu().numpy(),file=logfile)
        print((fbn_bias/(2**(bit-intbit))).cpu().numpy(),file=logfile)
    #reshape
    fbn_wt   = fbn_wt.reshape(1,fbn_wt.size(0),1,1)
    fbn_bias = fbn_bias.reshape(1,fbn_bias.size(0),1,1)
    return fbn_wt,fbn_bias


class Conv2d_Q(nn.Conv2d):
    def __init__(self, w_bit,in_channels, out_channels,kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight)
        # print(np.unique(weight_q.detach().numpy()))
        return F.conv2d(input, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                            #* for print model information

    def __repr__(self):
        return '{}( Wbit={}, {}, {}, kernel={}, padding={}, stride={}, group={} )'.format(self.__class__.__name__, self.w_bit,
            self.in_channels,self.out_channels,self.kernel_size,self.padding,self.stride,self.groups)



class Linear_Q(nn.Linear):
    def __init__(self, a_bit,w_bit,in_features, out_features, bias=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
        weight_q = self.quantize_fn(self.weight)
        # print(np.unique(weight_q.detach().numpy()))
        #if self.bias!=None:
            #bq = self.quantize_fn(self.bias)

        return F.linear(input, weight_q, self.bias)
    
    def INTforward(self,input,alpha_in):
        q_w = weight_int(self.weight.data,self.w_bit)
        #! bias is float type here, and the output of fc layer do not quantize
        scale_out,re_scale,range_out = get_scale(self.a_bit,alpha_in,self.w_bit,32,1.0)
        input,q_w = input.float(),q_w.float()
        q_x = F.linear(input,q_w,None) # output to CPU
        q_x = q_x.long()
        x = (q_x*1.0)*re_scale
        x+=self.bias
        return x

    def __repr__(self):
        return '{}( Wbit={}, {}, {})'.format(self.__class__.__name__, self.w_bit,
            self.in_features,self.out_features)


def Test_Linear_Q():
    torch.manual_seed(12345)
    ainbit,wbit=16,16
    range_in = 2**(ainbit-1)-1
    alpha_in = 0.8030471205711365
    a = torch.randint(-range_in-1,range_in,(4,32))
    linear = Linear_Q(a_bit=ainbit,w_bit=wbit,in_features=32,out_features=10,bias=True)
    float_a= a/(range_in/alpha_in)
    res_a = linear(float_a)
    hw_a = linear.INTforward(a,alpha_in)
    d = (res_a - hw_a).abs().mean().item()
    print("error: ",d)
    pass

if __name__ == '__main__':
    Test_Linear_Q()

    # a = torch.rand(2, 3, 32, 32)
    # conv = Conv2d_Q(w_bit=8,in_channels=3, out_channels=16, kernel_size=3, padding=1)
    # act = activation_quant(a_bit=4)
    # print(conv)

    # b = conv(a)
    # b.retain_grad()
    # c = act(b)
    
    # #avg = torch.nn.AdaptiveAvgPool2d(1)
    # avg = torch.nn.MaxPool2d(32)
    # avg_c = avg(c)
    # print(c.shape,avg_c.shape)
    # avg_c = avg_c.view(c.size(0),-1)

    # linear = Linear_Q(a_bit=16,w_bit=8,in_features=16, out_features=512)
    # out_c = linear(avg_c)
    # print(linear)
    # print(out_c.shape)

    # d = torch.mean(out_c)# grad only backward for the scalar type.
    # d.backward()
    # from thop import profile
    # from thop.vision.basic_hooks import count_convNd,count_linear
    # custom_ops = {Conv2d_Q: count_convNd,Linear_Q:count_linear}
    # class TestNet(nn.Module):
    #     def __init__(self):
    #         super(TestNet,self).__init__()
    #         self.conv1=Conv2d_Q(w_bit=8,in_channels=3, out_channels=16, kernel_size=3, padding=1)
    #         self.avg1 =torch.nn.MaxPool2d(32)
    #         self.fc1=Linear_Q(a_bit=16,w_bit=8,in_features=16, out_features=512)
    #     def forward(self,x):
    #         x= self.conv1(x)
    #         x=self.avg1(x)
    #         x=self.fc1(x.view(x.size(0),-1))
    #         return x
    # # Test the convd_q and Linear_q
    # model = TestNet()
    # #manuually results : mac= 32*32*3*16*9 + 16*512=450560 mem=3*16*9 + 16*512+512=9136
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    # print(flops,params)#maxpooling:450560,9136;466969,9136
    # #!avageage pooling contaions madds.
