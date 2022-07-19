# Hardware-friendly DoReFa\PACT-QAT 
Quantization (QAT) Demo on CIFAR10   
混合位宽量化、Quantization-aware-training on MobileNetv2、ResNet20、自定制的ConvNextNet    
这里之所以是硬件友好，是因为选择了粗粒度的量化以及对称量化，硬件涉及的额外开销很少。  


----

``config_train.py``: 选择模型、网络架构配置、量化位宽以及训练策略  
``quantize_fn.py``: 权重、激活量化策略。这里参照的是[DoReFa-Net](https://arxiv.org/abs/1606.06160)以及[PACT](https://arxiv.org/abs/1805.06085), 并做出了改进  
``QConvnextblock.py``，```HW_QConvnextblock```: 基础的block，包含整型推理，兼容前者  
``ToyNet.py\MobileNetv2.py``: 定义了待量化的模型 ,stem + multiple blocks + heard + fc宏架构, ToyNet中启用PACT训练  
``ResNet_CF``: 定义了ResNet20在Cifar10上的量化模型，DoReFa量化     
``train.py``:  训练文件  
## 总结
以往软件量化仅仅在conv、fc的输入、输出量化，这里block的输入、输出都经过了伪量化(该操作对精度会有影响，尤其是MBConv)   
ResNet类网络直接使用DoReFa量化对精度影响不大。但MBConv类则效果不行INT8 QAT下降都明显，关键在于DoReFa对激活的截取。  
训练超参数对最终精度影响也很大，从头训练可能不如如小batch微调100 epoch

## 量化选择
:gift_heart:  block内最后一个conv层激活量化放置在"+="之后，分支如有conv操作则对输出进行伪量化  
:black_heart: BN层不folding(qat时对精度影响较大，容易不收敛)，BN层的weight、bias通过INT32定点数近似  
:black_heart: 借鉴PACT对激活缩放后截断再扩放，scale设置为定值对MBConv有效

- [x] 每个block内部的激活和权重位宽相同
- [x] 首尾两层(输入层、FC层)敏感度很高(尤其是激活)
- [x] 平均池化与BN层的量化16bit时对精度影响不大

## 实验记录
😠ToyNet  
batch=128, lr=0.01, 'cos'学习率调整, epoch=300 (params:0.203626M, MADDS :25.601536M)   
```cfg-1*```:输入不量化，fc激活16bit(F32,L16)，其余均INT8  
|ToyNet|full Precision| cfg-1 w\o larger Batchsize|cfg-2 w\o modified pact|cfg-1* + mPACT |cfg-3 + mpact|
|:--:| :--:|:--:|:--:|:--:|:--:|
|ACC(%) |91.594 |89.814\89.482|91.317\89.458 |**91.416**|90.813|

```bit=32```意味着不量化,avgpooling的输出量化策略与``fc``的``a_bit``相同(默认量化)  
参数：stem(1)+blocks(1,3,3,3)+hearder(1)+fc(1)  

cfg-1:  
```python
C.layer_abit = [32,  8, 8,8,8, 8,8,8, 8,8,8,  32,32]
C.layer_wbit = [32,  8, 8,8,8, 8,8,8, 8,8,8,  16,16]
```
cfg-2:  
```python
C.layer_abit = [32,8, 8,8,8, 8,8,8, 8,8,8, 32,32]
C.layer_wbit = [32,8, 6,6,6, 6,6,6, 8,8,8, 16,16]
```
cfg-3:  
```python
C.layer_abit = [32,8, 6,6,6, 6,6,6, 6,6,6, 32,32]
C.layer_wbit = [32,8, 6,6,6, 6,6,6, 6,6,6, 16,16]
```
----
😠MobileNetv2  
训练参数不变，MEM：2.383050M, MADDS = 98.645504M   
``cfg-1*``:stem+head+fc的位宽相同，中间层均INT8; ``cfg-2*``类似调整了部分后端的block位宽   
|MBv2 |full Precision| cfg-1* w\o pact |cfg*-1 + mPACT| cfg-2*|
|:--:| :--:|:--:|:--:|:--:|
|ACC(%) |94.165 |88.983\81.665|93.084|xx|


----
:rocket:ResNet20   
QAT参数变为batch:256, lr:0.1. MEM:0.272474M, MADDS = 41.214656M    
这里仅仅DoReFa，效果就不错了不过。**训练参数需调整```90.477->92.021```**
|ResNet20 |full Precision| cfg-1* w\o branch_out quant | cfg-2*|
|:--:| :--:|:--:|:--:|
|ACC(%) |92.50 |92.021\92.344|xx|
## 整型推理
仅以ToyNet为例在每个block内部实现了```INT_forward```函数，以进行整型推理（由于数据类型限制，实际仍以```Torch.LongTensor```存储，加载进Conv2d或者BN之前转化为```float```）   
🙂:整型结果符合预期```91.41``` -> ```91.05```(这里略微下降是由于定点数小数位数确定)  

小细节：  
Conv+BN中BN参数可以吸收rescale，BN+CONV，则经过BN的输出用16bit or 32bit保存
## TODO
- [ ] 权重提取
- [ ] 硬件仿真
