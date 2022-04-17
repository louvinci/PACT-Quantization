# DoReFA\PACT-QAT
Quantization (QAT) Demo on CIFAR10 
混合位宽量化、Quantization-aware-training、MobileNetv2、ResNet20、自定制的ConvNextNet


----

``config_train.py``: 修改网络架构、量化位宽以及训练策略  
``quantize_fn.py``: 权重、激活量化策略。这里参照的是[DorefaNet](https://arxiv.org/abs/1606.06160),不过稍作修改  
``QConvnextblock.py``: 基础的block  
``ToyNet.py\MobileNetv2.py``: 定义了待量化的模型 ,stem + multiple blocks + hearder + fc宏架构   
``train.py``:  训练文件  

## 量化选择
:gift_heart:(部分conv层激活量化放置在"+="之后，BN层通过INT32定点数近似)  
:black_heart: 借鉴PACT对激活缩放后截断再扩放，scale设置为定值

- [x] 每个block内部的激活和权重位宽相同
- [x] 首尾两层敏感度很高(尤其是激活)
- [x] 平均池化与BN层的量化16bit时对精度影响不大

## 实验记录
ToyNet  
batch=128, lr=0.01, 'cos'学习率调整, epoch=300 (params:0.203626M, MADDS :25.601536M)  **模型参数、计算量较小**量化影响比较大  
|ToyNet-CIFAR10 |full Precision| cfg-1 w\o larger Batchsize| cfg-2|
|:--:| :--:|:--:|:--:|
|ACC(%) |91.594 |89.814\89.482|89.458 |

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
C.layer_wbit = [32,8, 8,8,8, 8,8,8, 6,6,6, 16,16]
```
------

MobileNetv2 in Cifar10   
训练参数不变，MEM：2.383050M, MADDS = 98.645504M   
``cfg-1*``:stem+head+fc的位宽相同，中间层均INT8; ``cfg-2*``类似调整了部分后端的block位宽   
|MBv2-CIFAR10 |full Precision| cfg-1* w\o pact |cfg*-1 pact+wo branch_out quant| cfg-2*|
|:--:| :--:|:--:|:--:|:--:|
|ACC(%) |94.165 |88.983\81.665|91.094|xx|

-------

ResNet20 in Cifar10   
训练参数变为batch:256, lr:0.1. MEM:0.272474M, MADDS = 41.214656M    
**这里改变训练参数后```90.477->92.021```**
|ResNet20 |full Precision| cfg-1* w\o branch_out quant | cfg-2*|
|:--:| :--:|:--:|:--:|
|ACC(%) |92.50 |92.021\92.344|xx|
## TODO
- [ ] 整型推理
- [ ] 权重提取
- [ ] 硬件仿真
