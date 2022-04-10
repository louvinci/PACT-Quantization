# DorefaNet-QAT
Quantization (QAT) Demo on CIFAR10 
自定义小网络模型、混合位宽量化、Quantization-aware-training


----

``config_train.py``: 修改网络架构、量化位宽以及训练策略  
``quantize_fn.py``: 权重、激活量化策略。这里参照的是[DorefaNet](https://arxiv.org/abs/1606.06160),不过稍作修改  
``QConvnextblock.py``: 基础的block  
``ToyNet.py``: 定义了待量化的模型 ,stem + multiple blocks + hearder + fc宏架构 
``train.py``:  训练文件  

## 量化选择
:gift_heart:(考虑硬件实现；激活量化放在"+="之后，同时考虑平均池化以及BN层)

- [x] 首尾两层敏感度很高(尤其是激活)
- [x] 平均池化与BN层的量化16bit时对精度影响不大

## 实验记录
batch=128, lr=0.01, 'cos'学习率调整, epoch=300 (params:0.203626M, FLOPs :25.601536M)  
**模型参数、计算量较小**量化影响比较大  
| |full Precision| cfg-1  | cfg-2 w\o-bn| cfg-3 w\o-bn|
|-| -|-|-|-|
|ACC |91.594 |91.238|90.457\90.457|82.971\82.288|

bit=32意味着不量化,avgpooling的输出量化策略与fc 的a_bit相同  stem(1)+blocks(3,3,3)+hearder(1)+fc(1)

cfg-1:
```python
'layer_abit': [32,8, 8,8,8,  8,8,8, 8,8,8,  16,32]
'layer_wbit': [32,8, 6,6,6,  6,6,6, 8,8,8,  16,32]
```
cfg-2:  
```python
C.layer_abit = [32,8, 8,8,8, 8,8,8, 8,8,8, 16,32]
C.layer_wbit = [16,8, 6,6,6, 6,6,6, 8,8,8, 16,16]
```
cfg-3:  
```python
C.layer_abit = [16,8, 8,8,8, 8,8,8, 8,8,8, 16,32]
C.layer_wbit = [16,8, 6,6,6, 6,6,6, 8,8,8, 16,16]
```

## TODO
- [ ] 权重提取
- [ ] 硬件仿真
