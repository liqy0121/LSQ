# LSQ and LSQ+<br>
LSQ+ net or LSQplus net and LSQ net <br>

## commit log<br>
add torch.nn.Parameter .data, retrain models 18-01-2022<br>

I'm not the author, I just complish an unofficial implementation of LSQ+ or LSQplus and LSQ，the origin paper you can find LSQ+ here [arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576) and LSQ here [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153).<br>

pytorch==1.8.1<br>

You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model<br>

Dataset used for training is CIFAR10 and model used is Resnet18 revised<br>

## Version introduction
lsqplus_quantize_V1.py: initialize s、beta of activation quantization according to LSQ+[arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576)<br><br>
lsqplus_quantize_V2.py: initialize s、beta of activation quantization according to min max values<br><br>
lsqquantize_V1.py：initialize s of activation quantization according to LSQ [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153)<br><br>
lsqquantize_V2.py: initialize s of activation quantization = 1<br><br>
lsqplus_quantize_V2.py has the best result when use cifar10 dataset<br>

## The Train Results 
### For the below table all set a_bit=8, w_bit=8
| version | weight per_channel | learning rate | A s initial | A beta initial | best epoch | Accuracy | models
| ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Float 32bit | - | <=66 0.1<br><=86 0.01<br><=99 0.001<br><=112 0.0001 | - | - | 112 | 92.6 | [download](https://share.weiyun.com/g7P6cL23) |
| lsqplus_quantize_V1 | × | <=31 0.1<br><=61 0.01<br><=81 0.001<br><112 0.0001 | 1 | -1e-9 | 90 | 90.3 | [download](https://share.weiyun.com/Cny7NNZn) |
| lsqplus_quantize_V2 | × | as before | - | - | 87 | 92.8 | [download](https://share.weiyun.com/B228P2ha) |
| lsqplus_quantize_V1 | ✔ | as before | - | - | 96 | 91.19  | [download](https://share.weiyun.com/Amgi2b6Q) |
| lsqplus_quantize_V2 | ✔ | as before | - | - | 69 | 92.8 | [download](https://share.weiyun.com/XHy57hmw) |
| lsqquantize_V1 | × | as before | - | - | 102 | 91.89 | [download](https://share.weiyun.com/Rpsevh5f) |
| lsqquantize_V2 | × | as before | - | - | 69 | 91.82 | [download](https://share.weiyun.com/xOQLjvTK) |
| lsqquantize_V1 | ✔ | as before | - | - | 108 | 91.29 | [download](https://share.weiyun.com/xkL9JBir) |
| lsqquantize_V2 | ✔ | as before | - | - | 72 | 91.72 | [download](https://share.weiyun.com/eQZbF3z3) |
<br>
A represent activation, I use moving average method to initialize s and beta.<br><br>

LEARNED STEP SIZE QUANTIZATION<br>
LSQ+: Improving low-bit quantization through learnable offsets and better initialization<br>

### References<br>
https://github.com/666DZY666/micronet<br>
https://github.com/hustzxd/LSQuantization<br>
https://github.com/zhutmost/lsq-net<br>
https://github.com/Zhen-Dong/HAWQ<br>
https://github.com/KwangHoonAn/PACT<br>
https://github.com/Jermmy/pytorch-quantization-demo<br>
# nn-compression
A Pytorch implementation of Neural Network Compression (pruning, quantization, encoding/decoding)

Most work of this repo is better done in [distiller](https://github.com/NervanaSystems/distiller). However, they have not implement channel pruning and coding yet. With coding in this repo, you can save the model with actually much smaller memory size.

## Pruning

Neural Network Pruning reduces the number of nonzero parameters and thus computation amount (FLOPs).

### Vanilla Pruning

Deep Compression uses vanilla pruning method. It prunes the parameters with the least importance.

* **_Elementwise_** Pruning: prune those with the smallest magnitude

* **_Kernelwise_** Pruning: prune 2D kernels with the smallest L1(default)/L2 norm

* **_Filterwise_** Pruning: prune 3D filters with the smallest L1(default)/L2 norm

```python
# vanilla pruner usage

from modules.prune import VanillaPruner

rule = [
        ('0.weight', 'element', [0.3, 0.5], 'abs'),
        ('1.weight', 'kernel', [0.4, 0.6], 'default')
        ('2.weight', 'filter', [0.5, 0.7], 'l2norm')
    ]

pruner = VanillaPruner(rule=rule)
"""
:param rule: str, path to the rule file, each line formats
                  'param_name granularity sparsity_stage_0, sparstiy_stage_1, ...'
             list of tuple, [(param_name(str), granularity(str),
                              sparsity(float) or [sparsity_stage_0(float), sparstiy_stage_1,],
                              fn_importance(optional, str or function))]
             'granularity': str, choose from ['element', 'kernel', 'filter']
             'fn_importance': str, choose from ['abs', 'l1norm', 'l2norm', 'default']
"""

stage = 0

for epoch in range(0, 90):
    if epoch == 0:
        pruner.prune(model=model, stage=stage, update_masks=True)
        best_prec1 = validate(val_loader, model, criterion, epoch)
    
    # in train function
    for i, (input, target) in enumerate(train_loader):
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pruner.prune(model=model, stage=stage, update_masks=False)
```

### Channel Pruning

Channel Pruning is another set of neural network pruning methods. It reduces the number of output channels 
in every convolution or fully-connected layers. Therefore, it can directly speed up the inference.

Channel Pruning takes 2 steps:

1. Channel Selection: select channels with least impact to prune
2. Parameter Reconstruction: reconstruct the parameter values to optimize the output feature of the next 
layer to the pruned one

These two steps are conducted layer by layer.

```python
# channel pruning usage

def prune_channel(sparsity, module, next_module, fn_next_input_feature, input_feature,
                  method='greedy', cpu=True):
    """
    channel pruning core function
    :param sparsity: float, pruning sparsity
    :param module: torch.nn.module, module of the layer being pruned
    :param next_module: torch.nn.module, module of the next layer to the one being pruned
    :param fn_next_input_feature: function, function to calculate the input feature map for next_module
    :param input_feature: torch.(cuda.)Tensor, input feature map of the layer being pruned
    :param method: str
        'greedy': select one contributed to the smallest next feature after another
        'lasso': pruned channels by lasso regression
        'random': randomly select
    :param cpu: bool, whether done in cpu for larger reconstruction batch size
    :return:
        void
    """
```

Detailed example shows in [here](examples/channel_pruning).

## Quantization

Neural Network Quantization is to represent the parameters with fewer bits.

### Vanilla Quantization

There are several ways to quantize neural network parameters:

* **_Fixed-point_** Quantization: the most common way, uses (*i*+*f*)-bits to represent the number, 
where *i*-bits for integer and *f*-bits for fraction.

* **_Uniform/Linear_** Quantization: quantization centroids lies uniformly in the range of parameter values, 
i.e., the quantization step equals $(max - min) / k$, where *k* is the quantization levels

* **_K-Means_** Quantization: quantization centroids calculated by K-Means clustering

```python
# vanilla quantizer usage

from modules.quantize import Quantizer

rule = [
        ('0.weight', 'k-means', 4, 'k-means++'),
        ('1.weight', 'fixed_point', 6, 1),
    ]

quantizer = Quantizer(rule=rule, fix_zeros=True)
"""
:param rule: str, path to the rule file, each line formats
                'param_name method bit_length initial_guess_or_bit_length_of_integer'
             list of tuple,
                [(param_name(str), method(str), bit_length(int),
                  initial_guess(str)_or_bit_length_of_integer(int))]
:param fix_zeros: whether to fix zeros when quantizing
"""

for epoch in range(0, 90):
    # in the train loop
    
    # in train function
    for i, (input, target) in enumerate(train_loader):
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        quantizer.quantize(model=model, update_labels=True, re_quantize=False)
        """
        :param update_labels: bool, whether to re-allocate the param elements
                                    to the latest centroids when using k-means
        :param re_quantize: bool, whether to re-quantize the param when using k-means
        """
```

## Coding

Coding is the last step to compress the neural network in Deep Compression:

* **_Fixed-point_** Coding: it actually is not a coding method, 
just in case if we want to actually save the model in fixed-point style.

* **_Vanilla (Linear)_** Coding: it uses $log_2 (N)$-bits to represent *N* float number in the codebook, 
i.e., there are only *N* possible values in a parameter matrix

* **_Huffman_** Coding: it uses huffman coding to represent *N* float number in the codebook

```python
# coding codec usage (encode)

import torch
from modules.coding import Codec

rule = [
        ('0.weight', 'huffman', 0, 0, 4),
        ('1.weight', 'fixed_point', 6, 1, 4)
    ]

codec = Codec(rule=rule)
"""
:param rule: str, path to the rule file, each line formats
                'param_name coding_method bit_length_fixed_point bit_length_fixed_point_of_integer_part
                 bit_length_of_zero_run_length'
             list of tuple,
                [(param_name(str), coding_method(str), bit_length_fixed_point(int),
                 bit_length_fixed_point_of_integer_part(int), bit_length_of_zero_run_length(int))]
"""

encoded_model = codec.encode(model=model)

torch.save({'state_dict': encoded_model.state_dict()}, 'encode.pth.tar', pickle_protocol=4)
```

```python
# coding codec usage (decode)

import torch
from modules.coding import Codec

checkpoint = torch.load('encode.pth.tar')

model = Codec.decode(model=model, state_dict=checkpoint['state_dict'])  # initial model is created before

torch.save({'state_dict': model.state_dict()}, 'decode.pth.tar')
```

## Rerference

```text
@article{han2015deep,
  title={Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding},
  author={Han, Song and Mao, Huizi and Dally, William J},
  journal={arXiv preprint arXiv:1510.00149},
  year={2015}
}
```

```text
@inproceedings{han2015learning,
  title={Learning both weights and connections for efficient neural network},
  author={Han, Song and Pool, Jeff and Tran, John and Dally, William},
  booktitle={Advances in neural information processing systems},
  pages={1135--1143},
  year={2015}
}
```

```text
@article{luo2017thinet,
  title={Thinet: A filter level pruning method for deep neural network compression},
  author={Luo, Jian-Hao and Wu, Jianxin and Lin, Weiyao},
  journal={arXiv preprint arXiv:1707.06342},
  year={2017}
}
```

```text
@inproceedings{he2017channel,
  title={Channel pruning for accelerating very deep neural networks},
  author={He, Yihui and Zhang, Xiangyu and Sun, Jian},
  booktitle={International Conference on Computer Vision (ICCV)},
  volume={2},
  number={6},
  year={2017}
}
```
