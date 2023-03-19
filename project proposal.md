### Exploring the efficacy of learned step size quantization

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Yuling Hou, Qiying Li
#### **Motivation and Problem Statement**

The deep neural network can achieve high performance on machine with high precision weights. There is now an increasing demand for methods to migrate the deep neural networks on machine with high precision weights efficiently to resource constrained edge-devices. Quantization is a technique to perform computations and storing tensors at lower bit widths than floating point precision. This allow a more compact model on many mobile or embedded hardware with lower bit width operation. Quantization is primarily a technique to speed up inference and only the forward pass is supported for quantized operators. But it is a challenge to maintain high accuracy for the compact model as bit width of weight decrease.  Learned Step Size Quantization(LSQ) [1] is a method that can train 4-bit models that reach full precision baseline accuracy in ImageNet dataset. It supports various level bit width of weight quantization by only a simple modification of existing training code. In the training, it can estimate and scale the loss gradient at each weight and activation layer’s quantizer step size, such that it can be learned in conjunction with other network parameters.

In the paper[1], the author test the results on Resnet-18, Resnet-34, Resnet-50, Resnet-101,Resnet-152 by using ImageNet dataset. It achieves almost the same accuracy in 4 bits as the accuracy in 32 bits (Float) of baseline. In reality, we usually do quantization after pruning so that we have a higher compression ration on the model and shorten the inference time further. We don't know whether we have similar result on the model after pruning by using LSQ method. Also Resnet-18, Resnet-34, Resnet-50, Resnet-101,Resnet-152 are large-sparse (Resnet etc) model[2], although LSQ can achieves almost the same accuracy as the result of baseline in these large sparse model, the results of LSQ on small-dense （MobileNets, ShuffleNets etc)   counterparts are yet to be seen. While the small-dense models are more common in mobile or embedded devices as they requires less power for the inference and have smaller size models than larger sparse models.

In this project, we will evaluate the performance of LSQ on the model after pruning first. Then we will evaluate the performance of LSQ on the small-dense models.

#### **Challenges and Technique Approaches**

LSQ algorithm is quite complicated. We need to know very clearly on LSQ first. 

Since there is no official implementation of code for LSQ, we need to compare carefully on the unofficial implementations. Also the pruning code base and LSQ code base requires different work environments, we need to modify  the two code base to make them work goether.

As the HPC resource is limited, we will CIFAR10 dataset for our test as ImageNet data is too large to be trained and tested. CIFAR10 contains RGB color pictures of 10 categories: aircraft, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The size of the picture is 32 × 32. There are 50000 training pictures and 10000 test pictures in the data set. 

In the test of applying LSQ on pruned model, we will use train ResNet18 on CIFAR10 to get a baseline of accuracy. Then we will pruning the model and get a reasonable accuracy after fine tune of the model. Then we will apply the LSQ method to see the model performance with bit 3, bit 4, bit 5, bit 6, bit 7, bit 8 weights. Finally we will compare the the result with the result before LSQ. 

In the test of apply LSQ on small dense models, we will choose ShuffleNets V2 model for the experiments since it usually has better performance in edge devices than MobileNets. We will train the network to get a baseline of accuracy of 32 bit float weights, then we will apply LSQ  method to see the model  performance with bit 3, bit 4, bit 5, bit 6, bit 7, bit 8 weights. Finally we will compare the the result with the result before LSQ. In the end, we will summary the LSQ's performance on both pruning model and sparse model and state the further research areas.

#### **Possible Extension beyond Current Approaches mentioned in 2**

If possible, we will see the compression ration and accuracy  of model by pruning, LSQ, cluster, Huffman encodeing. 
If possible, as there are 3 versions of LSQ ,  we can test the difference and do some modification. 

#### **Detailed execution plan and timeline**

1. Read the paper [1] LEARNED STEP SIZE QUANTIZATION carefully. Understand the detail of the LSQ algorithm.  Date before: 4/22
2. Find a model compression tool in pytorch for model pruning and fine tuning. This can be the link https://github.com/synxlin/nn-compressionfor introduced in course project announcement document. Date before: 4/22
3. Use a reference implement of github and download the code in HPC LAB and install the requirement of the codes, and play with some example run. On option for the reference is https://github.com/zhutmost/lsq-net. We may need to compare with other implementation. Date before: 4/23
4. Understand the LSQ and pruning code carefully to see the implementation detail and compare it with the paper and make sure it can work on HPC. Date before: 4/24
5. Merge the codes together and making the pruning codes and LSQ works together. Date before: 4/25
6. Training the Resnet with CIFAR10 and get a model A, baseline accuracy A. Date before: 4/26
7. Fine tune the model A to get a model B and get a reasonable accuracy B. Date before: 4/27
8. Apply LSQ on model B using bit 3, 4, 5, 6, 7, 8 and get corresponding accuracy. Date before: 4/28
9. Using digram to illustrate the LSQ difference with accuracy A, and accuracy B. Date before: 4/29
10. Training the ShuffleNet V2 with CIFAR10 and get a model SA, baseline accuracy SA. Date before: 5/1
11. Apply LSQ on model SA using bit 3, 4, 5, 6, 7, 8 and get corresponding accuracy. Date before: 5/2
12. Using digram to illustrate the LSQ difference with accuracy SA. Date before: 5/2
13. Write the presentation slide. Date before: 5/3
14. Continue refine  the results and see if we can make additional extension: 5/10
14. Write the final project paper. Date before: 5/13

#### **References**
[1] Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., and Modha, D. S., “Learned Step Size Quantization”, *arXiv e-prints*, 2019., [ https://doi.org/10.48550/arXiv.1902.08153], international Conference on Learning Representations (2020)

[2]Zhu, M. and Gupta, S., “To prune, or not to prune: exploring the efficacy of pruning for model compression”, *arXiv e-prints*, 2017.https://arxiv.org/abs/1710.01878
