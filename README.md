# DeepLearningCompilation
深度学习编译/AI芯片加速相关的的知识

[一、深度学习模型](model)

* [常见网络模型](model/network)
  * [卷积模型](model/network/CNN)
  * [序列模型](model/network/RNN)
  * [生成对抗模型](model/network/GAN)
  * [Transformer](model/network/transformer)
  * [预训练模型](model/network/pre_train)
* [常见网络层](model/operation)
  * [Matmul](model/operation/)
  * [Convolution](model/operation/convolution)
  * [Pooling](model/operation/pooling)
  * [Dropout](model/operation/dropout)
  * [BatchNormalization](operation/batch_normalization)
  * [Softmax](model/operation/softmax)
  * [Activation](model/operation/activation)
  

[二、深度学习编译]((compile))

* [芯片基础](compile/chip_basis)
  * [芯片架构](compile/chip_basis/chip_architecture.md)
* [TVM编译器](compile/tvm_basis)
  * [TVM原语](compile/tvm_basis/compute_schedule.ipynb)
  * [TVM算子加速](compile/tvm_basis/common_operation.ipynb)
  
* [nn加速库](compile/nn_lib)
  * [ NNVM](compile/nn_lib/NNVM.md)
  * [ MNN](compile/nn_lib/MNNmd)
  * [TensorRT](compile/nn_lib/tensorRT.md)
  * [cuDNN](compile/nn_lib/cuDNN.md)

[三、深度学习编译实践](endtoend)

* [GPU端到端部署模型](GPU)

参考资料

* [深度学习百科及面试资源](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/index.html)
* [ dive into deep learning](https://d2l.ai/)
