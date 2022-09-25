# 卷积基础

卷积适用的数据：例如图像

1. translation invariance 平移不变性：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
2. locality 局部性：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

卷积的特点：

* 权重共享：卷积计算实际上是使用一组卷积核在图片上进行滑动，计算乘加和。因此，对于同一个卷积核的计算过程而言，在与图像计算的过程中，它的权重是共享的

* 保存空间信息：计算范围是在像素点的空间邻域内进行的，它代表了对空间邻域内某种特征模式的提取。对比全连接层将输入展开成一维的计算方式，卷积运算可以有效学习到输入数据的空间信息。
* 局部连接：在卷积操作中，每个神经元只与局部的一块区域进行连接。对于二维图像，局部像素关联性较强，这种局部连接保证了训练后的滤波器能够对局部特征有最强的响应，使神经网络可以提取数据的局部特征
* **不同层级卷积提取不同特征**：在CNN网络中，通常使用多层卷积进行堆叠，从而达到提取不同类型特征的作用。比如:浅层卷积提取的是图像中的边缘等信息；中层卷积提取的是图像中的局部信息；深层卷积提取的则是图像中的全局信息。

# 卷积模型的发展

## LeNet-5

[论文链接](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

![lenet_model](image/lenet_model.png)

最早的卷积神经网络之一，LeNet被广泛用于自动取款机（ATM）机中，帮助识别处理支票的数字。由两个部分组成：

* 两个卷积conv + 池化层pool
* 三个全连接层full-connect

![../_images/lenet-vert.svg](https://d2l.ai/_images/lenet-vert.svg)

**模型结构**

第一个卷积层的in_channel=1, out_channel=6, kernel=5 * 5，stride=1，pad=2，激活函数为sigmoid

​           池化层的kernel=2，stride=2，使用了avgpool

第二个卷积层的in_channel=6, out_channel=16，kernel=5 * 5，stride=1，pad=0，激活函数为sigmoid

​           池化层的kernel=2，stride=2，使用了avgpool

对于一张输入像素为28 * 28的单通道图像而言，每一层的输出shape为

``````python
Input:          [1, 1, 28, 28]
Conv2d:         [1, 1, 28, 28]
AvgPool2d:      [1, 6, 14, 14]
Conv2d:         [1, 16, 10, 10]
AvgPool2d:      [1, 16, 5, 5]
Flatten:        [1, 400]
Linear:         [1, 120]
Linear:         [1, 84]
Linear:         [1, 10]
``````



## AlexNet

[论文链接](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

![alexnet_model](image/alexnet.png)

LeNet 在早期的小数据集上可以取得好的成績，但在更大的数据集的表现却不如人意，AlexNet在2012年ImageNet挑战赛中取得了轰动一时的成绩，开启了CNN的时代。AlexNet由几个部分组成

* 五个卷积层（其中第一、二、五个带池化层）
* 三个全连接层

![../_images/alexnet.svg](https://d2l.ai/_images/alexnet.svg)

**模型结构**

第一个卷积层的in_channel=1, out_channel=96, kernel=11 * 11，stride=4，pad=1，激活函数为ReLU

​            池化层的kernel=3，stride=2，使用了maxpool

第二个卷积层的in_channel=96, out_channel=256，kernel=5 * 5，stride=1，pad=2，激活函数为ReLU

​            池化层的kernel=3，stride=2，使用了maxpool

第三个卷积层的in_channel=256, out_channel=384，kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

第四个卷积层的in_channel=384, out_channel=384，kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

第五个卷积层的in_channel=384, out_channel=256，kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=3，stride=2，使用了maxpool

AlexNet的特点

* 使用maxpool，保存重要的特征，并且池化层的kernel > stride，pooling过程中有重叠，避免重要的特征被舍弃

* 第一层的kernel较大，扩大感受野，并且使用较大的步长提取特征

* 使用ReLU避免了梯度消失问题，且收敛速度较快

* 使用了dropout，增强了模型的泛化能力

* 使用了data augmentation增加了训练资料

* 使用GPU训练，加快了模型训练的速度

## VGG11

[论文链接](https://arxiv.org/pdf/1409.1556.pdf)

![img](https://miro.medium.com/max/1400/1*tnJji1tdkNSTxNDDVrZPog.png)

经典卷积神经网络的基本组成部分是下面的这个序列：

1. 卷积层Conv
2. 非线性激活函数，如ReLU
3. 池化层Pool

VGG网络提出了块的概念，每个块包含1-2个卷积层、ReLU和池化层，VGG网络由5个块和三个全连接层组成

![../_images/vgg.svg](https://d2l.ai/_images/vgg.svg)

**模型结构**

第一个块包含一个卷积层，in_channel=1, out_channel=64, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=2，stride=2，使用了maxpool

第二个块包含一个卷积层，in_channel=64, out_channel=128, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=2，stride=2，使用了maxpool

第三个块包含两个卷积层，in_channel=128, out_channel=256, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=2，stride=2，使用了maxpool

第四个块包含两个卷积层，in_channel=256, out_channel=512, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=2，stride=2，使用了maxpool

第五个块包含两个卷积层，in_channel=512, out_channel=512, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​            池化层的kernel=2，stride=2，使用了maxpool



**VGG的特点**

* VGG-11使用可复用的卷积块构造网络。不同的VGG模型可通过每个块中卷积层数量和输出通道数量的差异来定义。

* 块的使用导致网络定义的非常简洁。使用块可以有效地设计复杂的网络。

## Network in Network

[论文链接](https://arxiv.org/pdf/1312.4400.pdf)

![image-20220919195649781](image/nin.png)

NiN的网络结构中也有块的概念。每个NiN块由三个卷积层组成，其中第二、三个卷积层时1 * 1的卷积层，充当带有ReLU激活函数的逐像素全连接层。NiN网络由四个NiN块和四个池化层组成。

 ![../_images/nin.svg](https://zh.d2l.ai/_images/nin.svg)

**模型结构**

第一个块包含三个卷积层，

​		第一个卷积层的in_channel=1, out_channel=96, kernel=11 * 11，stride=4，pad=0，激活函数为ReLU

​		第二、三个卷积层的in_channel=96, out_channel=96, kernel=1 * 1，stride=1，pad=0，激活函数为ReLU

第一个池化层的kernel=3，stride=2，使用了maxpool

第二个块包含三个卷积层，

​		第一个卷积层的in_channel=96, out_channel=256, kernel=5 * 5，stride=1，pad=2，激活函数为ReLU

​		第二、三个卷积层的in_channel=256, out_channel=256, kernel=1 * 1，stride=1，pad=0，激活函数为ReLU

第二个池化层的kernel=3，stride=2，使用了maxpool

第三个块包含三个卷积层，

​		第一个卷积层的in_channel=256, out_channel=384, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​		第二、三个卷积层的in_channel=384, out_channel=384, kernel=1 * 1，stride=1，pad=0，激活函数为ReLU

第三个池化层的kernel=3，stride=2，使用了maxpool

第四个块包含三个卷积层，

​		第一个卷积层的in_channel=384, out_channel=10, kernel=3 * 3，stride=1，pad=1，激活函数为ReLU

​		第二、三个卷积层的in_channel=10, out_channel=10, kernel=1 * 1，stride=1，pad=0，激活函数为ReLU

第四个池化层的kernel=1，stride=1，使用AdaptiveAvgPool2d

**NiN的特点**

- NiN使用由一个卷积层和多个1×1卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的每像素非线性。
- NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和）。该Pool通道数量为所需的输出数量（例如，Fashion-MNIST的输出为10）。移除全连接层可减少过拟合，同时显著减少NiN的参数。

## GoogleNet

[论文链接](https://wmathor.com/usr/uploads/2020/01/3184187721.pdf)

GoogLeNet 是由 Google 开发的，比 AlexNet 参数量少，但网络更深、准确率更高，在 2014 年 ImageNet LSVRC 分类竞赛中获得了冠军。在GoogLeNet中，基本的卷积块被称为*Inception块*（Inception block）

![image-20220921232907318](image/googlenet_inception.png)

如上图所示，Inception块由四条并行路径组成。 前三条路径使用窗口大小为1×1、3×3和5×5的卷积层，从不同空间大小中提取信息。 中间的两条路径在输入上执行1×1卷积，以减少通道数，从而降低模型的复杂性。 第四条路径使用3×3MaxPool，然后使用1×1卷积层来改变通道数。 这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。

**模型结构**

![../_images/inception-full-90.svg](https://d2l.ai/_images/inception-full-90.svg)

如图所示，GoogLeNet一共使用9个Inception块和GlobalAvgPool的堆叠来生成其估计值。Inception块之间的MaxPool层可降低维度。 GoogLeNet 是深层的神经网络，因此会遇到梯度消失的問題。

<img src="image/googlenet.png" alt="image-20220921233343752" style="zoom:40%;" />

分类辅助器就是为了避免这个问题，并提高其稳定性和收敛速度，上图是分类辅助器的结构。方法是会在两个不同层的 Inception module输出结果并計算 loss，最后在将这两个 loss 跟真实 loss 加权求和。

**GoogleNet的特点**

- Inception块相当于一个有4条路径的子网络。它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用1×1卷积层减少每像素级别上的通道维数从而降低模型复杂度。
- GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
- GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度。

## ResNet

[论文链接]( https://arxiv.org/pdf/1512.03385.pdf)

ResNet 在 2015 年由微软的何愷明博士提出，并于同年 ImageNet LSVRC 分类竞赛中获得了冠军。在竞赛中 ResNet 一共使用了 152 层网络，其深度比 GoogLeNet 高了七倍多 (20层)，並且错误率为 3.6%，而人类在 ImageNet 的错误率为 5.1%，已经达到小于人类的错误率的程度。原因是ResNet提出的残差学习让深层网络更容易训练。

假设我们的原始输入为x，而希望学出的理想映射为f(x)。左图虚线框中的部分需要直接拟合出该映射f(x)，而右图虚线框中的部分则需要拟合出残差映射f(x)−x。 残差映射在现实中往往更容易优化。我们只需将右图虚线框内上方的加权运算的权重和偏置参数设成0，那么f(x)即为恒等映射。 实际中，当理想映射f(x)极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。右图是ResNet的基础架构–*残差块*（residual block）。 在残差块中，输入可通过跨层数据线路更快地向前传播。



![../_images/resnet-block.svg](https://d2l.ai/_images/resnet-block.svg)

**模型结构**

ResNet沿用了VGG完整的3×3卷积层设计。 残差块里首先有2个有相同输出通道数的3×3卷积层。 每个卷积层后接一个批量规范化层和ReLU激活函数。 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。 这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。 如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。 



![../_images/resnet18-90.svg](https://d2l.ai/_images/resnet18-90.svg)

ResNet的前两层跟之前介绍的GoogLeNet中的一样： 在输出通道数为64、步幅为2的7×7卷积层后，接步幅为2的3×3的MaxPool层。 不同之处在于ResNet每个卷积层后增加了BatchNorm层。GoogLeNet在后面接了4个由Inception块组成的模块。 ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。 第一个模块的通道数同输入通道数一致。 由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽。 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。



![../_images/resnext-block.svg](https://d2l.ai/_images/resnext-block.svg)

每个模块有4个卷积层（不包括恒等映射的1×1卷积层）。 加上第一个7×7卷积层和最后一个全连接层，共有18层。 因此，这种模型通常被称为ResNet-18。 通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。 虽然ResNet的主体架构跟GoogLeNet类似，但ResNet架构更简单，修改也更方便。这些因素都导致了ResNet迅速被广泛使用。

**ResNet特点**

- 学习嵌套函数（nested function）是训练神经网络的理想情况。在深层神经网络中，学习另一层作为恒等映射（identity function）较容易（尽管这是一个极端情况）。
- 残差映射可以更容易地学习同一函数，例如将权重层中的参数近似为零。
- 利用残差块（residual blocks）可以训练出一个有效的深层神经网络：输入可以通过层间的残余连接更快地向前传播。
- 残差网络（ResNet）对随后的深层神经网络设计产生了深远影响。

## DenseNet

​	[论文链接]( https://arxiv.org/pdf/1608.06993.pdf)

DenseNet 有别于以往的神经网络，不从网络的深度着手，而是由特征的角度去考慮，加強了特征的利用、减轻梯度消失的問題，大幅地减少了参数计算量。其做法就是将前面所有层的 feature map 作为输入，然后将其 concate 起來聚合信息，如此一来可以保留前面的特征，称为特征重用 (feature reuse)，让结构更加密集，提高了网络的信息和梯度流動，使得网络更加容易训练

![../_images/densenet-block.svg](https://d2l.ai/_images/densenet-block.svg)



如图所示，ResNet和DenseNet的关键区别在于，DenseNet输出是连接（用图中的[,]表示）而不是如ResNet的简单相加。 DenseNet这个名字由变量之间的“稠密连接”而得来，最后一层与之前的所有层紧密相连。 稠密连接如图所示。

<img src="image/densenet.png" alt="image-20220922081757558" style="zoom:50%;" />

**Dense block** 采用 Batch Normalization、ReLU、3x3 卷积层的结构，与ResNet block 不同的是 Dense block 有个超参数k 称为 growth rate，是指每层輸出的 channel 数目为 k 个 (输出通道数会随着层数而增加)。为了不使网络变宽，通常使用较小的 k

因为特征重用的原因，输出通道数会随着层数而增加，因此 Dense block 会采用 **Bottleneck** 来降低通道数、减少参数计算量

Bottleneck 的结构为 Batch Normalization、ReLU、1x1 卷积层、Batch Normalization、ReLU、3x3 卷积层，称为 DenseNet-B

Transition layer 目的是為了要压缩模型，使模型不會过于复杂，主要是将两个 Dense block 去做连接，使用 1x1 卷积层降维，並且使用平均池化來缩小特征图的尺寸

**模型结构**

![image-20220921234952128](image/densenet_model.png)

**DenseNet特点**

- 在跨层连接上，不同于ResNet中将输入与输出相加，稠密连接网络（DenseNet）在通道维上连结输入与输出。
- DenseNet的主要构建模块是稠密块和过渡层。
- 在构建DenseNet时，我们需要通过添加过渡层来控制网络的维数，从而再次减少通道的数量。

## YOLOv3



## MobileNet