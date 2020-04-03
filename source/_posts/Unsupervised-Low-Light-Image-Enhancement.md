---
title: Unsupervised Low-Light Image Enhancement
date: 2020-03-28 12:21:55
tags: ["Low-Light Image Enhancement", "Deep Learning", "GAN"]
categories: Low-Light Image Enhancement
mathjax: true
---
　　在自动驾驶中，相机能捕捉丰富的纹理信息，是不可或缺的传感器。但是受限于相机 Sensor 及 ISP 性能，其动态范围有限，往往会出现过曝或欠曝的情况。过曝的情况还能通过 3A(AE, AF, AW) 中的 AE 调节，而欠曝的情况，AE 中要么提高增益或 ISO 但是会增加噪声，要么增加曝光时间但是撑死 50ms(按照 20Hz)，光圈则一般是固定的，不会调节。所以在低光照自动驾驶场景下，对欠曝的图像进行亮度增强则显得尤其重要（当然也可用夜视相机如红外相机等辅助)。  
　　基于学习的图像增强方法，由于很难获得大量的欠爆图像与对应的增强图像。所以无监督的图像增强方法就更有应用价值，本文介绍几种无监督图像增强方法。

## 1.&ensp;Zero-DCE<a href="#1" id="1ref"><sup>[1]</sup></a>
　　无监督图像增强方法主要是指基于 GAN 的方法，基于 GAN 的方法还是需要选择欠爆图像及正常图像两个分布的数据集，选择不当也会导致性能下降。而 Zero-DCE 则无需选择正常图像数据集，消除了数据分布下过拟合或欠拟合的风险。  
　　Zero-DCE 基本思想是对每个像素作亮度变换，每个像素的变换方程为：
$$LE(I(\mathrm{x});\alpha) = I(\mathrm{x}) + \alpha I(\mathrm{x})(1-I(\mathrm{x})) \tag{1}$$
其中 \\(\\alpha\\in\[-1,1\]\\) 是变换系数。对图像的每个通道每个像素分别作不同系数的迭代变换，可得：
$$LE _ n(\mathrm{x}) = LE _ {n-1}(\mathrm{x}) + \mathcal{A} _ n LE _ {n-1}(\mathrm{x})(1-LE _ {n-1}(\mathrm{x})) \tag{2}$$
其中 \\(\\mathcal{A} _ n\\) 是变换系数集，与图像大小一致。
<img src="Zero-DCE.png" width="90%" height="90%" title="图 1. Zero-DCE Framework">
　　如图 1. 所示，Zero-DCE 框架中，一个基本网络预测几组 \\(\\mathcal{A} _ n\\) 集合，然后对原图每个通道进行迭代的亮度变换。LE-curves 不仅能增强暗处的曝光量，还能减弱过曝处的亮度值。  
　　该方法最重要的是 Loss 函数的设计，一共有以下 Loss 组成：

1. **Spatial Consisiency Loss**  
增强后的图像要求其与原图具有空间一致性：
$$ L _ {spa} = \frac{1}{K}\sum _ {i=1}^K\sum _ {j\in\Omega (i)}\left(\Vert Y _ i-Y _ j\Vert-\Vert I _ i-I _ j\Vert\right)^2 \tag{3}$$
其中 \\(\\Omega\\) 为某像素的领域集，可为四领域；\\(K\\) 为局部区域数量，可设定为 \\(4\\times 4\\) 大小；\\(Y,I\\) 分别为增强后与原始的像素亮度值。
2. **Exposure Control Loss**  
曝光控制 Loss 相当于设定曝光量去监督训练每个像素亮度，实现“无监督”的效果：
$$ L _ {exp} = \frac{1}{M}\sum _ {k=1}^M\Vert Y _ k-E\Vert \tag{4}$$
其中 \\(M\\) 为无重合的局部区域数量，可设定为 \\(16\\times 16\\) 大小；\\(Y _ k\\) 为局部区域的平均亮度值。作者实验中，设定 \\(E\\in\[0.4,0.7\]\\) 均能获得相似的较好的结果。
3. **Color Constancy Loss**  
根据 Gray-World color constancy 假设：rgb 每个通道的平均亮度值与 gray 灰度值一致。所以为了保证颜色不失真，构造：
$$ L _ {col}=\sum _ {\forall (p,q)\in \epsilon}(J^p-J^q), \epsilon=\{R,G,B\} \tag{5}$$
其中 \\(p,q\\) 表示一对不同的颜色通道，\\(J\\) 表示该通道的平均亮度值。
4. **Illumination Smoothness Loss**  
增强的过程要求相邻亮度值是平滑的，对增强变换系数作约束：
$$ L _ {tv _ {\mathcal{A}}} = \frac{1}{N}\sum _ {n=1}^N\sum _ {c\in\epsilon}(\nabla _ x\mathcal{A} _ n^c+\nabla _ y\mathcal{A} _ n^c)^2, \epsilon = \{R,G,B\}\tag{6}$$
其中 \\(N\\) 为增强迭代数；\\(\\nabla _ x,\\nabla _ y\\) 分别表示水平与垂直方向的求导操作。

最终 Loss 构成为：
$$ L _ {total} = L _ {spa} + L _ {exp} + W _ {col}L _ {col} + W _ {tv _ {\mathcal{A}}}L _ {tv _ {\mathcal{A}}} \tag{7}$$

## 2.&ensp;EnlightenGAN<a href="#2" id="2ref"><sup>[2]</sup></a>
　　图像增强本质上是作 domain transfer，所以能用 GAN 处理，实现无监督训练。
<img src="EnlightenGAN.png" width="90%" height="90%" title="图 2. EnlightenGAN Framework">
　　如图 2. 所示，EnlightenGAN 由 Generator 和 Discriminator 构成。Generator 是一个 attention-guided U-Net，因为我们期望欠曝的区域能增强，所以将亮度值归一化后，用 1 减去亮度值作为注意力图，与原图一起输入网络。Discriminator 由 Global Discriminator 与 Local Discriminator 组成，因为经常只需要局部区域的亮度，所以设计 Local Discriminator 就很有必要。  
　　Loss 的设计非常关键，EnlightenGAN 一共有以下 Loss 组成：

1. **Adversarial Loss**  
用于直接训练 Generator 以及 Discriminator 的 Loss，与传统的 GAN Loss 类似；
2. **Self Feature Preserving Loss**  
注意到，调整输入图像值的范围，对最终的高层任务影响不是很大，所以引入网络特征 Loss 来保证增强后图像的准确性。对原始图像与生成的图像，分别输入到在 ImageNet 上预训练的 VGG-16 模型，提取特征集合，将对应的特征对作 L1 Loss。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Guo, Chunle, et al. "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement." arXiv preprint arXiv:2001.06826 (2020).  
<a id="2" href="#2ref">[2]</a> Jiang, Yifan, et al. "Enlightengan: Deep light enhancement without paired supervision." arXiv preprint arXiv:1906.06972 (2019).
