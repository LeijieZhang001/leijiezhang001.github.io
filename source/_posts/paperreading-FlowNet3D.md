---
title: '[paper_reading]-"FlowNet3D"'
date: 2019-10-22 19:51:11
tags: ["paper reading", "Scene Flow", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: paper reading
mathjax: true
---
　　本来以为这篇文章是 FlowNet<a href="#1" id="1ref"><sup>[1]</sup></a>，FlowNet2.0<a href="#2" id="2ref"><sup>[2]</sup></a> 的续作，其实不是，大概只是借鉴了其网络框架。从网络细节上来说，应该算是 PointNet<a href="#3" id="3ref"><sup>[3]</sup></a>，PointNet++<a href="#4" id="4ref"><sup>[4]</sup></a> 系列的续作，本文<a href="#5" id="5ref"><sup>[5]</sup></a>二作也是 PointNet 系列的作者。  
　　光流(Optical Flow)是指图像坐标系下像素点的运动(详细可见 {% post_link KLT KLT%})，而 Scene Flow 是三维坐标下，物理点的运动。Scene Flow 是较底层的一种信息，可进一步提取高层的语义信息，如运动分割等。

## 1.&ensp;背景
### 1.1.&ensp;FlowNet 系列
<img src="flownet.png" width="90%" height="90%" title="图 1. FlowNet">
<img src="refine.png" width="80%" height="80%" title="图 2. FlowNet Refinement">
　　如图 1. 与 2. 所示，FlowNet 在特征提取编码阶段提出了两种网络结构：FlowNetSimple 以及 FlowNetCorr。FlowNetSimple 是将前后帧图像按通道维拼接作为输入，FlowNetCorr 则设计了互相关层，描述前后帧特征的相关性，从而得到像素级偏置。refinement 解码阶段则采用 FPN 形式进行上采样，这样每一层反卷积层在细化时，不仅可以获得深层的抽象信息，同时还能获得浅层的具体信息。
<img src="flownet2.png" width="90%" height="90%" title="图 3. FlowNet2.0">
　　FlowNet 虽然验证了用深度学习预测光流的可行性，但是性能比不上传统方法。FlowNet2.0 在此基础上进行了三大改进：

- **增加训练数据，改进训练策略**；  
在数据足够的情况下，证明了 FlowNetCorr 比 FlowNetSimple 较好。
- **利用堆叠结构使性能得到多级提升**；  
如图 3. 所示，采用 FlowNet2-CSS 形式堆叠一个 FlowNetCorr 以及两个 FlowNetSimple 模块，FlowNetSimple 的输入为前一模块预测的光流，原图像经过光流变换后的图像，以及与另一图像的误差，这样可以使得该模块专注去学习前序模块未预测准确的误差项。训练时，由前往后单独训练每个模块。
- **针对小位移的情况引入特定的子网络进行处理**；  
如图 3. 所示，FlowNet2-SD 网络卷积核均改为 3x3 形式，以增加对小位移的分辨率。最后再利用一个小网络将 FlowNet2-CSS 与 FlowNet2-SD 的结果进行融合。


### 1.2.&ensp;PointNet 系列
　　这部分详见 {% post_link PointNet-系列论文详读 PointNet-系列论文详读 %}。  
　　这里介绍下 PointNet++ 中点云采样的过程。点云采样有集中形式：

- 格点采样  
空间栅格化，然后按照栅格进行点云采样；
- 随机采样  
- 几何采样  
根据点云所在平面的曲率，将点云分成不同集合，在每一集合里面进行均匀采样，获得曲率大的地方采样点多的效果，即获得更多“细节”；
- 均匀采样  

PointNet++ 中采用的 Farthest Point Sample 属于均匀采样，其可以采样出特定个数的点，且比较均匀。大致过程为：

1. 点云总集合为 \\(\\mathcal{C}\\)，随机取一点，形成采样目标集合 \\(\\mathcal{S}\\)；
2. 在剩余点集 \\(\\mathcal{C}-\\mathcal{S}\\) 中取与集合 \\(\\mathcal{S}\\) 距离最远的一点，加入目标集合 \\(\\mathcal{S}\\)；
3. 如果目标集合 \\(\\mathcal{S}\\) 个数达到预定值，则终止，否则重复步骤 2.；

## 2.&ensp;FlowNet3D 网络结构
<img src="flownet3d.png" width="90%" height="90%" title="图 4. FlowNet3D">
　　如图 4. 所示，FlowNet3D 整体思路与 FlowNetCorr 非常像，其 set conv，flow embedding，set upconv 三个层相当于 FlowNetCorr 中的 conv，correlation，upconv 层。网络结构的连接方式也比较相像，上采样的过程都有接入前面浅层的具体特征。下面重点分析下这三个层的细节。
<img src="flownet3d-layers.png" width="90%" height="90%" title="图 5. FlowNet3D Layers">
　　假设两个连续帧的两堆点：\\(\\mathcal{P} = \\{x_i\\vert i = 1,...,n_1\\}\\) 以及 \\(\\mathcal{Q} = \\{y_j\\vert j = 1,...,n_2\\}\\)，其中 \\(x_i, y_j \\in \\mathbb{R}^3\\) 是每个点的物理空间坐标。Scene Flow 的目标是求解 \\(\\mathcal{D}=\\{x_i'-x_i \\vert i = 1,...,n_1\\} = \\{d_i\\vert i=1,...,n_1\\}\\)，其中 \\(x_i'\\) 是 \\(x_i\\) 在下一帧的位置。图 5. 较清晰地阐述了这三个层对点云的作用：

### 2.1.&ensp;set conv layer
　　set conv layer 就是 PointNet++ 中的 set abstraction layer，其作用相当于图像中的卷积操作，能提取环境上下文特征。假设输入 \\(n\\) 个点，每个点 \\(p_i = \\{x_i, f_i\\}\\)，其中 \\(x_i\\in \\mathbb{R}^3\\) 是物理坐标空间，\\(f_i\\in\\mathbb{R}^c\\) 是特征空间；输出 \\(n'\\) 个点，对应每个点为 \\(p_j'=\\{x_j',f_j'\\}\\)，其中 \\(f_j'\\in\\mathbb{R}^{c'}\\) 为特征空间。那么 set conv layer 可以描述为：
$$f_j' = \max_{\left\{i\vert\Vert x_i-x_j'\Vert \leq r\right\}}\left\{\mathbf{h}\left(\mathrm{concat}(f_i,x_i-x_j')\right)\right\}$$
其中 \\(x_j'\\) 是输入的 \\(n\\) 个点经过 Farthest Point Sample 后的点集，感知机 \\(\\mathbf{h}\\) 将空间 \\(\\mathbb{R}^{c+3}\\) 映射到空间 \\(\\mathbb{R}^{c'}\\)，然后进行 max 操作。

### 2.2.&ensp;flow embedding layer
　　有了 PointNet 思想后，其实比较容易想到如何进行两个点云的特征融合提取(看论文之前，自己有想过，和论文一样⊙o⊙)。对于两个点集：\\(\\left\\{p_i = \\{x_i, f_i\\}\\right\\}\_{i=1}^{n_1}\\) 以及 \\(\\left\\{q_j = \\{y_j, g_j\\}\\right\\}\_{j=1}^{n_2}\\)，其中 \\(x_i,y_j\\in\\mathbb{R}^3\\)，特征量 \\(f_i,g_j\\in\\mathbb{R}^c\\)， 那么输出为：\\(\\left\\{o_i=\\{x_i,e_i\\}\\right\\}\_{i=1}^{n_1}\\)，其中 \\(e_i\\in\\mathbb{R}^{c'}\\)。由此 flow embedding layer 可描述为：
$$e_i = \max_{\left\{j\vert\Vert y_j-x_i\Vert \leq r\right\}}\left\{\mathbf{h}\left(\mathrm{concat}(f_i,g_j,y_j-x_i)\right)\right\}$$
可见，其数学形式与 set conv layer 基本一致，但是物理意义是完全不一样的， flow embedding layer 是以 \\(x_i\\) 为锚点，在另一堆点云中找到距离 \\(r\\) 范围内的点，从何提取特征，用来描述该点与另一堆点云的相关性。这里的感知机作用可以有其它形式，作者试验后发现这种方式最简单有效。

### 2.3.&ensp;set upconv layer
　　PointNet++ 中 refinement 过程是 3D 插值上采样与 unit pointnet 过程，这里作者参考图像中 conv2D 与 upconv2D 的关系，提出了 set upconv layer。图像中 upconv2D 可以认为是特征扩大+填0+conv的结合(插值上采样则等价于扩大+插值的过程)，那么类似的，set upconv layer 就是点云扩大后，再对每个目标点进行 set conv layer 的操作。  
　　作者称这种方法比纯插值上采样好(这当然了)，也有可能是称比插值上采样+unit pointnet 好？但是这种方法本质上还是插值上采样+pointnet。

## 3.&ensp;其它细节
### 3.1.&ensp;Training Loss
　　输入两堆点云： \\(\\mathcal{P}=\\{x_i\\}\_{i=1}^{n_1}\\), \\(\\mathcal{Q}=\\{y_j\\}\_{j=1}^{n_2}\\)，网络预测的 Scene Flow 为 \\(\\mathcal{D}=F(\\mathcal{P,Q;\\theta})=\\{d_i\\}\_{i=1}^{n_1}\\)， 真值为 \\(\\mathcal{D}^\*=\\{d_i\^\*\\}\_{i=1}\^{n_1}\\)。经过 Scene Flow 变换后的点云为：\\(\\mathcal{P'}=\\{x_i+d_i\\}\_{i=1}^{n_1}\\)，那么经过网络预测的反向的 Scene Flow 为 \\(\\{d_i'\\}\_{i=1}^{n_1}=F(\\mathcal{P',P;\\theta})\\)，由此定义 cycle-consistency 项 \\(\\Vert d_i'+d_i\\Vert\\)，最终的 Loss 函数为：
$$L(\mathcal{P,Q,D^*,\theta})=\frac{1}{n_1}\sum_{i=1}^{n_1}\left(\Vert d_i-d_i^*\vert+\lambda\Vert d_i'+d_i\Vert\right)$$

### 3.2.&ensp;Three Meta-architectures
<img src="mixture.png" width="60%" height="60%" title="图 6. 三种特征融合方式对比">
　　如图 6. 所示，两个点云集合特征融合方式有三种，作者的 baseline 模型也是基于这三种，flow embedding layer 属于 Deep Mixture 类型。

### 3.3.&ensp; Runtime
<img src="runtime.png" width="70%" height="70%" title="图 7. NIVIDA 1080 GPU with TensorFlow">
　　速度嘛，还是比较慢的，要应用得做优化。

### 3.4.&ensp; Applications: Scan Registration & Motion Segmentation
　　待补充。

## 4.&ensp;实验结果
<img src="ablation.png" width="60%" height="60%" title="图 8. ablation study">
　　如图 8. 所示，可得结论：

- PointNet 中 max 操作比 avg 操作效果好；
- 上采样中 upconv 比 interpolation 效果好；
- cycle-consistency loss 项有助于提升性能；

## 5.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Dosovitskiy, Alexey, et al. "Flownet: Learning optical flow with convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.  
<a id="2" href="#2ref">[2]</a> Ilg, Eddy, et al. "Flownet 2.0: Evolution of optical flow estimation with deep networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.  
<a id="3" href="#3ref">[3]</a> Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.  
<a id="4" href="#4ref">[4]</a> Qi, Charles Ruizhongtai, et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." Advances in neural information processing systems. 2017.  
<a id="5" href="#5ref">[5]</a> Liu, Xingyu, Charles R. Qi, and Leonidas J. Guibas. "Flownet3d: Learning scene flow in 3d point clouds." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
