---
title: '[paper_reading]-"Multi-Task Multi-Sensor Fusion for 3D Object Detection""'
date: 2019-10-14 10:42:54
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving"]
categories: 3D Detection
mathjax: true
---

　　本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种 3D 检测的多任务多传感器融合方法。输入数据为图像以及点云，输出为地面估计，2D/3D检测，稠密深度图。为了让其它任务来帮助提升 3D 检测效果，作者设计了很多方法，工作还是比较细致且系统。  
<img src="算法框架.png" width="90%" height="90%" title="图 1. 算法框架">
　　整个算法框架如图 1. 所示。点云数据还是在俯视图(BEV)下进行栅格化处理，高度切割是在地面估计归一化后的基础上来做，因为要 3D 定位的目标都是在地面上的；另一方面，图像与投影到前视图的点云数据进行合并，作为网络的输入数据。
网络结构上作者提出了两种俯视图与前视图特征融合策略：1. Point-wise feature fusion; 2. ROI-wise feature fusion. 这也是文章比较重要的一个贡献点。  
　　文章所提的 3D 检测方法大多数细节技巧并无新意，这里主要讨论分析文章中与传统方法不太一样的两大贡献点：
1. 俯视图与前视图特征融合策略；
2. 其它两个任务对检测任务提升的作用。

## 1.&ensp;俯视图与前视图特征融合策略
　　由于网络输入有俯视图与前视图两个数据流，所以如何将这两个数据流进行特征级别的融合就显得尤为重要，文章提出了两种方式，backbone 网络级别的 point-wise feature fusion 以及第二阶段 ROI-wise feature fusion。

### 1.1.&ensp;Point-wise Feature Fusion
　　3D 检测主体还是在俯视图下来做的，相比前视图对 3D 检测的处理，俯视图 3D 检测有天然的优势。因此，如何有效地将前视图的特征融合到俯视图的特征中，就显得尤为重要（俯视图特征融合到前视图相对比较简单）。  
<img src="point-wise.png" width="55%" height="55%" title="图 2. Point-wise Feature Fusion">
　　如图 2. 所示，像素点级别的特征融合方式有两个模块，Multi-scale Fusion 以及 Continuous Fusion。Multi-scale Fusion 我们比较熟悉，可以采用类似 FPN 的结构实现。这里主要讨论 Continuous Fusion 模块。  
<img src="算法框架2.png" width="90%" height="90%" title="图 3. Deep Continuous Fusion 检测框架">
　　Continuous Fusion 源自作者的另一篇文章<a href="#2" id="2ref"><sup>[2]</sup></a>。如图 3. 所示，该文检测框架基本就是本文的主干，其中 Fusion Layers 就是 Continuous Fusion 模块。而 continuous fusion 前身是作者团队提出的 Deep Parametric Continuous Convolution<a href="#3" id="3ref"><sup>[3]</sup></a>。  

- **Deep Parametric Continuous Convolution**  
传统的卷积只能作用于网格结构(gird-structured)的数据上，为了能处理点云这种非网格结构的数据，<a href="#3" id="3ref">[3]</a>提出了带参数的卷积(Parametric Continuous Convolution)。对于第 \\(i\\) 个需要计算的特征位置，其特征值 \\(\\mathrm{h}_i \\in \\mathbb{R}^N\\) 数学形式为：
$$ \mathrm{h}_i=\sum_j \mathbf{MLP}(x_i-x_j)\cdot \mathrm{f}_j $$
其中 \\(j\\) 表示第 \\(i\\) 个点周围的点，\\(\\mathrm{f}_j \\in \\mathbb{R}^N\\) 为输入特征，\\(x_j\\in \\mathbb{R}^3\\) 是点的坐标值。多层感知机 \\(\\mathbf{MLP}\\) 则起到了参数核函数的作用，将 \\(\\mathbb{R}^{J\\times 3}\\) 映射为 \\(\\mathbb{R}^{J\\times N}\\) 空间，用作特征计算的权重值。

- **Continuous Fusion Layer**  
Continuous Fusion 则没有显示得计算卷积权重的过程，这样使得特征提取能力更强，而且计算效率更高，不用存储权重值。其数学描述为：
$$ \mathrm{h}_i=\sum_j \mathbf{MLP}(\mathrm{concat}[\mathrm{f}_j,x_i-x_j]) $$
多层感知机 \\(\\mathbf{MLP}\\) 直接将 \\(\\mathbb{R}^{J\\times (N+3)}\\) 映射到 \\(\\mathbb{R}^{J\\times M}\\) 空间，最后再做一个 element-wise 的相加即得空间为 \\(\\mathbb{R}^{M}\\) 的特征输出(**这个和 PointNet 几乎一模一样，本质就是将每个点的特征空间升维，然后用对称函数(pooling, sum)消除无序点的影响, 只是这里输入的点的特征空间 \\(N\\) 可能已经很大了**)。
<img src="continuous_fusion.png" width="70%" height="70%" title="图 4. Continuous Fusion">
具体步骤如图 4. 所示：

    1. 将点云投影到图像坐标系，在图像特征图上用双线性插值求取每个点对应的图像特征向量；
    2. 俯视图下对于每个需要求取特征的像素点，采样邻近的 \\(K\\) 个物理点，然后应用 Continuous Fusion，得到该像素点的特征向量；

### 1.2.&ensp;ROI-wise Feature Fusion
　　在俯视图上获得 3D 检测框后(见图 1.)，将其分别投影到图像特征图以及点云特征图上，图像特征图上用 ROIAlign 提取出目标框内的图像特征；点云特征图上用类似方法提取出带方向的目标框内的点云特征，两种特征合并到一起，再用网络进行 2D/3D 目标框的优化回归。
<img src="roi-wise.png" width="60%" height="60%" title="图 5. ROI-wise Fusion">
　　如图 5. 所示，点云特征图上的目标框是带有一定方向的，准确提取特征时会有一些问题。由于旋转框有周期性，所以将目标框分成两种情况来考虑，这样提取的特征就没有奇异性了，如图 5.2 所示。此外 3D 优化回归是在目标框旋转后的坐标系下进行的。

## 2.&ensp;多任务对检测任务的提升作用
<img src="ablation.png" width="100%" height="100%" title="图 6. Ablation on Kitti">
<img src="ablation2.png" width="50%" height="50%" title="图 7. Ablation on TOR4D">

### 2.1.&ensp;地面估计
　　俯视图下点云进行栅格化手工提取特征之前，作者作了一个地面归一化的操作。地面估计是在栅格分辨率下进行的，所以自然能对点云的每个栅格进行地面归一化。作者认为自动驾驶 3D 检测的目标都是在地面上的，所以地面的先验知识应该有助于 3D 定位，与 HDNET<a href="#4" id="4ref"><sup>[4]</sup></a> 思想类似。而在线地面估计(地面估计是建图的其中一个任务)不依赖离线地图，能提高系统鲁棒性。  
<img src="ground_est.png" width="70%" height="70%" title="图 8. 目标定位误差">
　　如图 6.,8 所示，地面估计的加入，确实使得 3D 检测性能有所提升。

### 2.2.&ensp;深度估计
　　由于前视图输入的是图像以及点云的投影图，所以可进一步通过网络预测稠密的前视深度图。作者对点云的投影图作了精心的设计，这里不做展开，有可能直接投影的 \\((x,y,z)\\) 3 通道的投影图也够用。  
　　获得了前视稠密深度图后，可将其反投影到点云俯视图下，这样稀疏的点云会变得更加稠密，更有利于图像到点云的 Point-wise Feature Fusion。这里作者只在邻近取不到点云的时候用这反投影的伪雷达点(pseudo LiDARP)。如图 7. 所示，在该数据集上效果提升还是比较明显，而 Kitti 上不太明显，因为两者的相机与雷达配置不太一样。在 TOR4D 数据集上，远距离的车上点云数量更小，所以该技术效果较好。

## 3.&ensp;其它细节
　　Loss 设计为：
$$ Loss = L_{cls} + \lambda(L_{box}+L_{r2d}+L_{r3d}) + \gamma L_{depth} $$
其中 \\(\\lambda\\) 与 \\(\\gamma\\) 为权重项，\\(L_{box}\\) 为俯视图下预测的 3D 框，\\(L_{r2d},L_{r3d}\\) 为优化回归的 2D/3D 框。每一项的 Loss 计算方式与传统无异。
<img src="eval.png" width="90%" height="90%" title="图 9. 算法对比">
　　本文方法与其它方法对比如图 9. 所示。

## 4.&ensp;参考文献
<a id="1" href="#1ref">[1]</a> Liang, Ming, et al. "Multi-Task Multi-Sensor Fusion for 3D Object Detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  
<a id="2" href="#2ref">[2]</a> Liang, Ming, et al. "Deep continuous fusion for multi-sensor 3d object detection." Proceedings of the European Conference on Computer Vision (ECCV). 2018.  
<a id="3" href="#3ref">[3]</a> Wang, Shenlong, et al. "Deep parametric continuous convolutional neural networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.  
<a id="4" href="#4ref">[4]</a> Yang, Bin, Ming Liang, and Raquel Urtasun. "Hdnet: Exploiting hd maps for 3d object detection." Conference on Robot Learning. 2018.

