---
title: '[paper_reading]-"Fast and Furious"'
date: 2019-09-24 10:23:23
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: paper reading
mathjax: true
---

　　动态目标状态估计传统的做法是将其分解为目标检测，目标跟踪，目标运动预测三个子问题进行链式求解，这回导致上游模块的误差在下游模块中会传递并放大。考虑到跟踪与预测能帮助提升检测的性能，比如对于遮挡或远距离目标，跟踪与预测能减少检测的漏检(FN)；而误检(FP)则可通过时域相关信息消除，由此本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种联合 3D 检测，跟踪，运动预测的多任务网络。

## 1.&ensp;Model Architecture

### 1.1.&ensp;Data Representation
　　雷达坐标系下，每帧点云限定范围为\\(\(x_{min}, x_{max}, y_{min}, y_{max}, z_{min}, z_{max}\)\\)，那么在分辨率 \\(r = \(dx,dy,dz\)\\) 下进行栅格化，可得到体素 \\(\(C, H, W\) = \(\\frac{z_{max}-z_{min}}{dz}, \\frac{y_{max}-y_{min}}{dy}, \\frac{x_{max}-x_{min}}{dx}\)\\), 如果体素中有点云那么该体素值置为1，否则置为0，这样就得到了俯视图下的伪图像。  
　　此外将历史 \\(T-1\\) 帧点云先转换到当前本体坐标系(需要 ego motion 信息)，然后串成一起，就获得 \\(\(T, C, H, W\) \\) 维的模型数据输入表示。

### 1.2.&ensp;Model Formulation
　　实际输入网络的应该是 \\(\(N, T, C, H, W\) \\) 维的数据，首先需要经过一个 fusion 层将数据映射到 \\(\(N, C', H', W'\) \\) 维，然后用一个类似与 SSD 结构的 backbone+head 网络即可。

#### 1.2.1.&ensp;Fusion
<img src="fusion.png" width="80%" height="80%" title="图 1. Fusion 结构">
　　本文提出了两种融合方式：

- Early Fusion  
如图 1. 所示，直接在 T 维度上进行一维卷积，卷积 \\(kernel_ size = T\\)，由此得到 \\(\(N, C, H, W\) \\) 维的特征。
- Late Fusion  
如图 1. 所示，通过两次 3D 卷积将 \\(T=5\\) 变换到 \\(T=1\\)，\\(kernel size = \(3, 3, 3\)\\),由此也得到 \\(\(N, C, H, W\) \\) 维的特征。

相比 Early Fusion，Late fusion 有更深的特征提取。

#### 1.2.2.&ensp;Backbone+Head
<img src="head.png" width="80%" height="80%" title="图 2. Fusion 结构">
　　backbone 采用 VGG16 结构，图 1. 可见。  
　　head 采用类似 SSD 检测头的形式。anchor 也是有不同比例不同尺寸的矩形组成(另一种方法是，由于俯视图下同种类别的尺寸相似性，所以针对不同类别采用同一尺寸的 anchor 即够用)，角度回归则采用 \\(cos, sin\\) 形式。  
　　如图 2. 所示，检测头有两个分之分支，第一个输出预测的分类 score map(n 个预测的 score map 是共享的)，第二个输出 n 个预测的 3D 框编码信息。

### 1.3.&ensp;Decoding Tracklets
　　由于有检测及预测的信息，所以可用简单的方法解析出跟踪 ID。历史的预测框信息可认为是当前的跟踪框，所以就自然得在 MOT 问题里进行求解。这里可直接计算跟踪框(历史预测框)与当前检测框的 overlap 误差项，然后将重合度高的目标框标记为同一 ID 即可。

### 1.4.&ensp;Loss Function
　　总的误差由分类误差与回归误差构成：
$$\xi = \sum\left(\alpha \cdot \xi_{cla} + \sum_{i=t,t+1,...,t+n}\xi_{reg}^t\right)$$
这两项误差具体计算与传统的并无很大差别，此外作者还用了 OHEM 的策略，来平衡正负样本量巨大的差异。

## 2.&ensp;Experimental Evaluation
<img src="test.png" width="80%" height="80%" title="图 3. ablation study">
　　作者用了比 kitti 大的数据集，图 3. 所示，late fusion 比 early fusion 效果好一点，但是 late fusion 需要 3D 卷积。其它实验结果可参见文章。

<a id="1" href="#1ref">[1]</a> Luo, Wenjie, Bin Yang, and Raquel Urtasun. "Fast and furious: Real time end-to-end 3d detection, tracking and motion forecasting with a single convolutional net." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.
