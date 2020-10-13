---
title: '[paper_reading]-"JSNet, JSIS3D"'
date: 2020-10-10 17:29:45
updated: 2020-10-15 09:34:12
tags: ["paper reading", "Segmentation", "Deep Learning", "Autonomous Driving", "Point Cloud", "Instance Segmentation"]
categories:
- Segmentation
- Instance Segmentation
mathjax: true
---

　　{%post_link paper-reading-PointGroup PointGroup%} 通过预测每个点与对应 instance 重心的 offset，然后在三维物理坐标系下作 instance 聚类。<a href="#1" id="1ref">[1]</a> 也是这种方案。另一种思路，是通过 Metric Learning 技术，预测每个点的高维特征(Embedding Space)，然后作 instance-level 聚类。本文介绍的 JSNet<a href="#2" id="2ref"><sup>[2]</sup></a> 以及 JSIS3D<a href="#3" id="3ref"><sup>[3]</sup></a> 就是采用的这种方式。

## 1.&ensp;JSNet
<img src="JSNet.png" width="90%" height="90%" title="图 1. JSNet">
　　如图 1. 所示，整个网络共享的 Backbone 只有点云特征的 Encode 阶段，两个分支分别作 Decode 并通过 PCFF 模块，最终输出用于 Instance-Seg 的特征 \\(F _ {IS}\\in\\mathbb{R} ^ {N _ a\\times 128}\\)，以及用于 Semantic-Seg 的特征 \\(F _ {SS}\\in\\mathbb{R} ^ {N _ a\\times 128}\\)。这一阶段完全可以用其它 Voxel 或 Point 网络代替。然后通过 JISS 模块进行两个分支的特征融合，最终输出点云类别，以及用于点云 Instance 聚类的特征 Embedding。最后采用 Mean-Shift 聚类方法即可根据 Embedding 作 Instance 聚类。

### 1.1.&ensp;PCFF
　　PCFF 类似图像 2D 卷积中上采样特征融合模块，如图 1.a 所示，目的是为了融合不同尺度的点云特征。PCFF 及之前的网络均可用其它点云特征网络代替。

### 1.2.&ensp;JISS
　　JISS 模块目的是将 Instance-Seg 和 Semantic-Seg 两个任务的特征作充分的融合。Semantic-Seg 一般比 Instance-Seg 更底层，所以相同深度的网络，理论上能学到更加抽象(高层)的特征，所以如图 1.c 所示，先将 \\(F _ {SS}\\) 特征融入 \\(F _ {IS}\\) 特征中，然后在 Instance-Seg 分支作进一步特征提取后，再将特征返回来与 \\(F _ {SS}\\) 特征作融合。此外，每个分支还引入了 Self-Attention 模块，通过 Sigmoid 操作实现。  
　　最终输出的是每个点的类别分数 \\(P _ {SSI}\\in\\mathbb{R} ^ {N _ a\\times C}\\)，以及用于 Instance 聚类的点云特征 \\(E _ {ISS}\\in\\mathbb{R} ^ {N _ a\\times K}\\)。

### 1.3.&ensp;Loss
　　Loss 由 Semantic-Seg 以及 Instance-Seg 两个任务组成：
$$\mathcal{L}=\mathcal{L} _ {sem}+\mathcal{L} _ {ins}\tag{1}$$
其中语义分割的 Loss 项 \\(\\mathcal{L} _ {sem}\\) 为传统的分类 Loss。\\(\\mathcal{L} _ {ins}\\) 则要求能区分不同 Instance 的点云 Embedding 特征，但是又要保证同一 Instance 的点云 Embedding 特征的相似性，设计为：
$$\begin{align}
\mathcal{L} _ {ins} &=\mathcal{L} _ {pull}+\mathcal{L} _ {push}\\
&= \frac{1}{M}\sum _ {m=1} ^ M\frac{1}{N _ m}\sum _ {n=1} ^ {N _ m}\left[\Vert\mu _ m-e _ n\Vert _ 1-\delta _ v\right] _ + ^ 2 + \frac{1}{M(M-1)}\mathop{\sum _ {i=1} ^ M\sum _ {j=1} ^ M} \limits _ {i\neq j}\left[2\delta _ d-\Vert\mu _ i-\mu _ j\Vert _ 1\right] _ + ^ 2
\tag{2}
\end{align}$$
其中 \\([x] _ +=\\mathrm{max}(0,x)\\)，\\(|| \\cdot || _ 1\\) 为 L1 距离，\\(\\delta _ v,\\delta _ d\\) 分别为 \\(\\mathcal{L} _ {pull},\\mathcal{L} _ {push}\\) 的幅度。

## 2.&ensp;JSIS3D
<img src="JSIS.png" width="90%" height="90%" title="图 2. JSIS">
　　如图 2. 所示，JSIS3D 由 MT-PNet 网络和 MV-CRF 构成。MV-CRF 是基于 MT-PNet 网络预测的 Semantic Label 和 Embeddings 作基于条件随机场的 instance 聚类，效果比直接对 Embeddings 作聚类要好，这里只讨论 MT-PNet 网络。

### 2.1.&ensp;MT-PNet
<img src="MT-PNet.png" width="90%" height="90%" title="图 3. MT-PNet">
　　如图 3. 所示，网络由基本的 PointNet 构成，最终预测的也是每个点的类别以及用于聚类的 Embedding。所以输出方案是与 JSNet 是一样的。Loss 项中的 Embedding(ins) 预测项加入了正则化：
$$\begin{align}
\mathcal{L} _ {ins} &=\alpha\mathcal{L} _ {pull}+\beta\mathcal{L} _ {push}+\gamma\mathcal{L} _ {reg}\\
&= \frac{\alpha}{M}\sum _ {m=1} ^ M\frac{1}{N _ m}\sum _ {n=1} ^ {N _ m}\left[\Vert\mu _ m-e _ n\Vert _ 2-\delta _ v\right] _ + ^ 2 + \frac{\beta}{M(M-1)}\mathop{\sum _ {i=1} ^ M\sum _ {j=1} ^ M} \limits _ {i\neq j}\left[2\delta _ d-\Vert\mu _ i-\mu _ j\Vert _ 2\right] _ + ^ 2 + \frac{\gamma}{M}\sum _ {m=1} ^ M \Vert \mu _ m\Vert _ 2
\tag{3}
\end{align}$$
其中 \\(M\\) 为 instance 数量，\\(N _ m\\) 为对应 instance 内点的个数，\\(e _ n\\) 为点的 Embedding，\\(\\mu _ m\\) 表示第 \\(m\\) 个 instance 内点的平均 Embedding。设计 \\(\\sigma _ d > 2\\sigma _ v,\\alpha=\\beta=1,\\gamma=0.001\\)，可以实现同一个 instance 内点的 Embedding 相近，不同 instance 的平均 Embedding 距离较远，并且正则化使得平均 Embedding 接近 0。

### 2.2.&ensp;Experiments
<img src="res.png" width="90%" height="90%" title="图 4. Mean-Shift VS. MV-CRF">
　　如图所示，用 MV-CRF 代替 Means-Shift 聚类，对于大物体，提升效果比较明显，但是小物体，精度会下降。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> F. Zhang, C. Guan, J. Fang, S. Bai, R. Yang, P. Torr, and V. Prisacariu, “Instance segmentation of lidar point clouds,” in ICRA, 2020  
<a id="2" href="#2ref">[2]</a> L. Zhao and W. Tao, “JSNet: Joint instance and semantic segmentation of 3D point clouds,” in AAAI, 2020.  
<a id="3" href="#3ref">[3]</a> Pham, Quang Hieu , et al. "JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds With Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) IEEE, 2020.
