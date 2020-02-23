---
title: MOT Multimodal Fusion
date: 2020-02-19 12:35:40
tags: ["MOT", "tracking", "autonomous driving"]
categories: MOT
mathjax: true
---

　　同一传感器的目标状态估计在{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}中已经有较详细的介绍。不同传感器在不同光照不同天气情况下，有不同的表现，比如相机在低光照下可靠性较差，而激光雷达能弥补这个缺陷。所以在目标状态估计中，多传感器融合非常重要，可以是**数据前融合，特征级融合，目标状态后融合**。本文关注目标状态后融合过程。

## 1.&ensp;问题描述
　　考虑两个传感器 \\(A,B\\) (传感器可为相机，激光雷达，毫米波雷达等)检测输出的(也可以是经过滤波的)多目标分别为：\\(A=\\{A _ i\\in\\mathbb{R}^D|i=1,...,M\\}\\)，\\(B=\\{B _ i\\in\\mathbb{R}^D|i=1,...,N\\}\\)，其中 \\(\\mathbb{R}^D\\) 表示目标状态的维数，如位置，速度，朝向，类别等。MOT 的多模态后融合问题即由此求解融合后结果 \\(C=\\{C _ i\\in\\mathbb{R}^D|i=1,...,L\\}\\)，该过程主要有三步：

1. 目标匹配/数据关联：从 \\(A,B\\) 中找出同一目标的两个多模态观测量，设匹配数为 \\(K\\)；
2. 目标状态的多模态融合：对匹配上的同一目标的两个多模态观测进行融合；
3. 整合目标，经过滤波输出最终结果，目标数目为 \\(L=M+N-K\\)；

## 2.&ensp;目标匹配
　　本质上与单传感器下目标状态估计中前后帧的数据关联问题一致，这里的关键步骤是：

1. 提取每个目标的特征向量：可以是位置，速度，角度，CNN特征层等；
2. 构建 cost function：对两个目标集合建立 Cost 矩阵；
3. 匈牙利算法找出最优匹配；

　　传统的 cost function 基本是向量的 Euclidean 距离或是 cosine 距离，<a href="#1" id="1ref">[1]</a> 提出了一种 Deep Affinity Network 来一次性解决两个目标集合的匹配问题。
<img src="affinity.png" width="70%" height="70%" title="图 1. Affinity Network">
　　如图 1. 所示，两个目标集 \\(A\\in\\mathbb{R}^{M\\times D}\\)，\\(B\\in\\mathbb{R}^{N\\times D}\\)，扩展到维度 \\(\\mathbb{R}^{M\\times N\\times D}\\)，相减后输入到网络中，预测出 affinity matrix，\\(C\\in\\mathbb{R}^{M\\times N}\\)，其中 \\(C _ {ij}=1\\) 表示匹配上同一目标，否则认为是两个目标。这里关键是 Loss 的设计，最简单的 Loss 为：
$$L(A,B)=\frac{1}{MN}\sum _ {i=1} ^ {M}\sum _ {j=1}^N |C _ {ij}-G _ {ij}| \tag{1}$$
其中 \\(G\\) 为亲和度矩阵的 groundtruth。实际对亲和度矩阵并没有 0-1 要求，最终是通过匈牙利算法找出匹配的，所以只要将同一目标的分数增大，不同目标的分数减小，最终即可选出匹配。由此设计 Loss：
$$L(A,B)=\sum _ {i,\,j;\,G _ {ij}=1} \left(\sum _ {k;\,G _ {ik}\neq 1}\mathrm{max}(0,C _ {ik}-C _ {ij}+m)+\sum _ {p;\,G _ {pj}\neq 1}\mathrm{max}(0,C _ {pj}-C _ {ij}+m)\right)\tag{2}$$
其中 \\(m\\) 控制正负样本的相对大小。式(2)更容易使网络收敛。

## 3.&ensp;多模态融合
　　当多传感器检测的同一目标匹配上后，需要融合出一个最终的观测。可以采用卡尔曼滤波的方法，{% post_link 卡尔曼滤波器在三维目标状态估计中的应用 卡尔曼滤波器在三维目标状态估计中的应用%}中的式(1)~(6)是时序下状态估计的迭代过程。对于多模态融合，虽然是同时获取的观测，但是融合过程类似，令测量矩阵 \\(H _ k\\) 为单位阵，所以可得卡尔曼增益：
$$K _ k=\frac{\bar{P} _ k}{\bar{P} _ k+R _ k} \tag{3}$$
由此计算后验概率<a href="#2" id="2ref"><sup>[2]</sup></a>：
$$\begin{align}
\hat{x} _ k &=\bar{x} _ k+K(z_k-\bar{x}) = \frac{\bar{P} _ kz _ k + \bar{x} _ kR _ k}{\bar{P} _ k+R _ k} \tag{4}\\
\hat{P} _ k &=(I-KH _ k)\bar{P} _ k =\frac{\bar{P} _ kR _ k}{\bar{P} _ k+R _ k}\tag{5}
\end{align}$$
对于多模态输入 \\(A,B\\)，令  \\(A = \\bar{x} _ k,\\sigma _ A^2 = \\bar{P} _ k\\)，\\(B=z _ k,\\sigma _ B^2 =R _ k\\)，可得多模态融合结果为：
$$\begin{align}
C &= \frac{\sigma _ A^2B+\sigma _ B^2A}{\sigma _ A^2+\sigma _ B^2}\\
\sigma _ C^2 &= \frac{\sigma _ A^2\sigma _ B^2}{\sigma _ A^2+\sigma _ B^2}\\
\tag{6}\end{align}$$
式(6)等价于：
$$\begin{align}
\sigma _ C^2 &= \frac{\sigma _ A^2\sigma _ B^2}{\sigma _ A^2+\sigma _ B^2}\\
C &= \sigma _ C^2\left(\frac{A}{\sigma _ A^2}+\frac{B}{\sigma _ B^2}\right)\\
\tag{7}\end{align}$$
这是 BCM<a href="#3" id="3ref"><sup>[3]</sup></a>！卡尔曼滤波器也是在贝叶斯概率模型下导出来的，可见两个高斯分布的同一状态的观测量，均可通过 BCM 进行融合。  
　　得到当前时刻多模态融合后的目标状态后，即可进一步作时序卡尔曼平滑获得最终估计的目标状态。  
　　另一种融合方法是在 JPDAF(Joint Probabilistic Data Association Filter)<a href="#4" id="4ref"><sup>[4]</sup></a>框架下作两次 PDA 融合<a href="#5" id="5ref"><sup>[5]</sup></a>，JPDAF 是另一种数据关联(目标匹配)的方法，这里不作展开。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Kuang, Hongwu, et al. "Multi-Modality Cascaded Fusion Technology for Autonomous Driving." arXiv preprint arXiv:2002.03138 (2020).  
<a id="2" href="#2ref">[2]</a> Fankhauser, Péter, et al. "Robot-centric elevation mapping with uncertainty estimates." Mobile Service Robotics. 2014. 433-440.  
<a id="3" href="#3ref">[3]</a> Tresp, Volker. "A Bayesian committee machine." Neural computation 12.11 (2000): 2719-2741.  
<a id="4" href="#4ref">[4]</a> Arya Senna Abdul Rachman, Arya. "3D-LIDAR Multi Object Tracking for Autonomous Driving: Multi-target Detection and Tracking under Urban Road Uncertainties." (2017).  
<a id="5" href="#5ref">[5]</a> JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset
