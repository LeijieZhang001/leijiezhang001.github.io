---
title: '[paper_reading]-"P2B"'
date: 2020-10-14 09:42:39
updated: 2020-10-16 17:29:12
tags: ["paper reading", "Deep Learning", "Autonomous Driving", "Point Cloud", "MOT", "Tracking"]
categories:
- MOT
mathjax: true
---

　　3D 目标状态估计中，对目标的跟踪测量非常重要，所谓跟踪测量，指的是给定前后目标框或者目标点云，计算目标的 R,t 的过程，由此可得到目标位置及速度的观测。**一般的测量量有：目标框中心点距离，目标点云重心距离，目标点云 ICP 结果**。**在目标点云观测较为完备的情况下，理论上类 ICP 方法的结果是最为准确的，但是在点云较少的情况下，通过基于深度学习脑补预测出来的目标框中心点可能会更靠谱**。ICP 一个比较大的问题是速度较慢，{%post_link ADH-Tracker ADH-Tracker%} 中的测量量本质上类似 ICP，但是其通过退火算法将 T 的搜索空间进行压缩，获得了极大的效率提升，但是对旋转量的估计还是比较棘手。  
　　将跟踪测量问题往外扩，就是图像领域常说的单目标跟踪：给定上一帧目标位置，找到当前帧目标的位置。点云场景中，一般不会作目标的单目标跟踪，而是直接采用目标运动模型预测当前帧位置以及目标检测当前帧位置，所以目标的定位精度基本完全靠目标检测结果以及卡尔曼平滑结果。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出一种基于深度学习的点云单目标跟踪方法。类似图像中的单目标跟踪，输入为上一帧目标或目标的模型，以及当前帧搜索区域，输出为当前帧目标的位置。进一步讲，**套用目标框中心，点云重心以及类 ICP 计算后，该模块结果可作为另一种更准确（相比直接采用检测框）的跟踪测量量**。

## 1.&ensp;Framework
<img src="framework.png" width="90%" height="90%" title="图 1. Framework">
　　如图 1. 所示，P2B 首先将目标(目标模型)特征与搜索区域的特征作融合，用来预测潜在的目标中心点，然后作端到端的目标 proposal 以及 verification。其网络主要由两部分组成：1. Target-specific feature augmentation; 2. 3D target proposal and verification。

### 1.1.&ensp;Target-specific feature augmentation
　　目标点云为 \\(P _ {tmp}\\in\\mathbb{R} ^ {N _ 1\\times 3}\\)，搜索区域的点云为 \\(P _ {sea}\\in\\mathbb{R} ^ {N _ 2\\times 3}\\)。经过 PointNet++ 采样及提取特征后，得到目标点云的种子点集 \\(Q=\\{q _ i\\} _ {i=1} ^ {M _ 1}\\)，搜索区域的种子点集 \\(R=\\{r _ j\\} _ {j=1} ^ {M _ 2}\\)，每个点特征向量为 \\([x;f]\\in\\mathbb{R} ^ {3+d _ 1}\\)。计算 \\(Q,R\\) 的相似度：
$$Sim _ {j,i} =\frac{f _ {q _ i} ^ T\cdot f _ {r _ j}}{\Vert f _ {q _ i}\Vert _ 2\cdot\Vert f _ {r _ j} \Vert _ 2},\;\forall q _ i\in Q,\;r _ j\in R \tag{1}$$
<img src="feat.png" width="90%" height="90%" title="图 2. Feature Aggregation">
　　得到的相似性特征图 \\(Sim\\) 与目标种子点集 \\(Q\\) 的顺序有关，如图 2. 所示，Target-Specific Feature Augmentation 模块目的就是消除 \\(Q\\) 顺序的影响。其基本思想也是通过对称操作将 \\(Q\\) 所在的 \\(M _ 1\\) 维度进行压缩。具体的，将 \\(Sim \\in\\mathbb{R} ^ {M _ 2\\times M _ 1}\\) 变换为 \\(M _ 2\\times (3+d _ 2)\\) 维度的特征。图 2 采用了 \\(Q\\) 特征，也可以采用其它方式，实验表明这种方式最好。

### 1.2.&ensp;3D target proposal and verification
　　有了搜索区域每个种子点的特征后，可以基于此作候选目标框的预测。接下来分为两个分支，一个分支输出 \\(M _ 2\\times 1\\)，表示每个种子点作为目标中心的概率分数；另一个分支采用 VoteNet<a href="#2" id="2ref"><sup>[2]</sup></a> ，输出相同维度及尺寸的特征，表示种子点与中心点的**坐标及特征残差**。VoteNet 中种子点的坐标残差用目标中心点与该种子点的距离来监督，而特征残差没有监督 Loss。这与 Instance-Seg 里面的套路非常相似，特征残差其实可以加上类内 Pull Loss 项，详见 {%post_link paper-reading-JSNet-JSIS3D JSNet-JSIS3D%}。  
　　对每个种子点作残差补偿后，类似 Instance-Seg 中的套路，可采用 Ball-Query 作目标的点集聚类，然后对聚类后的点集归一化并用 PointNet 即可预测目标的分数以及目标的 3D 属性，这里不作展开，详见<a href="#2" id="2ref">[2]</a>。  
　　Loss 项由残差回归，是否是 proposal 的概率，proposal 质量分数及目标 3D 属性回归等组成：
$$L = L _ {reg} +\gamma _ 1L _ {cla} + \gamma _ 2L _ {prop} + \gamma _ 3L _ {box} \tag{2}$$
其中 \\(\\gamma\\) 分别设计为 \\(0.2, 1.5, 0.2\\)。

## 2.&ensp;Workflow
<img src="workflow.png" width="60%" height="60%" title="图 3. Workflow">
　　P2B 的整个算法流程如图 3. 所示，需要注意的是，最后得到的 \\(K\\) 个 Proposal 只需要取分数最高的即可。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Qi, Haozhe, et al. "P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  
<a id="2" href="#2ref">[2]</a> Qi, Charles R., et al. "Deep hough voting for 3d object detection in point clouds." Proceedings of the IEEE International Conference on Computer Vision. 2019.

