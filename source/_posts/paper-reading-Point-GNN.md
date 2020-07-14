---
title: '[paper_reading]-"Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud"'
date: 2020-07-10 09:22:07
updated: 2020-07-10 09:19:12
tags: ["Deep Learning", "Autonomous Driving", "Point Cloud", "3D Detection"]
categories:
- 3D Detection
mathjax: true
---

　　本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种基于图网络来提取点云特征的方法，理论上可在不损失原始信息的情况下，高效的学习点云特征，其在点云 3D 检测任务中效果提升明显。

## 1.&ensp;Different Point Cloud Representations
<img src="repr.png" width="65%" height="65%" title="图 1. Point Cloud Representations">
　　如图 1. 所示，目前点云表示方式以及对应的特征学习方式有三种：Grids，栅格化后类似图像 2D/3D 卷积的形式；Sets，以 PointNet 为代表的最近邻查找周围点并学习的形式；Graph，将无序点集转换为图模型，特征信息通过点云顶点传递学习的形式。Grids 及 Sets 形式我们已经比较熟悉了，Graph 则查询效率比 Sets 高，特征提取能力又比 Grids 高。Graph 的建图时间复杂度为 \\(\\mathcal{O}(cN)\\)，领域查询复杂度则为 \\(\\mathcal{O}(1)\\)，Sets 中的 KNN 建树及查询复杂度可见 {%post_link PointCloud-Feature-Extraction PointCloud Feature Extraction%}。当然 KNN 式的领域查询方式可以用近似 \\(\\mathcal{O}(1)\\) 方法实现，但是会影响特征学习的准确度。

## 2.&ensp;Framework
<img src="framework.png" width="95%" height="95%" title="图 2. Framework of Point-GNN">
　　如图 2. 所示，基于 Graph 的 3D 点云检测，首先对点云作 Graph Construction，然后用 GNN 来学习每个顶点的特征，接着对每个顶点预测目标框，最后作目标框的整合和 NMS。

### 2.1.&ensp;Graph Construction
　　设点云集：\\(P=\\{p _ 1,...,p _ N\\}\\)，其中 \\(p _ i=(x _ i, s _ i)\\) 分别表示坐标 \\(x _ i\\in\\mathbb{R} ^ 3\\)，以及该点反射率，领域点相对位置等信息 \\(s _ i\\in\\mathbb{R} ^ k\\)。对该点集建图 \\(G=(P,E)\\)，将距离小于一定阈值的两个点进行连接，即：
$$E = \{(p _ i, p _ j)|\Vert x _ i-x _ j\Vert _ 2 < r\} \tag{1}$$
这种建图方式是 Fixed Radius Near-Neighbors 问题，可在 \\(\\mathcal{O}(cN)\\) 时间复杂度下解决，其中 \\(c\\) 为最大连接数。  
　　建图完成后，要对每个点信息状态 \\(s _ i\\) 作初始化。这里采用类似 Sets 的特征提取方式，即将该点的反射率，以及与领域内点的相对位置，串联成特征向量，然后用 MLP 作空间变换，最后在点维度上作 Max Pooling，即可得到初始化的该点特征状态量 \\(s _ i\\)。

### 2.2.&ensp;Graph Neural Network with Auto-Registration
　　传统的图神经网络，通过边迭代每个顶点的特征。在 \\((t+1) ^ {th}\\) 迭代时：
$$\begin{align}
 v _ i ^ {t+1} &= g ^ t\left(\rho\left(\{e _ {ij} ^ t|(i,j)\in E\}\right), v _ i ^ t\right) \\
e _ {ij} ^ t &= f ^ t(v _ i ^ t, v _ j ^ t) \tag{2}
\end{align}$$
其中 \\(e ^ t,v ^ t\\) 分别是边和顶点特征，\\(f ^ t(\\cdot)\\) 计算两个顶点之间边的特征，\\(\\rho(\\cdot)\\) 将与该点连接的边特征整合，得到该点特征增量，\\(g ^ t(\\cdot)\\) 将该点特征增量与原特征进行整合得到本次迭代后该点的最终特征。  
　　对于边特征，一种设计方式为，描述领域特征对该点位置的作用力，重写式 (2)：
$$s _ i ^ {t+1} = g ^ t\left(\rho\left(\{f ^ t(x _ j-x _ i,s _ j^t)|(i,j)\in E\}\right), s _ i ^ t\right) \tag{3}$$
这样就得到了图神经网络的迭代模型。此外，本文还指出，由于边特征对领域点的距离较为敏感，所以作者提出对相对位置作自动补偿，实验表明其实意义不大：
$$\begin{align}
\Delta x _ i ^ t &= h ^ t(s _ i^t) \\
s _ i ^ {t+1} &= g ^ t\left(\rho\left(\{f ^ t(x _ j-x _ i+\Delta x _ i ^ t,s _ j^t)|(i,j)\in E\}\right), s _ i ^ t\right) \tag{4}
\end{align}$$
　　具体的，\\(f ^ t(\\cdot),g ^ t(\\cdot), h ^ t(\\cdot)\\) 可用 MLP 来建模，\\(\\rho(\\cdot)\\) 则采用 Max 操作：
$$\begin{align}
\Delta x _ i ^ t &= MLP _ h ^ t(s _ i^t) \\
e _ {ij} ^ t &= MLP _ f ^ t([x _ j - x _ i + \Delta x _ i ^ t, s _ j ^ t]) \\
s _ i ^ {t+1} &= MLP _ g ^ t\left(MAX(\{e _ {ij}|(i,j)\in E\})\right)+ s _ i ^ t \tag{5}
\end{align}$$

### 2.3.&ensp;Loss
　　为了作 3D 检测的任务，网络头输出为每个顶点的类别，目标框中心的 offset，以及目标框的尺寸，朝向。这与传统的基于 Ancho-Free 的 3D 目标检测基本一致，这里不做展开。

### 2.4.&ensp;Box Merging and Scoring
　　本方法的 3D 检测需要作 NMS 后处理，由于分类的 Score 不能反应目标框的 Uncertainty，所以基于 Score 的 NMS 是不合理的。这个问题在 2D 检测中也有比较多的研究，比如采用预测 IoU 值的方式来作权重。本文则认为遮挡信息能作为 NMS 操作的指导，由此定义了遮挡值的计算方式。但是实验显示，其实提升并不明显，所以这里不做具体展开。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Shi, Weijing, and Raj Rajkumar. "Point-gnn: Graph neural network for 3d object detection in a point cloud." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
