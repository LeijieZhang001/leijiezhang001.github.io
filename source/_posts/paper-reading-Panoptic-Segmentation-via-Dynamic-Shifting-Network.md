---
title: '[paper_reading]-"Panoptic Segmentation via Dynamic Shifting Network"'
date: 2020-12-03 10:00:27
updated: 2020-12-09 09:34:12
tags: ["paper reading", "Segmentation", "Deep Learning", "Autonomous Driving", "Point Cloud", "Instance Segmentation"]
categories:
- Segmentation
- Instance Segmentation
mathjax: true
---

　　实例分割一般与语义分割同时进行，其难点是后处理如何准确的聚类出实例目标。目标聚类可归纳出的策略或方法有：

- **提高语义分割的准确率**，在聚类的时候加入语义约束，能改善不同类别的欠分割，以及同类别的过分割；
- **Spatial Offset & Embedding Offset**，提升聚类空间下的目标的聚集性，如 {%post_link paper-reading-OccuSeg OccuSeg%}，{%post_link paper-reading-PointGroup PointGroup%}，{%post_link paper-reading-JSNet-JSIS3D JSNet, JSIS3D%}；
- **将聚类问题转换为每个点属于对应实例的概率问题**，网络下 End-to-End 来优化聚类效果，如 {%post_link paper-reading-Instance-Segmentation-by-Jointly-Optimizing-Spatial-Embeddings-and-Clustering-Bandwidth Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth%}；
- **引入聚类时的 Bandwidth**，不同尺寸的实例容忍不同程度的 Offset，如 {%post_link paper-reading-Instance-Segmentation-by-Jointly-Optimizing-Spatial-Embeddings-and-Clustering-Bandwidth Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth%}；

　　本文<a href="#1" id="1ref"><sup>[1]</sup></a>属于引入聚类时的 Bandwidth 策略，对于大目标，聚类时自适应选择更大的 offset 容忍度，同时可调节点经过 offset 迭代至中心点的次数来提高聚类准确率。

## 1.&ensp;Framework
<img src="framework.png" width="100%" height="100%" title="图 1. Framework">
　　如图 1. 所示，DS-Net 由 Cylinder-Convolution Backbone，Semantic Branch，Instance Branch 构成。{%post_link paper-reading-Cylinder3D Cylinder3D%} 以及 {%post_link paper-reading-Pillar-based-Object-Detection%} 已经较为详细得介绍了在 Cylinder 视野下的卷积过程，其相比 Spherical 和 Bird-eye 视野有一定的优势。Instance Branch 则由 Dynamic Shifting 以及 Consensus-driven Fusion 构成，以聚类实例目标。

## 2.&ensp;Instance Branch
　　Instance Branch 预测每个实例目标的点 \\(P\\in\\mathbb{R} ^ {M\\times 3}\\) 到实例中心 \\(C _ {gt}\\in\\mathbb{R} ^ {M\\times 3}\\) 的 offset \\(O\\in\\mathbb{R} ^ {M\\times 3}\\)。其 Loss 的基本形式为：
$$ L _ {ins} = \frac{1}{M}\sum _ {i=0} ^ M\Vert O[i]-(C _ {gt}[i]-P[i])\Vert \tag{1}$$
其中 \\(M\\) 是实例目标的点个数，预测的回归中心点 \\(O+P\\) 可用于实例目标的聚类，一般通过 Heuristic Clustering 方法，本文则提出 Dynamic Shifting 方法。  
　　自底向上的 Heuristic Clustering 方法有 Breadth First Search(BFS)，DBSCAN，HDBSCAN，Mean Shift 等。{%post_link paper-reading-PointGroup PointGroup%} 采用了 BFS 方法，对于点云这种密度不一样的数据形式，固定的搜索半径是不太合理的，小的搜索半径容易过分割，大的搜索半径则容易欠分割。DBSCAN/HDBSCAN 是 density-based 方法，所以与 BFS 一样，对密度不一致的点云聚类效果不好。Mean Shift 对密度不一致的点云聚类更加友好，但是固定的 bandwidth 也不是一个好的选择。

### 2.1.&ensp;Dynamic Shifting
　　对于待聚类的点云集 \\(X\\in\\mathbb{R} ^ {M\\times 3}\\)，预测的实例中心由回归的 offset \\(S\\in\\mathbf{R} ^ {M\\times 3}\\) 与点云坐标计算得到：
$$X\leftarrow X + \eta S\tag{2}$$
offset \\(S\\) 的计算可定义为 \\(S=f(X)-X\\)，其中 \\(f(\\cdot)\\) 为核函数。一种简单的平面核函数为：
$$f(X)=D ^ {-1} KX \tag{3}$$
其中 \\(K=(XX ^ T\\leq \\delta)\\) 表示点周围 bandwidth \\(\\delta\\) 区域的点集；\\(D=diag(K\\mathbf{1})\\) 表示 \\(\\delta\\) 内点集个数。  
　　为了自适应不同的 bandwidth，设计 \\(l\\) 个候选 bandwidth \\(L=\\{\\delta _ 1,\\delta _ 2,...,\\delta _ l\\}\\)。对候选 bandwidth 作权重化处理，权重通过 MLP，Softmax 得到 \\(\\sum _ {j=1} ^ l W[:,j] = \\mathbf{1}\\)。最终的核函数为：
$$\hat{f}(X) = \sum _ {j=1} ^ l W[:,j]\odot(D _ j ^ {-1}K _ j X) \tag{4}$$
其中 \\(K _ j=(XX ^ T\\leq\\delta _ j)\\), \\(D _ j=diag(K _ j\\mathbf{1})\\)。
<img src="DSM.png" width="60%" height="60%" title="图 2. Dynamic Shifting Module">
　　DSM 算法流程如图 2. 所示，种子点通过 offset 预测实例中心点迭代 \\(I\\) 次，第 \\(i\\) 次回归的 Loss 为：
$$l _ i=\frac{1}{M '}\sum _ {x=1} ^ {M '}\Vert X _ i[x]-C ' _ {gt}[x]\Vert _ 1 \tag{5}$$
总的 Loss 为：
$$L _ {ds} = \sum _ {i=1} ^ I w _ i l _ i \tag{6}$$
其中 \\(w _ i\\) 为权重，设为 1。


### 2.2.&ensp;Consensus-driven Fusion
　　最终实例的类别由点集合投票决定。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Hong, Fangzhou, et al. "LiDAR-based Panoptic Segmentation via Dynamic Shifting Network." arXiv preprint arXiv:2011.11964 (2020).


