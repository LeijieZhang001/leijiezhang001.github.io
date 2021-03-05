---
title: 4D Panoptic LiDAR Segmentation
date: 2021-03-02 09:51:00
updated: 2021-03-05 09:34:12
tags: ["paper reading", "Segmentation", "Deep Learning", "Autonomous Driving", "Point Cloud", "Instance Segmentation"]
categories:
- Segmentation
- Instance Segmentation
mathjax: true
---

　　4D 激光点云的全景分割任务是在 3D 基础上，引入时间维度，同时在时间和空间下作点云的语义分割以及实例分割，输出的是每个点的语义类别信息，以及时序实例 ID。本文<a href="#1" id="1ref"><sup>[1]</sup></a> 提出了一种简单 4D 全景分割框架。

## 1.&ensp;Framework
<img src="framework.png" width="70%" height="70%" title="图 1. Framework">
　　如图 1. 所示，将时序点云作累积，然后用 Encoder-Decoder 网络，输出量有：

- Semantic Map，每个点的类别；
- Objectness Map，目标中心点，或者靠近中心的某点；
- Point Embeddings，每个点的特征；
- Point Variance Map，每个点的位置方差；

有了这些信息后，就可以聚类出目标实例，方法类似 {%post_link paper-reading-Instance-Segmentation-by-Jointly-Optimizing-Spatial-Embeddings-and-Clustering-Bandwidth Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth%}，将 variance map 拓展到 Embedding 任意空间，应该是目前比较先进的策略了。  
　　假设目标实例的点云符合高斯分布，那么在已知中心点，或者属于该实例的某点的情况下(不用必须是中心点，这样能处理遮挡等情况)，就可以判断其它点属于该实例的概率。具体的，已知实例中心点 \\(\\mathbf{p} _ i\\)，以及对应的 Embedding 特征 \\(e _ i\\)，可以计算其它点 \\(p _ j\\) 是否属于该实例的概率：
$$\hat{p} _ {ij}=\frac{1}{(2\pi) ^ {D/2}|\Sigma _ i| ^ {\frac{1}{2}}}\mathrm{exp}\left(-\frac{1}{2}(e _ i-e _ j)^T\Sigma _ i^ {-1}(e _ i-e _ j)\right) \tag{1}$$
其中 \\(\\Sigma _ i\\) 是通过点 \\(p _ i\\) 预测的 \\(\\sigma _ i\\) 所构建的对角矩阵。值得注意的是，Embedding 特征串联了空间和时序量，这样更有利于聚类，同时预测对应的 variance map。  
　　ID 则是通过时序下作实例的数据关联得到的。

## 2.&ensp;Loss
　　网络输出都是点级别的。总的 Loss 为：
$$L=L _ {class}+L _ {obj}+L _ {ins}+L _ {var} \tag{2}$$
具体的：

- Semantic Segmentation  
用 Cross-entropy 分类损失函数，并且通过采样的方法来解决类别不平衡问题。
- Point Centerness  
计算每个点与其实例中心点，或者实例点云重心点的距离，归一化后作为中心点得分损失函数：
$$L _ {obj} = \sum _ {i=1} ^ N(\hat{o} _ i-o _ i) ^ 2,\;\;\;\hat{o} _ i,o _ i\in[0,1]\tag{3}$$
- Instance Probability  
计算每个实例中每个点属于该实例的概率，回归其值到 1:
$$L _ {ins}=\sum _ {j=1} ^ K\sum _ {i=1} ^ N(\hat{p} _ {ij}-p _ {ij}) ^ 2,\;\;\;p _ {ij} = 1,\;\mathrm{if}\; p _ i\in I _ j,\;\mathrm{else}\; 0\tag{4}$$
- Variance Smooth  
类似 {%post_link paper-reading-Instance-Segmentation-by-Jointly-Optimizing-Spatial-Embeddings-and-Clustering-Bandwidth Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth%}，作实例内每个点 Variance 的一致性约束：
$$L _ {var} = \frac{1}{|I _ j|}\sum _ {i\in I _ j}\Vert \sigma _ i-\sigma _ j\Vert ^ 2\tag{5}$$

## 3.&ensp;Measuring Performance
To be continued...

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Aygün, Mehmet, et al. "4D Panoptic LiDAR Segmentation." arXiv preprint arXiv:2102.12472 (2021).

