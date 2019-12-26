---
title: Object Registration with Point Cloud
date: 2019-12-25 09:13:19
tags: ["MOT", "Point Cloud", "tracking", "ICP"]
categories: MOT
mathjax: true
---

　　{% post_link ADH-Tracker ADH Tracker%} 通过 ADH 方法有效得在两目标点云的 T 变换的解空间中搜索出高概率解集，并用简单的运动模型，在贝叶斯概率框架下进行目标状态(位置，速度)的估计。这其中关键的环节还是两目标点云之间变换关系 \\((R,T)\\) 的求解，即 Object Registration。  
　　求解两点云之间的位姿关系，传统的做法是 ICP。以 ICP 为代表的方法大多数都是迭代法，本文介绍两种 learning-based 点云注册方法。

## 1.&ensp;Deep Closet Point<a href="#1" id="1ref"><sup>[1]</sup></a>
### 1.1.&ensp;ICP 描述
　　假设两个点云集：\\(\\mathcal{X}=\\{x _ 1,...,x _ i,...,x _ N\\}\\in\\mathbb{R}^3\\)，\\(\\mathcal{Y}=\\{y _ 1,...,y _ j,...y _ M\\}\\in\\mathbb{R}^3\\)。两个点集之间的变换为 \\(R,t\\)，定义点集匹配的误差函数：
$$ E(R,t) = \frac{1}{N}\sum_i^N\Vert Rx_i+t-y _ {m(x_i)}\Vert \tag{1}$$
其中 \\(y_{m(x_i)}\\) 为 \\(x_i\\) 经过变换后匹配上的最近点，即：
$$ m(x_i,\mathcal{Y}) = \mathop{\arg\min}_j\Vert Rx_i+t-y_j\Vert \tag{2}$$
定义点云重心：\\(\\bar{x}=\\frac{1}{N}\\sum _ {i=1}^Nx _ i\\)，\\(\\bar{y}=\\frac{1}{M}\\sum _ {j=1}^Ny _ j\\)。计算 Cross-covariance 矩阵：
$$ H = \sum_{i=1}^N(x_i-\bar{x})(y_i-\bar{y}) \tag{3}$$
\\(R,t\\) 变换可通过 \\(H=USV^T\\) 最小化误差函数 \\(E(R,t)\\) 实现：
$$\left\{\begin{array}{l}
R= VU^T\\
t= -R\bar{x}+\bar{y}
\end{array}\tag{4}\right.$$
ICP 算法就是迭代得求解式(2)与式(1)的过程。

### 1.2.&ensp;网络结构
<img src="DCP.png" width="80%" height="80%" title="图 1. DCP">
　　如图 1. 所示，DCP 网络结构由三部分组成：

- **Embedding Module**  
特征提取层，可以用 PointNet，也可以用 DGCNN 网络({% post_link PointCloud-Feature-Extraction PointCloud Feature Extraction%})，DGCNN 能更有效的提取局部特征。
- **Transformer**  
该模块基于 Attention 机制，详情可参考<a href="#3" id="3ref">[3]</a><a href="#4" id="4ref">[4]</a>。
- **Head**  
该模块用于预测 \\((R,t)\\)，可以简单的用 MLP 回归，也可以用 SVD 层来预测，因为 Transformer 会输出 \\(x_i\\) 在 \\(\\mathcal{Y}\\) 中的匹配点。

### 1.3.&ensp;Loss
　　Loss 比较简单，也是基于有监督的学习：
$$ Loss = \Vert R^TR_g-I\Vert ^2 + \Vert t-t_g\Vert ^2 + \lambda \Vert\theta\Vert ^2$$

## 2.&ensp;AlignNet-3D<a href="#2" id="2ref"><sup>[2]</sup></a>
### 2.1.&ensp;网络结构
<img src="AlignNet.png" width="60%" height="60%" title="图 2. AlignNet">
　　如图 2. 所示，AlignNet 由两个网络组成：

- **CanonicalNet**  
CanonicalNet 作用是预测点集目标3D框的中心点坐标系，从而将点集坐标转换到中心点坐标系。预测点集目标3D框的中心点坐标系通过 coarse-to-fine 方式实现，stage1(T-CoarseNet) 只粗略预测中心点的位置信息，stage2(T-FineNet) 预测中心点位置相对 Stage1 的残差，以及中心点坐标系的旋转量。参考以前的方法，旋转量通过角度区域分类＋残差实现。通过该网络，每个点集的坐标均在各自目标框中心点坐标系下，能直观的反应目标的形状。
- **Head**  
Head(stage3) 则将两个点集特征聚合，预测各中心点坐标系下两个点集的相对位姿。  
设点集 \\(s_1\\) 经过 CanonicalNet 预测的变换为 \\(T_1\\)，\\(s_2\\) 对应的变换为 \\(T_2\\)，stage3 预测的两者的变换为 \\(T_f\\)，那么最终得到的两个点集的变换为 \\(T_1T_fT_2^{-1}\\)。

### 2.2.&ensp;Loss
　　stage1 预测了 translation，stage2/stage3 预测了 translation 和 rotation，总的 Loss 为：
$$\begin{align}
L &= L_{trans,overall}+\lambda_2\cdot L_{angle,overall}\\
  &= \lambda_1(L_{trans,s1}+L_{trans,s2}) + L_{trans,s3} + \lambda_2(\lambda_1L_{angle,s2}+L_{angle,s3})
\end{align}$$
stage1/stage2 预测的目标框中心点坐标系(包括中心点坐标及目标框的朝向)真值由点云所构成的目标框提供。

### 2.3.&ensp;不足点
　　这种级联式的方法，思想是非常好的，将两个点集的相对位姿分解为两大部来求解，即先将点集转换到中心点坐标系，然后再求解点集剩下位姿残差，coarse-to-fine，能较好回归且收敛。  
　　但是存在一些问题。我们假设两个点集作为同一刚性目标，其3D框没有偏差(标注非常准)，那么 CanonicalNet 出来结果，已经可以作为相对位姿结果。但是标注肯定会有抖动(除非是生成的数据)，可以认为是高斯分布，以及获取点云的传感器的测量噪音，这样的话，看起来 stage3 就是只用来拟合这种均值为 0 的高斯分布了。  
　　所以本方法对生成的数据与真实的数据，存在一定的偏差，因为目标框真值的抖动分布不一致。这样的话在生成的数据上训练的网络，直接迁移到真实数据中，可能性能会下降比较明显，反之可能还好。

## 3.&ensp;参考文献

<a id="1" href="#1ref">[1]</a> Wang, Yue, and Justin M. Solomon. "Deep Closest Point: Learning Representations for Point Cloud Registration." arXiv preprint arXiv:1905.03304 (2019).  
<a id="2" href="#2ref">[2]</a> Groß, Johannes, Aljoša Ošep, and Bastian Leibe. "AlignNet-3D: Fast Point Cloud Registration of Partially Observed Objects." 2019 International Conference on 3D Vision (3DV). IEEE, 2019.  
<a id="3" href="#3ref">[3]</a> Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.  
<a id="4" href="#4ref">[4]</a> https://zhuanlan.zhihu.com/p/48508221


