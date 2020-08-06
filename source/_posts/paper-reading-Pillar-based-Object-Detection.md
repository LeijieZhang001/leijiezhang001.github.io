---
title: '[paper_reading]-"Pillar-based Object Detection"'
date: 2020-08-04 11:42:08
updated: 2020-08-06 09:19:12
tags: ["paper reading", "3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories:
- 3D Detection
mathjax: true
---

　　{%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds MVF%} 在俯视图点云特征的基础上，融合了点云的前视图特征，由此解决点云在远处比较稀疏，以及行人等狭长型目标特征信息较少的问题。本文<a href="#1" id="1ref"><sup>[1]</sup></a>基于 {%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds MVF%} 作了三部分的改进：

1. 检测头改为 Anchor-Free 的形式，本文称之为 Pillar-based，其实就是图像中对应的像素点；
2. 前视图用 Cylindrical View 代替 Spherical View，解决目标高度失真的问题；
3. 两个视图的栅格特征反投影回点特征作融合时，采用双线性插值的形式，避免量化误差的影响。

## 1.&ensp;Framework
<img src="framework.png" width="90%" height="90%" title="图 1. Framework">
　　如图 1. 所示，点云分别投影到 BEV(Brids-Eye)，CYV(Cylindrical) 视角，然后作类似图像卷积的 2D 卷积操作以提取特征，并将特征反投影回点作融合(与 {%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds MVF%} 一致)，接着将点云特征再次投影到 BEV 下，最后作 Anchor-Free 的分类与回归任务。  
　　具体的，设 \\(N\\) 个点的点云 \\(P=\\{p _ i\\} _ {i=0} ^ {N-1}\\subseteq\\mathbb{R} ^ 3\\)，对应的特征向量为 \\(F = \\{f _ i\\} _ {i=0} ^ {N-1}\\subseteq\\mathbb{R} ^ K\\)。令 \\(F _ V(p _ i)\\) 返回点 \\(p _ i\\) 对应的栅格柱子 \\(v _ j\\) 的索引 \\(j\\)；\\(F _ P(v _ j)\\) 则返回栅格柱子 \\(v _ j\\) 对应的点集。对每个柱子进行特征整合，一般采用类似 PointNet(PN) 的方法：
$$f _ j ^{pillar} = \mathrm{PN} (\{f _ i|\forall p _ i\in F _ P(v _ j)\}) \tag{1}$$
pillar 级别的特征经过 CNN \\(\\phi\\) 后得到进一步的 pillar 级别特征：\\(\\varphi=\\phi(f ^ {pillar})\\)。然后分别对 BEV，CYV 作 pillar-to-point 的特征投影变换：
$$f _ i^{point}=f _ j^{pillar}\;\mathrm{and}\;\varphi _ i^{point} = \varphi _ j^{pillar},\;\mathrm{where}\; j = F _ V(p _ i) \tag{2}$$
最后的检测头是应用已经较为广泛的 Anchor-Free 形式。

## 2.&ensp;Cylindrical View
<img src="proj.png" width="60%" height="60%" title="图 2. Projection">
　　{%post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds MVF%} 采用 Spherical 投影方式，对于点 \\(p _ i=(x _ i, y _ i, z _ i)\\)，其球坐标 \\(\\varphi _ i,\\theta _ i,d _ i\\) 为：
$$\left\{\begin{array}{l}
\varphi _ i &= \mathrm{arctan}\frac{y _ i}{x _ i}\\
\theta _ i &= \mathrm{arccos}\frac{z _ i}{d _ i}\\
d _ i &= \sqrt{x _ i ^ 2+y _ i ^ 2+z _ i^2}
\end{array}\tag{3}\right.$$
如图 2. 所示，球坐标系下目标高度的形变比较严重，本文采用柱坐标系，其柱坐标 \\(\\rho _ i,\\varphi _ i,z _ i\\) 表示为：
$$\left\{\begin{array}{l}
\rho _ i &=\sqrt{x _ i ^ 2+y _ i^2}\\
\varphi _ i &= \mathrm{arctan}\frac{y _ i}{x _ i}\\
z _ i &= z _ i
\end{array}\tag{4}\right.$$
　　在此视角下作 pillar-level 的特征提取，与俯视图视角一样，只不过作卷积的时候，是环状卷积。具体实现方式是，将柱坐标系下的 pillar 展开，然后边缘补对应展开处另一边的 pillar 值，最后作传统的 2D 卷积即可。

## 3.&ensp;Pillar-based Prediction
　　这里所谓的 Pillar-based 预测，本质上就是图像中常说的 Anchor-Free 的 Pixel-Level 的检测方法。最后特征图上的每个点预测类别概率，以及 3D 框属性 \\(\\Delta _ x,\\Delta _ y,\\Delta _ z,\\Delta _ l,\\Delta _ w,\\Delta _ h,\\theta ^ p\\)。这里不作展开。

## 4.&ensp;Bilinear Interpolation
<img src="bilinear.png" width="60%" height="60%" title="图 3. Bilinear">
　　将 Pillar-Level 提取的特征反投影到 Point-Level 的特征时，需要进行插值处理。如图 3. 所示，传统的方式是最近邻插值，这种方式会引入量化误差，使得点投影反投影后的空间坐标不一致，产生的影响是同一 Pillar 内的点特征都是一样了。本文采用双线性插值的方法，使得 Point-Pillar-Point 的空间坐标一致，这样保证了 Pillar 内点特征的原始精度。该思想还是非常有借鉴意义的，实验效果提升也比较明显。

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Wang, Yue, et al. "Pillar-based Object Detection for Autonomous Driving." arXiv preprint arXiv:2007.10323 (2020).  
