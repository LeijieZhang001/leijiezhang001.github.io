---
title: A Comprehensive Survey on Point Cloud Registration
date: 2021-03-10 09:24:45
updated: 2021-02-28 09:34:12
tags: ["paper reading", "Deep Learning", "Autonomous Driving", "Point Cloud", "Point Cloud Registration"]
categories:
- Deep Learning
- Review
mathjax: true
---

　　点云注册是是点云数据处理中非常重要的一个方向。{%post_link Object-Registration-with-Point-Cloud Object Registration with Point Cloud%} 中描述了基于点云的目标注册方法，主要阐述了传统 ICP 原理以及基于深度学习进行目标注册(相对位姿估计)的方法。本文<a href="#1" id="1ref"><sup>[1]</sup></a>则详细介绍整个点云注册方法的类别与细节。

## 1.&ensp;Problem Definition
　　假设两个点云集 \\(X\\in\\mathbb{R} ^ {M\\times 3}, Y\\in\\mathbb{R} ^ {N\\times 3}\\)，其中每个点表示为 \\(\\mathbf{x} _ i(i\\in [1,M])\\)，\\(\\mathbf{y} _ i(y\\in [1,N])\\)。两个点云集合中有 \\(K\\) 个匹配点对，点云注册问题就是找到参数 \\(g\\)，即旋转矩阵 \\(R\\in\\mathcal{SO}(3)\\) 和位移矩阵 \\(t\\in\\mathbb{R} ^ 3\\)，使得：
$$\mathop{\arg\min}\limits _ {R\in\mathcal{SO}(3),t\in\mathbb{R} ^ 3}\Vert d(X,g(Y))\Vert ^ 2 _ 2 \tag{1}$$
其中 \\(d(X,g(Y))=d(X,RY+t)=\\sum _ {k=1} ^ K \\Vert \\mathbf{x} _ k-(R\\mathbf{y} _ k+t)\\Vert _ 2\\)。这个问题是典型的鸡生蛋蛋生鸡问题，如果匹配点对已知，那么变换矩阵可以求解；如果变换矩阵已知，那么就能得到匹配点对。

## 2.&ensp;Challenges
　　根据数据源类型，点云注册可分为 same-source 以及 cross-source 两类。其挑战分别有：

- Same-source  
    1. Noise and Outliers  
    2. Partial overlap  
- Cross-source  
    1. Noise and Outliers  
    2. Partial overlap  
    3. Density difference  
    不同传感器获得的数据源，点云密度可能不一样。
    4. Scale variation  
    不同传感器获得的数据源，点云的空间尺度可能不一样。

## 3.&ensp;Categories
<img src="taxonomy.png" width="90%" height="90%" title="图 1. Taxonomy">

- Optimisation-based  
- Feature learning  
- End-to-end learning-based
- Cross-source registration

## 4.&ensp;Optimisation-based
<img src="optimization-based.png" width="90%" height="90%" title="图 2. Optimization-based">

### 4.1.&ensp;ICP-based

### 4.2.&ensp;Graph-based

### 4.3.&ensp;GMM-based

### 4.4.&ensp;Semi-definite Registration

## 5.&ensp;Feature-learning
<img src="feature-learning.png" width="90%" height="90%" title="图 3. Feature-learning">

### 5.1.&ensp;Learning on volumetric data

### 5.2.&ensp;Learning on point cloud

## 6.&ensp;End-to-end Learning-based
<img src="end-end.png" width="90%" height="90%" title="图 4. End-to-end">

### 6.1.&ensp;Registration by regression

### 6.2.&ensp;Registration by optimization and neural network

## 7.&ensp;Cross-source
<img src="cross-source.png" width="90%" height="90%" title="图 5. cross-source">

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a>

