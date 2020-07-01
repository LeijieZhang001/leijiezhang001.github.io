---
title: Rethinking of Sparse 3D Convolution
date: 2020-06-23 17:37:12
updated: 2020-06-25 09:19:12
tags: ["Deep Learning", "Autonomous Driving", "Point Cloud"]
categories:
- Deep Learning
mathjax: true
---

　　Sparse 3D Convolution 最早在<a href="#1" id="1ref">[1]</a>中提出，然后该作者又提出了 Submanifold Sparse Convolution<a href="#2" id="2ref"><sup>[2]</sup></a>，并将其应用于 3D 语义分割中<a href="#3" id="3ref"><sup>[3]</sup></a>。<a href="#4" id="4ref">[4]</a>则改进了 Sparse 3D Convolution 的实现方式，并应用于 3D 目标检测中。之前一直没仔细看 Sparse 3D Convolution 原理，以为只是基于稀疏矩阵的矩阵相乘加速，最近的一些实验发现 Sparse 3D Convolution 在点云相关的任务中不仅仅是加速，还能提升网络特征提取的性能，所以回过头来重新思考 Sparse 3D Convolution 原理及作用。

## 1.&ensp;Sparse Convolution
<img src="spconv.png" width="85%" height="85%" title="图 1. sparse VS. submanifold sparse">
　　如图 1. 左图所示，对于稀疏的特征输入，传统的 Sparse Convolution 与 Convolution 一致，只是对于卷积核覆盖的输入特征为零的区域不做计算，直接置为零。这种方式下，随着卷积层的增加，特征层会变得不那么稀疏，这样不仅使得计算量上升，而且会使得提取的信息变得不那么准确。

## 2.&ensp;Submanifold Sparse Convolution
　　如图 1. 右图所示，Submanifold Sparse Convolution 解决了 Sparse Convolution 存在的问题。原理也很直观：只计算输出特征层映射到输入特征层不为零的位置区域。这种方式下，随着卷积层的增加，不仅能保持稀疏性，而且能保证原始信息的准确性。
<img src="flops.png" width="85%" height="85%" title="图 2. Flops">
　　如图 2. 所示，Sparse Convolution 相比传统的 Convolution 已经能减少较多的计算量，而 Submanifold Sparse Convolution 则能减少更多的计算量。特征输入越稀疏，减少的计算量就越多，这对点云的三维特征提取，或者是俯视图下的二维特征提取有很大的帮助。

## 3.&ensp;Implementation
<img src="speed.png" width="85%" height="85%" title="图 3. Speed">
　　<a href="#2" id="2ref">[2]</a> 中实现了 Submanifold Sparse Convolution，其中的卷积运算是手写的矩阵相乘，所以速度较慢；<a href="#4" id="4ref">[4]</a> 则基于 GEMM 实现了更高效的 Submanifold Sparse Convolution。如图 3. 所示，其有将近一倍的速度提升。
<img src="imple.png" width="90%" height="90%" title="图 4. Implementation">
　　图 4. 描述了<a href="#4" id="4ref">[4]</a>实现的 Submanifold Sparse Convolution 原理。其首先通过 gather 操作将非零的元素进行矩阵相乘，然后通过 scatter 操作将结果映射回原位置。为了加速，前后元素的映射矩阵计算比较关键，这里实现了一种 GPU 计算方法，这里不做展开。

## 4.&ensp;Application
<img src="second.png" width="90%" height="90%" title="图 5. SECOND Framework">
　　Submanifold Sparse Convolution 可应用于点云的分类，分割，检测等任务的特征提取中，SECOND<a href="#4" id="4ref"><sup>[4]</sup></a>是一种点云检测方法，如图 5. 所示，其检测框架与传统的一致，只是将体素化后的点云特征信息，进一步用 Sparse Convolution 来作特征提取。该方法不仅速度较快，而且性能也有不少提升。所以 Submanifold Sparse Convolution 是非常高效的，可作为点云特征提取的基本操作。但是传统的 Convolution，在 GPU 平台下，已经有较多的硬件级优化(cudnn)，在 CPU 平台下也有很多的指令集优化，所以最终在特定硬件下作 Inference 时，到底 Submanifold Sparse Convolution 速度能提升多少，还得看 Submanifold Sparse Convolution 实现的好不好。不过可以猜测，在目前的实现下，Submanifold Sparse Convolution 在 GPU 平台下应该能有不少的速度提升。  
　　此外，传统的卷积量化操作也比较成熟，cudnn 已经有基本的操作引擎，而 Submanifold Sparse Convolution 的 INT8 引擎则目前还没有。所以 float32/float16 的 Submanifold Sparse Convolution 与 INT8 的 Convolution，孰快孰慢？这两条路大概就是部署的思路了，当然 INT8 的 Submanifold Sparse Convolution 会更好，但是开发成本会比较高。

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Graham, Ben. "Sparse 3D convolutional neural networks." arXiv preprint arXiv:1505.02890 (2015).  
<a id="2" href="#2ref">[2]</a> Graham, Benjamin, and Laurens van der Maaten. "Submanifold sparse convolutional networks." arXiv preprint arXiv:1706.01307 (2017).  
<a id="3" href="#3ref">[3]</a> Graham, Benjamin, Martin Engelcke, and Laurens Van Der Maaten. "3d semantic segmentation with submanifold sparse convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.  
<a id="4" href="#4ref">[4]</a> Yan, Yan, Yuxing Mao, and Bo Li. "Second: Sparsely embedded convolutional detection." Sensors 18.10 (2018): 3337.  
