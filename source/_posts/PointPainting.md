---
title: '[paper_reading]-"PointPainting: Sequential Fusion for 3D Object Detection"'
date: 2020-06-17 11:27:38
tags: ["paper reading", "3D Detection", "Deep Learning", "Autonomous Driving"]
categories: 3D Detection
mathjax: true
---
　　相机能很好的捕捉场景的语义信息，激光雷达则能很好的捕捉场景的三维信息，所以图像与点云的融合，对检测，分割等任务有非常大的帮助。融合可分为，**数据级或特征级的前融合**，以及**任务级的后融合**。本文<a href="#1" id="1ref"><sup>[1]</sup></a>提出了一种将图像分割结果的语义信息映射到点云，进而作 3D 检测的方法。这种串行方式的融合，既有点前融合的意思，也有点后融合的意思，暂且可归为前融合吧。本方法可认为是个框架，该框架下，基于图像的语义分割，以及基于点云的 3D 检测，均为独立模块。实验表明，融合了图像的语义信息后，点云针对行人等小目标的检测有较大的性能提升。

## 1.&ensp;Framework
<img src="framework.png" width="100%" height="100%" title="图 1. Framework">
　　如图 1. 所示，算法框架非常简单，一句话能说明白：1). 首先经过图像语义分割获得语义图；2). 然后将点云投影到图像上，查询点云的语义信息，并连接到坐标信息中；3). 最后用点云 3D 检测的方法作 3D 检测。

## 2.&ensp;Experiments
<img src="sota.png" width="90%" height="90%" title="图 2. PointPainting Applied to SOTA">
　　采用 DeepLabv3+ 作为语义分割模块，应用到不同的点云 3D 检测后，结果如图 2. 所示，均有不同程度的提升，尤其是行人这种小目标。
<img src="pointrcnn.png" width="90%" height="90%" title="图 3. Painted PointRCNN">
　　图 3. 显示了 Painted PointRCNN 与各个方法的对比结果，mAP 是最高的。
<img src="per-class.png" width="90%" height="90%" title="图 4. 不同类别的提升程度">
　　由图 4. 可知，对行人，自行车，雪糕筒等小目标(俯视图下来说)，本方法提升非常显著。这也比较好理解，因为前视图下，这些目标所占的像素会比较多，所以更容易在前视相机图像下提取有效信息，辅助俯视图下作更准确的检测。  

## 3.&ensp;Rethinking of Early Fusion
　　这里将本方法归为前融合，但是并不是真正意义上的前融合。如果是前融合，那么一般是 concate 语义分割网络的中低层特征到点云信息中，然而本文是直接取语义分割网络的最高层特征(即分类结果)。**所以问题来了，所谓的前融合，一定比后融合更好吗？**我想，这篇文章可能给了一些答案(不知道作者有没有做过取其它特征的实验，姑且认为做过，然后选择了本方法的策略)，虽然理论上前融合信息最完整，但是，如果这种完整的信息无法有效学出来或者对标定外参比较敏感，那么这种前融合也提升不了后续任务的性能，更有甚者，由于信息空间的变大或紊乱，导致后续任务性能下降。相反，对于不是那么“前”的后融合，我们能极大得保证各个任务学习结果的有效性，基于此，融合后学习的有效性也会比较确定。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Vora, Sourabh, et al. "Pointpainting: Sequential fusion for 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
