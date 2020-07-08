---
title: CenterTrack
date: 2020-07-02 09:16:36
updated: 2020-07-08 09:19:12
tags: ["Deep Learning", "Autonomous Driving", "Tracking", "MOT"]
categories:
- MOT
mathjax: true
---

　　障碍物感知由目标检测，目标跟踪(MOT)，目标状态估计等三个模块构成。目标状态估计一般是指将位置，速度等观测量作卡尔曼滤波平滑；广义的目标跟踪也包含了状态估计过程，这里采用狭义的目标跟踪定义方式，主要指出目标 ID 的过程。传统的做法，目标检测与目标跟踪是分开进行的，检测模块分别对前后帧作目标检测，目标跟踪模块则接收前后帧检测结果，然后用 Motion Model 将上一帧的检测结果预测到这一帧，最后与这一帧的检测结果作数据关联(Data Association)出目标 ID。这里的 Motion Model 可以是 3D 下目标的物理运动模型，也可以是图像下的单目标跟踪结果，如 KCF 算法。详细介绍可参考 {%post_link MOT-综述-Multiple-Object-Tracking-A-Literature-Review Multiple Object Tracking: A Literature Review%}。  
　　随着检测技术的发展，检测与跟踪的整合成为了趋势。<a href="#1" id="1ref">[1]</a> 是较早将跟踪的 “Motion Model” 用 Anchor-based Two-stage 网络来预测的方法，其网络输入为前后帧图像，其中一个分支输出当前帧的检测框，另一个分支用上一帧的检测结果作为 proposal，输出这一帧的跟踪框，最后用传统的数据关联方法得到目标的 ID。随着检测技术往 Anchor-Free One-stage 方向发展，在此基础上整合目标检测与跟踪也就顺理成章。  
　　{%post_link Anchor-Free-Detection Anchor-Free Detection%} 中详细描述了 Anchor-Free 的目标检测方法，相比于 Anchor-Based 的目标检测，其有很多优势，这里不做赘述。本文基于 CenterNet<a href="#2" id="2ref"><sup>[2]</sup></a>，总结了 CenterTrack<a href="#3" id="3ref"><sup>[3]</sup></a>，以及 CenterPoint(3D CenterTrack)<a href="#4" id="4ref"><sup>[4]</sup></a>方法。

## 1.&ensp;CenterNet
　　CenterNet 在 {%post_link Anchor-Free-Detection Anchor-Free Detection%} 中已经较为详细得阐述了。需要补充的是，中心点的正负样本设计为：正样本只有中心点像素，负样本则为其它区域，并加入以中心点为中心的高斯权重，越靠近中心点，负样本权重越小。其 Loss 基于 Focal Loss，数学描述为：
$$L _ k = \frac{1}{N}\sum _ {xyc}\left\{\begin{array}{l}
(1-\hat{Y} _ {xyc})^{\alpha}\mathrm{log}(\hat{Y} _ {xyc}) & \mathrm{if}\; Y _ {xyc} = 1\\
(1- Y _ {xyc})^{\beta}(\hat{Y} _ {xyc})^{\alpha}\mathrm{log}(1-\hat{Y} _ {xyc}) & \mathrm{otherwise}
\end{array}\tag{1}\right.$$
其中 \\(Y _ {xyc}\\) 为高斯权重后的正负样本分布值。  
　　具体的，设图像 \\(I\\in \\mathbb{R}^{W\\times H\\times 3}\\)，CenterNet 输出的每个类别 \\(c\\in\\{0,...,C-1\\}\\) 的目标为 \\(\\{(\\mathbf{p} _ i, \\mathbf{s} _ i)\\} _ {i=0} ^ {N-1}\\)。其中 \\(\\mathbf{p}\\in \\mathbb{R} ^ 2\\)，\\(\\mathbf{s}\\in\\mathbb{R} ^ 2\\) 为目标框的尺寸。对应的，最终输出的 heatmap 位置和尺寸图为：\\(\\hat{Y}\\in [0,1]^{\\frac{W}{R}\\times\\frac{H}{R}\\times C}\\)，\\(\\hat{S}\\in\\mathbb{R}^{\\frac{W}{R}\\times\\frac{H}{R}\\times 2}\\)。对 \\(\\hat{Y}\\) 作 \\(3\\times 3\\) 的 max pooling，即可获得目标中心点，\\(\\hat{S}\\) 上对应的的点即为该目标的尺寸。此外还用额外的 heatmap 作位置 offset 的回归，因为 \\(\\hat{Y}\\) 存在量化误差。最终由中心点位置 loss，位置 offset loss，尺寸 loss 三部分组成。

## 2.&ensp;CenterTrack

### 2.1.&ensp;Framework
<img src="centertrack.png" width="85%" height="85%" title="图 1. CenterTrack">
　　如图 1. 所示，CenterTrack 基于 CenterNet，框架也较为简单：输入前后帧图像，以及上一帧跟踪到的目标中心点所渲染的 heatmap，经过网络后输出为当前帧的检测 heatmap，size map，以及这一帧相对上一帧跟踪的 offset map。最后通过最近距离匹配即可作数据关联获得目标的 ID。算法得到的目标属性有 \\(b = (\\mathbf{p,s},w,id)\\)，分别为目标的 location，size，confidence，identity。  
　　相比于 CenterNet，CenterTrack 还预测了这一帧相对上一帧，目标的 2D displacement：\\(\\hat{D}\\in\\mathbb{R}^{\\frac{W}{R}\\times\\frac{H}{R}\\times 2}\\)。这相当于 Tracking 中 Motion Model 的结果，分别计算上一帧目标经过该 displacement 变换到这一帧后的目标位置与当前帧检测的目标位置的距离误差，用最小距离的贪心法即可将目标作数据关联，得到目标的 ID。

### 2.2.&ensp;Experiments
　　网络结构相比于 CenterNet 只是增加了输入的四个通道特征，输出的两个通道特征。网络可在视频流图像或者单帧图像上训练，对于单帧图像，可对图像中的目标作伸缩平移变换来模拟目标运动，实验表明，也非常有效。
<img src="motion_models2d.png" width="85%" height="85%" title="图 2. Motion Models">
　　如图 2. 所示，本文比较了 displacement 与 kalman filter，optical flow 等 Motion Model，显示本文效果是最好的，我猜测是因为 displacement 回归的直接是物体级别的像素运动，抗噪性更强。

## 3.&ensp;Center-based 3D Object Detection and Tracking
### 3.1.&ensp;Framework
<img src="centertrack3d.png" width="85%" height="85%" title="图 3. 3D CenterTrack">
　　如图 3. 所示，CenterPoint 将点云在俯视图下栅格化，然后采用 CenterTrack 一样的网络结构，只是输出为目标的 3D location，size，orientation，velocity。  
　　点云俯视图下的栅格化，如果对栅格不做点云的精细化估计，那么会影响到目标位置及速度估计的精度，所以理论上 PointPillars 这种栅格点云特征学习方式能更有效的提取点云的信息，保留特征的连续化信息(但是论文的实验表明 VoxelNet 比 PointPillars 效果更好)。否则，虽然目标位置等信息的监督项是连续量，但是栅格化的特征是离散量，这会降低预测精度。  
　　具体的，网络输出为：\\(K\\) 个类别的 \\(K\\)-channel heatmap 表示目标中心点，目标的尺寸 \\(\\mathbf{s}=(w,l,h)\\) heatmap，目标的中心点 offset \\(\\mathbf{o}=(o _ x,o _ y,o _ z)\\) heatmap，朝向角 \\(\\mathbf{e} = (\\mathrm{sin}(\\alpha),\\mathrm{cos}(\\alpha))\\) heatmap，目标速度 \\(\\mathbf{v}=(v _ x,v _ y)\\) heatmap。与 CenterTrack 非常相似，只不过这里的速度就是真实的物理速度。

### 3.2.&ensp;Experiments
<img src="detmap.png" width="85%" height="85%" title="图 4. 3D Detection Benchmark">
　　如图 4. 所示，引入 Velocity 预测，能有效提升检测的性能，这应该是网络输入前一帧信息的结果，对半遮挡情况能有较好效果。
<img src="experiment3d.png" width="85%" height="85%" title="图 5. 3D MOT Benchmark">
　　如图 5. 所示，跟踪性能也是有很大提升，而且数据关联等后处理相对比较简单。

## 5.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Feichtenhofer, Christoph, Axel Pinz, and Andrew Zisserman. "Detect to track and track to detect." Proceedings of the IEEE International Conference on Computer Vision. 2017.  
<a id="2" href="#2ref">[2]</a> Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).  
<a id="3" href="#3ref">[3]</a> Zhou, Xingyi, Vladlen Koltun, and Philipp Krähenbühl. "Tracking Objects as Points." arXiv preprint arXiv:2004.01177 (2020).  
<a id="4" href="#4ref">[4]</a> Yin, Tianwei, Xingyi Zhou, and Philipp Krähenbühl. "Center-based 3D Object Detection and Tracking." arXiv preprint arXiv:2006.11275 (2020).  
