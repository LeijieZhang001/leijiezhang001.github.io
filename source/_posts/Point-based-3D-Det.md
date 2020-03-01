---
title: Point-based 3D Detetection
date: 2020-02-29 12:30:15
tags: ["3D Detection", "Deep Learning", "autonomous driving", "Point Cloud"]
categories: 3D Detection
mathjax: true
---

　　基于激光点云的 3D 目标检测是自动驾驶系统中的核心感知模块。由于点云的稀疏性以及空间结构的无序性，一系列 Voxel-based 3D 检测方法得以发展：{% post_link paperreading-PointPillars PointPillars%}，{% post_link paperreading-Fast-and-Furious FaF%}，{% post_link paperreading-End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds MVF%} 等。然而 Voxel-based 方法需要预定义空间栅格的分辨率，其特征提取的有效性依赖于空间分辨率。同时在点云语义分割领域，对点云的点级别特征提取方法研究较为广泛，{% post_link PointCloud-Feature-Extraction PointCloud Feature Extraction%} 中已经较详细的介绍了针对点云的点级别特征提取方法，{% post_link paper-reading-Grid-GCN-for-Fast-and-Scalable-Point-Cloud-Learning Grid-GCN%} 提出了几种策略来加速特征提取。  
　　由此高效的 Point-based 3D 检测方法成为可能，这种方法首先提取点级别的特征(相比 Voxel-based，理论上没有信息损失)，然后用点级别的 Anchor-based 或 Anchor-free 方法作 3D 检测。

## 1.&ensp;Anchor-based

### 1.1.&ensp;IPOD<a href="#1" id="1ref"><sup>[1]</sup></a>
<img src="ipod.png" width="90%" height="90%" title="图 1. IPOD Framework">
　　如图 1. 所示， IPOD 与 F-PointNet 类似，只不过 IPOD 在俯视图下生成 Proposal 取点，而 F-PointNet 是直接在锥形视野的点云中作分割。IPOD 由三部分组成：

1. **Semantic Segmentation**  
目的是将点云中的背景点过滤掉，只生成前景点的 Anchor。作者采用图像语义分割的方法，这里也可直接用点云分割来做；
2. **Point-based Proposal Generation**  
生成点级别的候选框，去掉冗余的候选框；
3. **Head for Classification and Regression**  
根据候选框，提取特征，作分类和回归；

这里的前两步是要得到少量但又能保证召回率的 Proposal，其中 Anchor 是根据每个点来设置的，然后作 NMS 操作，这里不做展开。
<img src="proposal_feat.png" width="80%" height="80%" title="图 2. Proposal Feature Generation">
　　如图 2. 所示，每个 Proposal 提取出点云信息，然后通过 PointNet++ 直接来预测该 Proposal 的 3D 属性。这里用到了 T-Net(Spatial Transformation Network 的一种) 将点云变换到规范坐标系(Canonical coordinates)，这个套路用的也比较多。其它细节就是正常的 3D 属性回归策略，不作展开。

### 1.2.&ensp;STD<a href="#2" id="2ref"><sup>[2]</sup></a>
<img src="STD.png" width="80%" height="80%" title="图 3. STD Framework">
　　如图 3. 所示，STD 模块有：

1. **Backbone**  
用 PointNet++ 提取点级别特征以及作点级别的 Classification；
2. **PGM(Proposal Generation Module)**  
根据点级别的分类结果，对目标点设计球状 Spherical Anchor；不同类别设计不同的球状 Anchor 半径。将球状 Anchor 里面的点收集起来，作坐标规范化并且 concate 点级别特征，然后用 PointNet 来预测实际的矩形 proposal：包括中心 Offsets 以及 size offsets。同时对角度进行预测，角度预测通过分类加预测 Offsets 实现。
3. **Proposal Feature Generation**  
有了 proposal 后，其实可以直接通过 PointNet 作进一步的预测及分类，但是作者为了加速，这时候采用了 Voxel Feature Encoding。将 proposal 里面的点都转换到中心点坐标系，然后栅格化提取特征；
4. **Box Prediction**  
除了通常的类别预测以及 3D Box 相关属性的 Offsets 预测，作者还加入了与真值的 IoU 预测，该 IoU 值与类别分数相乘作为最终的该预测分数(这个在 2D Detection 中已经有应用)。

## 2.&ensp;Anchor-free

### 2.1.&ensp;PointRCNN<a href="#3" id="3ref"><sup>[3]</sup></a>
<img src="PointRCNN.png" width="80%" height="80%" title="图 4. PointRCNN Framework">
　　如图 4. 所示，PointRCNN 是一个 two-stage 3D 检测方法，类似 Faster-RCNN，其由 Bottom-up 3D Proposal Generation 和 Canonical 3D Box Refinement 两个模块组成。

#### 2.1.1&ensp;Bottom-up 3D Proposal Generation
　　Proposal 的生成要求是，数量少，召回率高。3D Anchor 由于要覆盖 3D 空间，所以数量会很大(如 AVOD)，本文采用目标点生成 Proposal 的方法。与 IPOD，STD 类似，首先对点云进行点级别的特征提取并作前景分割(或语义分割)，对前景的每个点用 Bin-based 方法生成 3D proposal。由此在生成尽量少的 Proposal 下，保证目标的高召回率。  
　　点级别的特征提取及前景分割，可以采用任意的语义分割网络，这里前景的真值即为目标框内的点云，用 Focal Loss 来平衡正负样本。  
<img src="bin-based.png" width="60%" height="60%" title="图 5. Bin-based Localization">
　　如图 5. 所示，对每个前景点用 Bin-based 方法生成 proposal。将平面的 \\(x,z\\) (与一般的雷达坐标系不同) 方向分成若干个 bin，然后对每个前景点，预测目标中心点属于哪个 bin，以及中心点与该 bin 的 Offsets(与角度处理的方式非常像)。针对尺寸，预测该类别平均尺寸的 Residual；针对角度，还是分解成分类加回归任务进行处理。最后再作 NMS 即可得到较少的 Proposal，给到下一模块作 refine。本模块的 Loss  设计为：
$$\begin{align}
\mathcal{L} _ 1 &= \mathcal{L} _ {seg} + \mathcal{L} _ {proposal} \\
&= \mathcal{L} _ {seg} + \frac{1}{N _ {pos}} \sum _ {p\in pos} \left(\mathcal{L} _ {bin} ^ {(p)} + \mathcal{L} _ {res} ^ {(p)}\right) \\
&= \mathcal{L} _ {seg} + \sum _ {u\in{\{x,z,\theta\}}} \left(\mathcal{F} _ {cls}(\widehat{bin} _ u^{(p)}, bin _ u^{(p)})+\mathcal{F} _ {reg}(\widehat{res} _ u^{(p)}, res _ u^{(p)})\right) + \sum _ {v\in\{y,h,w,l\}} \mathcal{F} _ {reg}(\widehat{res} _ v^{(p)}, res _ v^{(p)})\\
\tag{1}
\end{align}$$
其中 \\(\\mathcal{F} _ {cls}, \\mathcal{F} _ {reg}\\) 分别为 cross-entropy Loss 和 smooth L1 Loss。

#### 2.1.2&ensp;Canonical 3D Box Refinement
　　有了 3D proposal 后，经过 Point Cloud Region Pooling 提取该 proposal 的点特征，步骤如下：先对 proposal 进行一定程度的扩大，然后提取内部点的 semantic features，foreground mask score，Point distance等。由此获得每个 proposal 的点及点特征，用来作 3D Box Refinement。  
<img src="canonical.png" width="60%" height="60%" title="图 6. Canonical Transformation">
　　如图 4. 所示，为了更好的学习 proposal 的局部空间特征，增加每个 proposal 在自身 Canonical 坐标系下的空间点。Canonical 变换如图 6. 所示，因为这里每个 proposal 的位置及角度已经有了，所以直接对其内的点作变换。如果没有，那就需要 STN(T-Net) 来学习这个变换。  
　　Loss 也是在 Canonical 坐标系下计算的，假设 proposal：\\(\\mathrm{b _ i} = (x _ i,y _ i,z _ i,h _ i,w _ i,l _ i,\\theta _ i)\\)，真值: \\(\\mathrm{b} _ i^{gt} = (x _ i^{gt}, y _ i^{gt},z _ i^{gt},h _ i^{gt},w _ i^{gt},l _ i^{gt},\\theta _ i^{gt})\\)。那么两者变换到 Canonical 坐标系后：
$$\begin{align}
\mathrm{\tilde{b}} _ i &=(0,0,0,h _ i,w _ i,l _ i,0) \\
\mathrm{\tilde{b}} _ i^{gt} &= (x _ i^{gt}-x _ i, y _ i^{gt}-y _ i,z _ i^{gt}-z _ i,h _ i^{gt},w _ i^{gt},l _ i^{gt},\theta _ i^{gt}-\theta _ i)
\tag{2}
\end{align}$$
对于中心点，还是 bin 分类加 Residual 回归，但是可以减少 bin 的尺度；对于尺寸，还是回归 Residual；对于角度，由于限定 positive 与 gt 的 IoU>0.55，所以可以将回归的角度限定为 \\((-\\frac{\\pi}{4},\\frac{\\pi}{4})\\) 的范围，由此进行 bin 分类及 Residual 回归。最终本阶段的 Loss 为：
$$ \mathcal{L} _ 2= \frac{1}{N _ {pos}+ N _ {neg}} \sum _ {p\in all} \mathcal{L} _ {label} ^{(p)}+ \frac{1}{N _ {pos}} \sum _ {p\in pos} \left(\mathcal{\tilde{L}} _ {bin} ^ {(p)} + \mathcal{\tilde{L}} _ {res} ^ {(p)}\right) \tag{3}$$

### 2.2.&ensp;3DSSD<a href="#4" id="4ref"><sup>[4]</sup></a>
<img src="3DSSD.png" width="100%" height="100%" title="图 7. 3DSSD Framework">
　　如图 7. 所示，3DSSD 是 one-stage 网络，由 Backbone，Candidate Generation Layer，Head 构成。Backbone 作者提出了 Fusion Sampling 以提升前景点在采样时候的召回率。Candidate Generation Layer 中根据前景点，生成 3D box 预测的 Candidate 锚点。最后 Head 根据锚点，作 Anchor-free 的 3D Box 预测。

#### 2.2.1&ensp;Fusion Sampling
　　为了扩大感受野提取局部特征，点云通常需要作下采样处理，一般采用 D-FPS 方法(点空间距离作为采样度量)，但是这样会使前景点大量丢失。前面几种方法不管是用图像分割还是点云分割，都会去除背景点云，保留前景点云以提高生成 Proposal 的召回率。  
　　这里作者提出了 Feature-FPS，加入特征间的距离作为采样的度量方式。对于地面等背景，其特征基本类似，所以很容易就去除了；而对于目标区域，其点特征都不太一样，又得以保留。如果只保留同一目标的点，也会产生冗余，所以融合点特征距离及空间距离，设计采样度量方式为：
$$ C(A,B) = \lambda L _ d(A,B) + L _ f(A,B) \tag{4}$$
　　因为 F-FPS 去除了大量的背景点，虽然有利于回归，但是不利于分类，所以设计了融合 D-FPS 和 F-FPS 的 Fusion Sampling 方法。如图 7. 所示，最终分别输出 F-FPS 与 D-FPS 的特征点。

#### 2.2.2&ensp;Candidate Generation Layer
<img src="candidate_pts.png" width="60%" height="60%" title="图 8. Candidate Generation">
　　如图 8. 所示，根据 F-FPS 采样的点，在真值框中心点的监督下，用一个 T-Net 去学习采样点与中心点的变换。变换后的点即作为 Candidate 锚点。对每个 Candidate 点提取周围一定距离的 F-FPS 与 D-FPS(大量背景点利于分类)中点集的特征(空间坐标作归一化或变换到 Candidate 坐标系，类似 Canonical 坐标系)，然后作 MaxPool 提取该 Candidate 对应区域的特征。

#### 2.2.3&ensp;Prediction Head
　　对于每个 Candidate 特征，作 3D Box 属性的回归。本文采用 Anchor-free 的方法。对于中心点，直接回归 Candidate 坐标点与真值框中心点的 Offsets；对于尺寸，直接回归与该类别平均尺寸的 Residual；对于角度，还是采用 bin 分类加 Residual 回归的策略。  
　　这里期望的是 Candidate 点能接近目标框中心点，所以作者借鉴 FCOS(详见 {% post_link Anchor-Free-Detection Anchor-Free Detection %})中的 Center-ness Loss 来选取靠近中心点的 Candidate，真值 Label 为:
$$l _ {ctrness}=\sqrt[3]{\frac{\mathrm{min}(f,b)}{\mathrm{max}(f,b)}+\frac{\mathrm{min}(l,r)}{\mathrm{max}(l,r)}+\frac{\mathrm{min}(t,d)}{\mathrm{max}(t,d)}} \tag{5}$$
其中 \\(f,b,l,r,t,d\\) 分别表示前后左右上下与中心点的距离。FCOS 中，加了一个与分类平行的分支来预测 Center-ness，最终的预测 Score 是分类 Score 乘以 Center-ness 得到(与预测 IoU 套路一样，本质上都是引入与真值的距离度量)，该预测 Score 用于之后的 NMS 等处理。本文则没有显示的预测 Center-ness，其直接将真值 Center-ness 与真值类别相乘，作为类别真值，所以一个类别分支即得到最终的预测 Score。  
　　最终的 Loss 为：
$$L = \frac{1}{N _ c}\sum _ iL _ c(s _ i, u _ i) + \lambda _ 1\frac{1}{N _ p}\sum _ i[u _ i>0]L _ r + \lambda _ 2\frac{1}{N _ p}L _ s \tag{5}$$
其中 \\(s _ i\\) 为预测的类别 Score，\\(u _ i\\) 为经过 Center-ness 处理后的类别真值；\\(L _ c\\) 表示类别预测 Loss；\\(L _ r\\) 表示 3D Box Loss，包括中心点距离，尺寸，角度，8个角点位置；\\(L _ s\\) 表示生成 Candidate 点的 shift 变换 Loss。

## 3.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Yang, Zetong, et al. "Ipod: Intensive point-based object detector for point cloud." arXiv preprint arXiv:1812.05276 (2018).  
<a id="2" href="#2ref">[2]</a> Yang, Zetong, et al. "Std: Sparse-to-dense 3d object detector for point cloud." Proceedings of the IEEE International Conference on Computer Vision. 2019.  
<a id="3" href="#3ref">[3]</a> Shi, Shaoshuai, Xiaogang Wang, and Hongsheng Li. "Pointrcnn: 3d object proposal generation and detection from point cloud." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  
<a id="4" href="#4ref">[4]</a> Yang, Zetong, et al. "3DSSD: Point-based 3D Single Stage Object Detector." arXiv preprint arXiv:2002.10187 (2020).  
