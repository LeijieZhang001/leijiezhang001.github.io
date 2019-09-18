---
title: MOT Metrics in Academia and Industry
date: 2019-06-03 13:47:00
tags: ["MOT", "tracking", "autonomous driving"]
categories: MOT
mathjax: true
---

　　MOT 是一个比较基本的技术模块，在视频监控中，常用于行人行为分析、姿态估计等任务的前序模块；在自动驾驶中，MOT 是动态目标状态估计的重要环节。在学术界，MOT 算法性能的评价准则已经较为完善，其指标主要关注，尽可能地覆盖所有性能维度，以及指标的简洁性（上一篇有较多介绍，[the CLEAR MOT Metrics](https://leijiezhang001.github.io/MOT-%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87-Evaluating-Multiple-Object-Tracking-Performance-the-CLEAR-MOT-Metrics/#more)）。而工业界则尚无统一的标准，实际的指标需求情况也比学术界复杂。  
　　指标的计算过程可由三部分组成，真值过滤(Filter)，匹配构建(Establishing Correspondences)与指标计算(Calculating Metrics)。其中真值过滤，更多的是工程细节，学术界没有文章对这一部分进行讨论研究。本文首先介绍学术界各评价指标详情，然后讨论工业界需要的评价指标又是怎样的。

## 1.&ensp;Metrics in Academia

　　在学术界，因为数据集质量较高，噪声相对较小，匹配构建中距离的度量偏向于严格且简单的方式。对于区域(框)跟踪器，采用重叠区域来度量；对于点跟踪器，采用中心点的欧式距离来度量。指标汇总如下：

A.&ensp;**检测指标**  
　\\(\\lozenge\\)　准确性(Accuracy)

- **Recall** = \\(\\frac{TP}{GT}\\)；
- **Precision** = \\(\\frac{TP}{TP+FP}\\)；
- **FAF/FPPI**<a href="#1" id="1ref"><sup>[1]</sup></a><a href="#2" id="2ref"><sup>[2]</sup></a> ，Average False Alarms per Frame；False Positive Per Image;
- **MODA**<a href="#3" id="3ref"><sup>[3]</sup></a>，Multipe Object Detection Precision，整合了 FN 与 FP，设 \\(c_m, c_f\\) 分别为 FN，FP 的权重：
$$MODA=1-\frac{\sum_{t=1}^{N_frames}(c_m(fn_t)+c_f(fp_t))}{\sum_{t=1}^{N_frames}gt_t}$$

　\\(\\lozenge\\)　精确性(Precision)

- **MODP**<a href="#3" id="3ref"><sup>[3]</sup></a>，Multiple Object Detection Accuracy，
$$MODP=\frac{\sum_{t=1}^{N_frames} \sum_{i=1}^{N_{mapped}^{(t)}} \;\; dist}{\sum_{t=1}^{N_frames} N_{mapped}^{(t)}}$$
其中 \\(N_{mapped}^{(t)}\\) 为第 \\(t\\) 帧匹配的目标数；\\(dist\\) 为距离度量方法，如框的交并比度量法：
$$Mapped Overlap Ratio = \frac{\lvert G_i^{(t)}\bigcap D_i^{(t)}\rvert}{|G_i^{(t)}\bigcup D_i^{(t)}|}$$

B.&ensp;**跟踪指标**  
　\\(\\lozenge\\)　准确性(Accuracy)

- **IDS**<a href="#4" id="4ref"><sup>[4]</sup></a>，ID switch，a tracked target changes its ID with another target(预测关联真值)；
- **MOTA**<a href="#5" id="5ref"><sup>[5]</sup></a>，Multiple Object Tracking Accuracy，整合了 FN，FP，ID-Switch：
$$MOTA=1-\frac{\sum_{t=1}^{N_{frames}} \;\; (c_m(fn_t)+c_f(fp_t)+c_s(ID-SWITCHES_t))}{\sum_{t=1}^{N_{frames}} \;\; gt_t}$$
其中权重方程一般可设为：\\(c_m=c_f=1, \\quad c_s=log_{10}\\)；

　\\(\\lozenge\\)　精确性(Precision)

- **MOTP**<a href="#5" id="5ref"><sup>[5]</sup></a>，Multiple Object Tracking Precision，
$$MODP=\frac{\sum_{t=1}^{N_frames} \sum_{i=1}^{N_{mapped}^{(t)}} \;\; \left(\frac{\lvert G_i^{(t)}\bigcap D_i^{(t)}\rvert}{|G_i^{(t)}\bigcup D_i^{(t)}|} \right)}{\sum_{t=1}^{N_frames} N_{mapped}^{(t)}}$$
- **TDE**<a href="#6" id="6ref"><sup>[6]</sup></a>，Distance between the ground-truth annotation and the tracking result；像素级别的误差计算，适用于人群跟踪；
- **OSPA**<a href="#7" id="7ref"><sup>[7]</sup></a><a href="#8" id="8ref"><sup>[8]</sup></a>，Optimal Subpattern assignment，由定位 (localization) 误差及基数 (cardinality) 误差构成，对于第 \\(t\\) 帧：
$$e^t=\left[\frac{1}{n^t}\left( \mathop{\min}_{\pi\in\Pi_n} \sum_{i=1}^{m^t} d^{(c)}(x_i^t,y_{\pi(i)}^t)^p + (n^t-m^t)\cdot c^p \right) \right]^{1/p}$$
其中，\\(n^t\\) 为目标真值与算法输出中数量较大者。\\(\\Pi_n\\) 为从 \\(n^t\\) 中取出的 \\(m\\) 个目标。\\(p\\) 为距离指数范数。其中定位截断误差为：
$$d^{(c)}(x_i^t,y_{\pi(i)}^t) = \mathop{\min}\left(c,d(x_i^t,y_{\pi(i)}^t)\right)$$
\\(c\\) 为截断参数。定位误差又由距离误差和标签误差组成：
$$d(x_i^t,y_{\pi(i)}^t=\parallel x_i^t-y_{\pi(i)}^t\parallel + \alpha \; \bar{\delta}(l_x, l_y)$$
其中 \\(\\alpha\\in[0,c]\\)，为标签误差的权重系数。如果 \\(l_x=l_y\\)，\\(\\bar{\\delta}(l_x, l_y)=0\\)，否则 \\(\\bar{\\delta}(l_x, l_y)=1\\).

　\\(\\lozenge\\)　完整性(Completeness)

- **MT**<a href="#9" id="9ref"><sup>[9]</sup></a>，Mostly Tracked，真值轨迹长度被跟踪大于80%的比例；
- **ML**<a href="#9" id="9ref"><sup>[9]</sup></a>，Mostly Lost，真值轨迹长度被跟踪小于20%的比例；
- **PT**<a href="#9" id="9ref"><sup>[9]</sup></a>，Partially Tracked，\\(1-MT-ML\\);
- **FM**<a href="#9" id="9ref"><sup>[9]</sup></a>，Fragments，ID of a target changed along a GT trajectory, or no ID(真值关联预测)；

　\\(\\lozenge\\)　鲁棒性(Robustne)

- **RS**<a href="#10" id="10ref"><sup>[10]</sup></a>，Recover from short term occlusion;
- **RL**<a href="#10" id="10ref"><sup>[10]</sup></a>，Recover from long term occlusion;

## 2.&ensp;Metrics in Industry
　　工业界的数据噪声较大，传感器配置也比较多样，不同的产品（传感器+算法），对 MOT 性能维度要求也不一样。更重要的是，评价指标应该从功能层面进行定义，在模块层面 (MOT) 进行调整及细化。可以说，工业界是以学术界为基础来设计 MOT 指标的，不同的产品没有统一的标准，但有比较通用的设计准则。  
　　这里以自动驾驶/辅助驾驶中动态目标状态估计模块为例，模块详细分析[日后再写]()。该模块的基本输入为：

- **传感器数据**，可以是图像，激光等；
- ***自定位系统***，可以是基于视觉的 VO，基于视觉-IMU 的 VINS等；

其中自定位系统能使目标状态估计在世界坐标系（惯性系）下优化，否则只能在本体（ego）非惯性系下优化，会减少一些约束量。该功能的基本输出为：

- **位置**，本体坐标系下目标的三维位置，\\(x,y,z\\)；
- **尺寸**，目标的物理尺寸大小，包括立方体的长宽高；或者图像坐标系下的像素大小；或者图像/点云下目标的 mask，即分割后的目标；
- ***朝向***，一般只考虑目标的航向角；
- **速度**，本体坐标系或世界坐标系下的三维速度，一般只考虑航向平面的速度；  

其中朝向是非必须项，有了朝向后，能更有效地进行状态优化。该模块的子模块有（注意，MOT 只包含前三者）：

- **检测(Detection)**，进行多目标检测；
- **跟踪(Tracking)**，根据上一帧结果，进行多目标跟踪；
- **数据关联(Association)**，检测结果与跟踪结果的融合，出目标的 tracklets，生成 ID；
- **状态估计(State Estimation)**，不同的方法包括不同的部分；  

　　工业界设计产品时，基本遵循自顶向下的策略：产品需求-功能需求-模块需求，层层推倒。所以我们设计评价准则时，一般会问几个问题：

- 该模块服务的产品功能，其需求及对应的指标是什么？
- 要达到功能指标，本模块的输出需要哪些指标来评测？
- 各个子模块对模块的影响是怎样的，对应需要增加哪些指标？  

这里提到了功能指标，模块指标，子模块指标三层概念。功能指标及部分模块指标是可以写入产品手册的，所以需要突出重点，易于理解；部分模块及子模块指标则主要是为了产品上工程优化迭代，这就要求这部分指标要相当细致，将模块的不足尽可能解耦，且完全暴露出来。以下通过两个例子来分析设计过程。

### 2.1.&ensp;ADAS 中的 FCW 功能
　　FCW 基本功能要求为：

- 不允许误报，尽可能不漏报；
- 在 V km/h 下，以一定的刹车加速度 a，能避免与静止的前车相碰撞；  

　　由以上两个功能需求，可确定必须的功能指标：

- （百公里）误报率；
- （百公里）漏报率；
- 观测距离，可由第二项功能要求推到出（人反应时间已知）；  

　　相应的 MOT +状态估计模块输出的指标为**各距离维度各类别维度**下的：

- 误检率；
- 漏检率；
- ID Switch；
- 定位精度；
- 速度估计精度；  

　　其中 MOT 主要涉及误检率，漏检率，ID Switch（直接影响状态估计模块）。这些指标的计算方式可以在学术界定义的基础上做进一步改进，比如漏检率，就需要体现出百公里漏报率的性能，所以可以考虑将连续 N 帧漏检的目标才归为漏检，分母可以定义为每多少帧。此外，要在各距离维度各类别维度下进行计算，这就涉及到过滤（filter）策略。对于 FCW 而言，首要关注的是本车前方近距离位置，距离维度上的功能重要程度要突显出来，类别维度也要区别对待，以便算法模块可以重点优化。

### 2.2.&ensp;自动驾驶中的动态障碍物检测功能

　　自动驾驶中动态障碍物检测的要求就高了，子模块也较为复杂，指标除了评估功能模块的性能，还需要指导迭代各子模块算法，包括本子模块的迭代比较，以及上下游模块相关指标的对比。  
　　功能需求，我们简单列举几项：

- 不允许漏检，尽可能不误检；
- 前向，后向，侧向观测距离分别要达到 x, y, z；  

　　相应的功能指标为：

- 漏检率；
- 误检率；
- 观测距离；
- 观测精度；
- 观测时延(delay)；  

　　MOT +状态估计模块输出的指标依然在**各距离维度各类别维度**下：

- 误检率；
- 漏检率；
- ID Switch；
- 定位精度；
- 尺寸，朝向，速度估计精度；
- 状态估计收敛时间；
- 一系列描述时序稳定性的指标；  

　　与前述 FCW 功能类似，只是多了较多的指标。过滤操作也做的更加细致，我们还可以将目标做重要性等级划分，比如本车道前车多少米内，那指标基本都要达到 99%+；还可以将地面区域做重要性划分（比距离维度更加细致，可以认为是三维层面），周围几米内，那误检率肯定要非常低。除了过滤策略需要仔细设计外，匹配策略也需要进一步思考。如果传感器本身精度就有限，那么匹配策略就要相应放宽。还需注意的是引入过滤策略后，FP与FN计算的细微差别，比如有个过滤条件为去除目标像素面积小于一定阈值的目标集 A，观测值与真值匹配时，如果与 A 中的目标匹配上，那么不应该记为 FP，如果没匹配上 A 中的目标，那么 A 中地目标也不应该被记为 FN。这种类似的情况逻辑要思考清楚。

## 3.&ensp;Summary

　　以上设计的出发点是，我们要承认**算法的不完美性**以及**传感器的局限性**，在工程领域，一定要首先解决主要矛盾，再打磨细节。本文还对以下内容未作进一步分析（以后有机会再写文细究）：

- 状态估计时序相关指标，描述估计的时序稳定性，也可以用于 MOT 的评估；
- 标注与过滤策略的关系，过滤策略往往依赖于标注策略；
- 各个指标的阈值确定，确定阈值也是产品中一件重要而又系统的事，有时候比指标设计更复杂；
　　

<a id="1" href="#1ref">[1]</a> Yang B, Huang C, Nevatia R. [Learning affinities and dependencies for multi-target tracking using a CRF model](https://scholar.google.com/scholar?lookup=0&q=Learning+affinities+and+dependencies+for+multi-target+tracking+using+a+CRF+model&hl=zh-CN&as_sdt=0,5&as_vis=1)[C]//CVPR 2011. IEEE, 2011: 1233-1240.  
<a id="2" href="#2ref">[2]</a> Choi W, Savarese S. [Multiple target tracking in world coordinate with single, minimally calibrated camera](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=Multiple+target+tracking+in+world+coordinate+with+single%2C+minimally+calibrated+camera&btnG=)[C]//European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2010: 553-567.  
<a id="3" href="#3ref">[3]</a> Kasturi, Rangachar, et al. [Framework for performance evaluation of face, text, and vehicle detection and tracking in video: Data, metrics, and protocol](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=Framework+for+performance+evaluation+of+face%2C+text%2C+and+vehicle+detection+and+tracking+in+video%3A+Data%2C+metrics%2C+and+protocol&btnG=) IEEE transactions on Pattern Analysis and Machine intelligence 31.2 (2008): 319-336.  
<a id="4" href="#4ref">[4]</a> Yamaguchi K, Berg A C, Ortiz L E, et al. [Who are you with and where are you going?](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=who+are+you+with+and+where+are+you+going&btnG=)[C]//CVPR 2011. IEEE, 2011: 1345-1352.  
<a id="5" href="#5ref">[5]</a> Bernardin K, Stiefelhagen R. [Evaluating multiple object tracking performance: the CLEAR MOT metrics](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=evaluating+multiple+object+tracking+performance+the+clear+mot+metrics&btnG=)[J]. Journal on Image and Video Processing, 2008, 2008: 1.  
<a id="6" href="#6ref">[6]</a> Kratz L, Nishino K. [Tracking with local spatio-temporal motion patterns in extremely crowded scenes](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=%E2%80%9CTracking+with+local+spatio-temporal+motion+patterns+in+extremely+crowded+scenes&btnG=)[C]//2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. IEEE, 2010: 693-700.  
<a id="7" href="#7ref">[7]</a> Ristic B, Vo B N, Clark D, et al. [A metric for performance evaluation of multi-target tracking algorithms](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=a+metric+for+performance+evaluation+of+multi-target+tracking+algorithms&btnG=)[J]. IEEE Transactions on Signal Processing, 2011, 59(7): 3452-3457.  
<a id="8" href="#8ref">[8]</a> Schuhmacher D, Vo B T, Vo B N. [A consistent metric for performance evaluation of multi-object filters](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=A+Consistent+Metric+for+Performance+Evaluation+of+Multi-Object+Filters&btnG=)[J]. IEEE transactions on signal processing, 2008, 56(8): 3447-3457.  
<a id="9" href="#9ref">[9]</a> Li Y, Huang C, Nevatia R. [Learning to associate: Hybridboosted multi-target tracker for crowded scene](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=Learning+to+associate%3A+Hybridboosted+multi-target+tracker+for+crowded+scene&btnG=)[C]//2009 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2009: 2953-2960.  
<a id="10" href="#10ref">[10]</a> Song B, Jeng T Y, Staudt E, et al. [A stochastic graph evolution framework for robust multi-target tracking](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&as_vis=1&q=A+stochastic+graph+evolution+framework+for+robust+multi-target+tracking&btnG=)[C]//European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2010: 605-619.
