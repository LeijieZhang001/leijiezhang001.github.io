---
title: OctoMap
date: 2020-04-23 10:23:05
tags: ["SLAM", "Point Cloud", "Mapping"]
categories: SLAM
mathjax: true
---
　　地图是机器人领域非常重要的模块，也可以认为是自动驾驶保障安全的基础模块。根据存储建模类型，地图可分为拓扑地图，栅格地图，点云地图等。{%post_link Grid-Mapping Grid-Mapping %} 就是一种能在线检测静态障碍物的栅格地图。自动驾驶领域，地图的用处有：

- **高精度定位**，一般是 3D 栅格地图，但是栅格中近似存储点云原始信息；
- **路径规划**，不同规划算法依赖不同地图，自动驾驶中比较靠谱又简单的规划算法一般依赖拓扑地图，俗称高精度语义地图，描述一些车道线等路面拓扑关系；而在室内或低速无道路信息场景，则会用如 \\(A ^ * \\) 算法在栅格地图上进行路径规划；
- **辅助感知检测未定义类别的障碍物**，有人称之为静态地图，一般是 2.5D 栅格地图，图层可以自定义一些语义信息；

下游不同模块对不同存储方式的利用效率是不同的，所以需要针对不同下游任务设计不同地图建模方式。本文<a href="#1" id="1ref"><sup>[1]</sup></a>介绍了一种基于八叉树的栅格地图建模方法。  
　　对于机器人而言，类似 {%post_link Grid-Mapping Grid-Mapping %} 能建模 FREE，OCCUPIED，UNMAPPED AREAS 的地图是信息量比较丰富的，但是 Grid-Mapping 是 2D 的。这里对 3D 地图有以下要求：

- **Probabilistic Representation**  
测量都会有不确定性，这种不确定性需要用概率表征出来；另外多传感器融合也需要基于概率的表示；
- **Modeling of Unmapped Areas**  
对机器人导航而言，显式得表示哪些区域是观测未知的也非常重要；
- **Efficiency**  
地图构建与存储需要非常高效，一般而言，地图的内存消耗会是瓶颈；

<img src="maps.png" width="90%" height="90%" title="图 1. Different Representations of Maps">
　　如图 1. 所示，原始点云地图信息量丰富，但是不能结构化存储；Elevation Maps 与 Multi-level Surface Maps 虽然高效，但是不能表征未观测的区域信息。OctoMap 可以认为是 {%post_link Grid-Mapping Grid-Mapping %} 的 3D 版本，信息量丰富且高效。

## 1.&ensp;OctoMap Mapping Framework

### 1.1.&ensp;Octrees
<img src="OctoMap.png" width="40%" height="40%" title="图 2. 八叉树地图存储">
　　如图 2. 所示，八叉树是将空间递归得等分成八份(QuadTree 四叉树则等分为四份)，每个节点可以存储 Occupied，Free，Unknown 信息(Occupied 概率即可)。此外，如果子节点的状态都一样，那么可以进行剪枝，只保留大节点低分辨率的 Voxel，达到紧凑存储的目的。  
　　时间复杂度上，对于有 \\(n\\) 个节点，深度为 \\(d\\) 的八叉树，那么单次查询的时间复杂度为 \\(\\mathcal{O}(d)=\\mathcal{O}(\\mathrm{log}\\,n)\\)；遍历节点的时间复杂度为 \\(\\mathcal{O}(n)\\)。\\(d = 16, r = 1cm\\)，可以覆盖 \\((655.36m)^3\\)的区域。

### 1.2.&ensp;Probabilistic Sensor Fusion
　　时序概率融合也是基于贝叶斯滤波，详见 {%post_link Grid-Mapping Grid-Mapping %}，只不过这里是 3D Mapping，作 Raycasting 的时候采用 {%post_link paper-reading-What-You-See-is-What-You-Get-Exploiting-Visibility-for-3D-Object-Detection What You See is What You Get%} 中提到的 Fast Voxel Traversal 算法。实际应用中，一般都会采用上下界限制概率值，这种限制也能提高八叉树的剪枝率。

### 1.3.&ensp;Multi-Resolution Queries
　　由于八叉树的特性，OctoMap 支持低于最高分辨率的 Voxel 概率查询，即父节点是子节点的平均概率，或是子节点的最大概率:
$$
\bar{l}(n)=\frac{1}{8}\sum _ {i=1}^8 L (n _ i)\\
\hat{l}(n)=\max\limits _ iL(n _ i)
\tag{1}$$
其中 \\(l\\) 是测量模型下概率的 log-odds 值。

## 2.&ensp;Implementation Details & Statics

### 2.1.&ensp;Memory-Efficient Node & Map File Generation
<img src="save.png" width="60%" height="60%" title="图 3. Node Memory Consumption and Serialization">
　　如图 3. 左图所示，每个节点只分配一个 float 型的数据存储以及指向子节点地址数组的地址指针(而不是直接包含子节点地址的指针)，只有存在子节点时，才会分配子节点的地址数组空间。由此在 32-bit 系统中(4 字节对齐)，每个父节点需要 40B，子节点需要 8B；在 64-bit 系统中(8 字节对齐)，每个父节点需要 80B(\\(4+9\\times 8\\))，子节点需要 16B(\\(4+8)\\)。  
　　地图存储需要在信息量损失最小的情况下进行压缩。如图 3. 右图所示，存储序列化时，每个叶子节点总共需要 4B 概率值，不需要状态量；每个父节点总共需要 2B，表示 8 个子节点的 2bit 状态量(貌似与论文有出入，其不是最优的压缩)。在这种压缩方式下，大范围地图的存储大小一般也能接受。根据存储的地图重建地图时，只需要知道坐标原点即可。

### 2.2.&ensp;Accessing Data & Memory Consumption
<img src="memusage1.png" width="60%" height="60%" title="图 4. Memory Usage VS. Scan Num.">
　　Freiburg 建图大小为 \\((202\\times 167\\times 28) m^3\\)，如图 4. 所示，随着点云扫描区域扩大，OctoMap 表示方式能有效降低建图大小。
<img src="memusage2.png" width="60%" height="60%" title="图 5. Memory Usage VS. Resolution">
　　图 5. 则说明建图大小与分辨率的关系。
<img src="inserttime.png" width="60%" height="60%" title="图 6. Insert Date Time VS. Resolution">
<img src="traversetime.png" width="60%" height="60%" title="图 7. Traverse Data Time VS. Depth">
　　图 6. 显示了往图中插入一个节点所需时间，1000 个节点在毫秒级；图 7. 显示了遍历所有节点所需的时间，基本也在毫秒级。
<img src="compress.png" width="60%" height="60%" title="图 8. Compression Ratio">
　　通过限制概率上下界，可以剪枝压缩图，用 KL-diverge 来评估压缩前后图的分布相似性，图 8. 显示了压缩比与网络大小及相似性的关系。

### 2.3.&ensp;Some Strategies
<img src="case.png" width="60%" height="60%" title="图 9. Corner Case Handle">
　　如图 9. 所示，前后帧位姿的抖动，会导致 Occupied 持续观测的不稳定，所以需要一些领域约束策略来保证 Occupied 的稳定观测。这种类似的策略在 {% post_link Grid-Mapping Grid-Mapping %} 工程实现中也需要采用，因为实际的 Pose 肯定会有噪声，导致同一目标的栅格前后有一定概率不能完全命中。

## 3.&ensp;ReThinking
　　对于自动驾驶来说，高度方向的范围不需要很大，甚至四叉树足矣，如果采用八叉树，那么需要将高度方向的分辨率降低，从而更加紧凑的构建地图。  
　　此外自动驾驶肯定是需要大范围建图的，如平方千公里级别。所以切片式的地图存储与查询就显得尤为重要，换句话说，需要动态得载入局部地图，这就有两种思路：

- 动态载入完全局部地图  
要求前后局部地图有一定的重叠，通过索引式的存储可以不存储重叠区域的地图信息；
- 动态载入部分局部地图  
随着机器人本体的运动，实时动态载入前方更远处的地图，丢掉后方远处的历史地图。这对在线地图结构的灵活性要求比较高，如果基于八叉树，那么需要作片区域剪枝及插入的操作，效率不一定高；

　　在自动驾驶领域，目前用于高精度定位的栅格地图与用于 PNC 规划控制的拓扑地图(高精地图)已经比较成熟；而用于环境感知的静态语义地图还没形成大范围的共识。不管从工程实现效果及效率上，还是语义信息描述定义上，还需作很多探索与实践。比如，可以定义最底层的语义信息：地面高度，此外也可以把车道线信息打到栅格图层中去(但是可能加大对 PNC 的搜索计算量)，等等。所以可能最优的存储查询方式并不是八叉树，**可能还是栅格化后并对每个栅格哈希化，牺牲一定的内存空间，然后作 \\(O(1)\\) 的快速插入与查询**。

## 4.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Hornung, Armin, et al. "OctoMap: An efficient probabilistic 3D mapping framework based on octrees." Autonomous robots 34.3 (2013): 189-206.
