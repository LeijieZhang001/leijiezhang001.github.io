---
title: '[paper_reading]-"Learning to See in the Dark"'
date: 2020-04-03 11:03:20
tags: ["Low-Light Image Enhancement", "Deep Learning"]
categories: Low-Light Image Enhancement
mathjax: true
---
　　无监督低光照图像增强更有应用价值，{% post_link Unsupervised-Low-Light-Image-Enhancement Unsupervised Low Light Image Enhancement%} 中介绍了几种无监督方法。本文则是有监督方法，但是值得一读。在 Sensor，曝光时间，光圈，ISO 等(在线调节通过 AE 完成)确定后，图像低光照下曝光不足主要是因为 ISP 过程对图像的亮度矫正不理想。本文直接重构 ISP 过程，对 Raw 图像进行一系列操作，以增强亮度。


## 1.&ensp;算法过程
<img src="ISP.png" width="90%" height="90%" title="图 1. Raw Image Processing Pipeline 对比">
　　如图 1. 所示，传统 ISP 过程包括：White Balance, Demosaic, Denoise/Sharpen, Color Space Conversion, Gamma Correction(与亮度变化相关)等。L3 与 Burst 是其它 ISP pipeline 学习的方法，本文网络算法过程如图 1.b 所示，首先提取 RGB sensor 值并放大一定比例(该放大系数用来控制最终增强的曝光级别)，然后经过网络层，最终输出全尺寸的 RGB 图像。  
　　训练数据采集自室内静态场景，每对数据由短曝光的低光照图像与长曝光的标签图像构成，由此可进行有监督训练。

## 2.&ensp;Reference
<a id="1" href="#1ref">[1]</a> Chen, Chen, et al. "Learning to see in the dark." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
