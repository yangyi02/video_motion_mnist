
# Unsupervised Learning Segmentation Features from Small Motion Videos

## Milestones

/* 计划中的项目可以写上预计开展的时间 */

## Introduction

### Background and Motivation

This project is aiming to find out whether it's possible to learn image segmentation features from videos.

We follow the idea of sequentially compute features from low level to high level. 
For the low level features, we hypothesize it could be motion and depth.
For the high level features, we hypothesize it could be object segmentation.

### Related Work and Their Problems

/* 有没有类似的项目？它们有哪些缺陷？*/

### Our Method

Our method is based on predicting the next frames which is the "only" supervised signals for learning a deep neural network.
We design the network structure to particulary reconstruct the next frame using all previous frames and camera motion.
In order to reconstruct well, the neural network needs to estimate the optical flow which is an effect caused by object motion, camera motion and depth together.

### Our Goal

Our goal is to train a neural network that can predict foreground object mask.

### Advantage of Our Method

/* 我们做这件事情有什么方面的优势？ */

### Expected Results

/* 描述做完之后希望达到的愿景，关键的价值产出是什么，可以从以下几个方面来考虑：短期收益、长期收益、局部收益、整体收益 */

### Risks

/* 有哪些方面的风险 */

### Time Cost

/* 可以从人力/时间、资源（如服务器）等几个角度来描述 */

### Performance Evaluation

/* 如何在项目中期和结束时衡量效果 */

### 产出形式

/* 最终的产出是什么，如工具、库、平台、插件、文档等 */

## Authors
* Wang Yang
* Yi Yang
* Xu Wei 

## Future Directions

/* 用于填写其它项目相关的信息和后续变化，比如推广和使用情况等 */

# Progress

## Synthetic Experiments

### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
Image resolution: 11x11x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| scalar, motion range = 1, supervised 2 frames | 0.00 | 100 |
| scalar, motion range = 2, supervised 2 frames | 0.00 | 100 |
| scalar, motion range = 1, unsupervised 2 frames | 0.00 | 100 |
| scalar, motion range = 2, unsupervised 2 frames | 0.00 | 100 |
| mask, motion range = 1, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 2, supervised 2 frames | 0.00 | 65 |
| mask, motion range = 1, unsupervised 2 frames | 0.40 | 11 (random guess) |
| mask, motion range = 2, unsupervised 2 frames | 0.30 | 4 (random guess) |
| mask, motion range = 1, unsupervised 3 frames | 0.00 | 100 |
| mask, motion range = 2, unsupervised 3 frames | 5 | 88 |

### Synthetic motion on real images
We use [Mnist](http://www.cs.toronto.edu/~emansim/datasets/mnist.h5) dataset to work.
Related work can be found on [github](https://github.com/emansim/unsupervised-videos).
The imgaes are handwritten digits with 10 classes.
Image resolution: 28x28x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| scalar, motion range = 1, supervised 2 frames | 0.00 | 100 |
| scalar, motion range = 1, unsupervised 2 frames | 0-55 | 99 |
| mask, motion range = 1, supervised 2 frames | 0.00 | 75 |
| mask, motion range = 1, unsupervised 3 frames | 0-10 | 50 |

If add a little noise (0-0.2) to the background.

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 1, unsupervised 3 frames | 0-10 | 96 |

Now we try two motions in one video.
We generate one motion for the foreground and another motion for the background.

| Two motions | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.1-0.2 | 96 |
| mask, motion range = 1, unsupervised 3 frames | 6-10 | 92 |

Then we try multiple motions in multiple digits in one video.

| Multiple motions | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.1 | 97 |
| mask, motion range = 1, unsupervised 3 frames | 400-600 | 85 |

Then if we train with L1 loss, even the unsupervised learning gets much better.

| Multiple motions | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.1 | 97 |
| mask, motion range = 1, unsupervised 3 frames | 1000-1500 | 95 |


## Real Data Experiments
