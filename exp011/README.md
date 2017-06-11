local motion using fully convolutional network to predict every pixel motion
MNIST dataset
MNIST digit has motion, background has no motion
use 5 frames instead of 3 frames, the third frame is the supervision
hence use both previous 2 frames and last 2 frames to predict the middle frame
the potential benefit of this is to handle occluded pixel prediction, because if the motion is consistent, then pixels in previous and later frames can recover the middle frame 
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: four frames (i.e. 28x28x4)
output: two local motion (i.e. 28x28x9x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | 100 |
| motion range = 2, supervised 2 frames, UNet | 100 |
| motion range = 3, supervised 2 frames, UNet | 98 |
| motion range = 5, supervised 2 frames, UNet | 96 |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 93 |
| motion range = 5, unsupervised 3 frames, UNet | |

Motivation

This experiment differs from exp008 at attention generation.
In exp008, the attention comes from motion with convolution.
Here, the attention comes from segmentation combination, where segmentation before motion moving is defined as all one at every pixel location.
The motivation is hoping to learn a better attention map by manually design the mechanism.

Take Home Message:

The attention visualization is defnitely much better, however the overall motion estimation accuracy does not improve, and the reconstruction loss still not zero.
