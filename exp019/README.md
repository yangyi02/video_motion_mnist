local motion using fully convolutional network to predict every pixel motion
MNIST dataset
MNIST digit has motion, background also has motion
use 3 frames instead of 5 frames, the second frame is the supervision
hence use both previous 1 frame and last 1 frame to predict the middle frame
the potential benefit of this is to handle occluded pixel prediction, because if the motion is consistent, then pixels in previous and later frames can recover the middle frame 
another potential benefit of this is the motion consistency because we only use 3 frames
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 32x32x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 32x32x4)
output: two local motion (i.e. 32x32x10x2) and two disappear map (i.e. 32x32x1x2) and two attention map (i.e. 32x32x1x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet |  |
| motion range = 2, supervised 2 frames, UNet |  |
| motion range = 3, supervised 2 frames, UNet | 94 |
| motion range = 5, supervised 2 frames, UNet |  |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 48 |
| motion range = 5, unsupervised 3 frames, UNet | |

Motivation

This experiment again test the performance on using only 3 frames to replace 5 frames.
The potential benefit is the computational speed, particularly when the low level convolution filters will be shared.

Take Home Message:

Most of the failures happen at the background.
I guess the reason is because the background moves much larger (6 pixels motion) when we use frame 1 and frame 3 as the input.
I think either we should use more deeper model such as stacked hourglass network, or we should use multiple scale to handle this.
The multiple scale could be a potential important direction to deal with larger motion.

