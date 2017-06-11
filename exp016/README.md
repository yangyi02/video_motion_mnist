local motion using fully convolutional network to predict every pixel motion
MNIST dataset
MNIST digit has motion, background has no motion
use 3 frames instead of 5 frames, the second frame is the supervision
hence use both previous 1 frame and last 1 frame to predict the middle frame
the potential benefit of this is to handle occluded pixel prediction, because if the motion is consistent, then pixels in previous and later frames can recover the middle frame 
another potential benefit of this is the motion consistency because we only use 3 frames
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

So sad it works not as well as 5 frames ...

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 32x32x1.
motion range = 1 corresponds to 9 + 1 motion classes.
motion range = 2 corresponds to 25 + 1 motion classes.
motion range = 3 corresponds to 49 + 1 motion classes.
motion range = 5 corresponds to 121 + 1 motion classes.

input: two frames (i.e. 32x32x4)
output: two local motion (i.e. 32x32x10x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet |  |
| motion range = 2, supervised 2 frames, UNet |  |
| motion range = 3, supervised 2 frames, UNet | 99 |
| motion range = 5, supervised 2 frames, UNet |  |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 96 |
| motion range = 5, unsupervised 3 frames, UNet | |

Motivation

This experiment contains additional motion class (disappear class).
The motivation is to use later frames to predict the occluded part in the previous frames.

Take Home Message:

This works but not as good as using 4 frames to predict the middle frame.
Attention visualization also looks not as reasonable as using 4 frames. 
Potential reason is because the motion between frame 1 and frame 3 is twice larger, hence harder to match correspondence!!!
Probably with a larger motion kernel can solve this problem.
