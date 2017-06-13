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
Image resolution: 32x32x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: four frames (i.e. 32x32x4)
output: two local motion (i.e. 32x32x9x2) and two disappear map (i.e. 32x32x1x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet |  |
| motion range = 2, supervised 2 frames, UNet |  |
| motion range = 3, supervised 2 frames, UNet | 99 |
| motion range = 5, supervised 2 frames, UNet |  |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet, learning rate = 0.01 | 92 |
| motion range = 3, unsupervised 3 frames, UNet, learning_rate = 0.001 | 94 |
| motion range = 5, unsupervised 3 frames, UNet | |

Motivation

This experiment contains additional disappear label.
The motivation is to use later frames to predict the occluded part in the previous frames.

Take Home Message:

This actually works.
Attention visualization and disappear visualization looks very reasonable.
But the motion estimation accuracy is not as high as by adding disappear class inside motion class.
The loss also suggest that it is better to directly add the disappear class inside motion class instead of using an extra convolutional layer. Why?
This is because we decompose the motion and disappear class, such that there will be some cross-labels that is predicted as both disappear and with some motion (Overall 49 * 2 = 98 classes instead of 49 + 1 = 50 clases) 
This is originally supposed to be good but makes motion estimation harder.

Learning rate = 0.001 is much better than learning rate = 0.01.
The reason is because this converges much slower hence the disappear class is not predicted in overall, hence it is equivalent to predict every pixel as not disappeared.

Overall it is still a better idea to add an extra class to motion prediction instead of estimating each other independently.
