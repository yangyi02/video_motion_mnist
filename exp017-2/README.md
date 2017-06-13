local motion using fully convolutional network to predict every pixel motion
MNIST dataset
Digit has motion, background has motion
use 5 frames instead of 3 frames, the third frame is the supervision
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
Add extra layer for predicting disappeared pixels, these pixels should be able to be predicted by looking at the motion and image segmentation.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 28x28x4)
output: two local motion (i.e. 32x32x9x2) and two disappear map (i.e. 32x32x1x2) and two attention map

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | |
| motion range = 2, supervised 2 frames, UNet | |
| motion range = 3, supervised 2 frames, UNet, num_hidden = 32 | 91 |
| motion range = 3, supervised 2 frames, UNet, num_hidden = 64 | 95 |
| motion range = 5, supervised 2 frames, UNet | |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 94 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

It works!!!

Using 5 frames with disappear and attention modeling significantly better than using 3 frames or without disappear modeling, on data where background also have motion.

