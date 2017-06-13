local motion using fully convolutional network to predict every pixel motion
MNIST dataset
Digit has motion, background has motion
use 3 frames instead of 2 frames, the third frame is the supervision
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
Add extra 1 dimension for predicting disappeared pixels, these pixels should be able to be predicted by looking at the motion and image segmentation.
At this moment, we assume only foreground moves, hence the disappeared pixels are those background close to the foreground moving direction.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 28x28x2)
output: two local motion (i.e. 32x32x9x2) and two disappear map (i.e. 32x32x1x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | |
| motion range = 2, supervised 2 frames, UNet | |
| motion range = 3, supervised 2 frames, UNet, num_hidden = 32 | 91 |
| motion range = 3, supervised 2 frames, UNet, num_hidden = 64 | 95 |
| motion range = 5, supervised 2 frames, UNet | |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 92 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

It works!!!

When background also moves, then this time modeling disappear as an extra label space (49 * 2 = 98 classes) significantly better than previously adding extra disappear class (49 + 1 = 50 classes).
