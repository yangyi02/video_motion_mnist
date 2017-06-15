local motion using fully convolutional network to predict every pixel motion
MNIST dataset
Digit has motion, background has motion

USE 2 FRAMES, the second frame is the supervision, adding smoothness loss

adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
Add extra 1 dimension for predicting disappeared pixels, these pixels should be able to be predicted by looking at the motion and image segmentation.
At this moment, we assume only foreground moves, hence the disappeared pixels are those background close to the foreground moving direction.
motion range = 1 corresponds to 9 motion classes.  motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 28x28x2)
output: one local motion (i.e. 32x32x9) and one disappear map (i.e. 32x32x1)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, unsupervised 2 frames, UNet | |
| motion range = 2, unsupervised 2 frames, UNet | |
| motion range = 3, unsupervised 2 frames, UNet | 3 |
| motion range = 5, unsupervised 2 frames, UNet | |

Take Home Message:

Smoothness loss does not help!!!
The visualization explains everything. 
Although motion estimation is completely wrong, the reconstruction error is very low.
