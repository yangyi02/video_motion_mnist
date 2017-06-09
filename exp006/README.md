global motion but using fully convolutional network to predict every pixel motion
MNIST dataset
use 3 frames instead of 2 frames, the third frame is the supervision

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 28x28x2)
output: global motion (i.e. 1x1x9)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| scalar, motion range = 1, supervised 2 frames | 100 |
| scalar, motion range = 2, supervised 2 frames | 99 |
| scalar, motion range = 3, supervised 2 frames, FullyConvNet | 97 |
| scalar, motion range = 5, supervised 2 frames, FullyConvNet | 50 |
| scalar, motion range = 5, supervised 2 frames, FullyConvResNet | 71 |
| scalar, motion range = 3, supervised 2 frames, UNet | 97 |
| scalar, motion range = 5, supervised 2 frames, UNet | 94 |
| scalar, motion range = 3, supervised 2 frames, UNet3, 32 resolution | 99 |
| scalar, motion range = 5, supervised 2 frames, UNet3, 32 resolution | 96 |
| scalar, motion range = 1, unsupervised 3 frames, UNet3, 32 resolution | 100 |
| scalar, motion range = 2, unsupervised 3 frames, UNet3, 32 resolution | 99 |
| scalar, motion range = 3, unsupervised 3 frames, UNet3, 32 resolution | 98 |
| scalar, motion range = 5, unsupervised 3 frames, UNet3, 32 resolution | 49 |

Take Home Message:

Get to much better optical flow estimation accuracy after adding noise to MNIST background
noise level also affect the texture significance hence affect the flow estimation on the background

When estimating large motion, UNet works better than FullyConvResNet, FullyConvResNet works better than FullyConvNet
UNet is definitely the best choice at this moment, after resizing image size to 32, deeper UNet converges much faster 

Unsupervised motion estimation accuarcy heavily depends on the relative ratio between motion range and image size
If the motion range is large (i.e. 5) and image size is small (i.e. 32), it cannot learn good motion estimation with unsupervised reconstruction
We hence need a larger image size, or small motion w.r.t. image size (probably small than 1/10)

batch size = 32 also works well when image is larger 
