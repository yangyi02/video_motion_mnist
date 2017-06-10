local motion using fully convolutional network to predict every pixel motion
MNIST dataset
Static background: digit has motion, background has no motion
use 3 frames instead of 2 frames, the third frame is the supervision
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

input: two frames (i.e. 28x28x2)
output: local motion (i.e. 28x28x9)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | 100 |
| motion range = 2, supervised 2 frames, UNet | 100 |
| motion range = 3, supervised 2 frames, UNet | 98 |
| motion range = 5, supervised 2 frames, UNet | 96 |
| motion range = 1, unsupervised 3 frames, UNet | 97 |
| motion range = 2, unsupervised 3 frames, UNet | 97 |
| motion range = 3, unsupervised 3 frames, UNet | 95 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

Unsupervised Training can get to very close performance on local motion estimation compared to supersied learning.
But if suffers at the occlusion boundary, where if we only use feedforward through time, we cannot reconstruct the occluded pixels in the background.
Hence we observed very thin object segments instead of full object segments.
This motivates us to use both previous frames to do feedforward through time and later frames to do feedback through time to reconstruct the middle frame.
