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
| scalar, motion range = 1, supervised 2 frames | 95 |
| scalar, motion range = 2, supervised 2 frames | |
| scalar, motion range = 3, supervised 2 frames | 82 |
| scalar, motion range = 5, supervised 2 frames | |
| scalar, motion range = 1, unsupervised 3 frames | 69 |
| scalar, motion range = 2, unsupervised 3 frames | |
| scalar, motion range = 3, unsupervised 3 frames | |
| scalar, motion range = 5, unsupervised 3 frames | |

Only get to 69% optical flow estimation accuracy, because most of the MNIST images has 0 background
And 0 background cannot estimate flow

