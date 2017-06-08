global motion with MNIST dataset 
only output one scalar for every two frames

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
| scalar, motion range = 2, supervised 2 frames | 100 |
| scalar, motion range = 3, supervised 2 frames | 100 |
| scalar, motion range = 5, supervised 2 frames | 98  |
| scalar, motion range = 1, unsupervised 2 frames | 100 |
| scalar, motion range = 2, unsupervised 2 frames | |
| scalar, motion range = 3, unsupervised 2 frames | 92 |
| scalar, motion range = 5, unsupervised 2 frames | 73 |

Net3 gets better results than Net2, Net2 gets better results than Net
