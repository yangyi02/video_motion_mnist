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
Add extra 1 dimension for predicting disappeared pixels, these pixels should be able to be predicted by looking at the motion and image segmentation.
At this moment, we assume only foreground moves, hence the disappeared pixels are those background close to the foreground moving direction.
motion range = 1 corresponds to 9+1 motion classes.
motion range = 2 corresponds to 25+1 motion classes.
motion range = 3 corresponds to 49+1 motion classes.
motion range = 5 corresponds to 121+1 motion classes.

input: two frames (i.e. 28x28x2)
output: local motion (i.e. 28x28x10)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | 100 |
| motion range = 2, supervised 2 frames, UNet | 100 |
| motion range = 3, supervised 2 frames, UNet | 98 |
| motion range = 5, supervised 2 frames, UNet | 96 |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet, batch size 64 | 96 |
| motion range = 3, unsupervised 3 frames, UNet, batch size 32 | 95 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

Adding extra disappeared class definitely help, although improvement is only 1%, but it's very important.
batch size 64 is still better than batch size 32, improve also 1%.
