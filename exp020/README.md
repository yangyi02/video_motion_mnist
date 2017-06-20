continuous motion prediction, with spatial transformer networks.

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
output: one local motion (i.e. 32x32x2) and one disappear map (i.e. 32x32x1)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | |
| motion range = 2, supervised 2 frames, UNet | |
| motion range = 3, supervised 2 frames, UNet | |
| motion range = 5, supervised 2 frames, UNet | |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

Somehow the pytorch spatial transformer network training does not work!!!
The loss does not decrease.
Something is wrong.

