local motion using fully convolutional network to predict every pixel motion
MNIST dataset
MNIST digit has motion, background has no motion
use 3 frames, do both forward and backward, hence both the first frame and the third frame are the supervision
however, when predicting first frame, only use second and third. when predicting third frame, only use first and second.
then add an extra constraints to ask the two predicted motion are completely inverse
use L1 loss instead of L2 loss for unsupervised learning

### Synthetic motion on synthetic images
The images are randomly sampled from MNIST dataset.
MNIST contains 50000 training images and 10000 testing images.
Image resolution: 28x28x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: three frames (i.e. 28x28x4)
output: two local motion (i.e. 28x28x9x2)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | 100 |
| motion range = 2, supervised 2 frames, UNet | 100 |
| motion range = 3, supervised 2 frames, UNet | 98 |
| motion range = 5, supervised 2 frames, UNet | 96 |
| motion range = 1, unsupervised 3 frames, UNet | 98 |
| motion range = 2, unsupervised 3 frames, UNet | 97 |
| motion range = 3, unsupervised 3 frames, UNet | 94 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

Although this seems a more sophisticated design than puring estimating the third frame (exp007), the final results do not seem improved.
There are still some images with slightly incorrectly estimated motion.
The optimal cross validation for hyperparameter in the motion contrary loss is 0.0001 which is almost 0, hence means this term is somewhat useless.
