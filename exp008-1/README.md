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
Add extra layer to predict disappeared pixels, instead of adding an extra class, this will helps on allowing the system to both predict a motion for each pixel and a disappear label for each pixel.
This will solve the previous bad assumption that only foreground moves.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (i.e. 28x28x2)
output: local motion (i.e. 28x28x9) + disappear label (i.e. 28x28x1)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, supervised 2 frames, UNet | |
| motion range = 2, supervised 2 frames, UNet | |
| motion range = 3, supervised 2 frames, UNet | 99 |
| motion range = 5, supervised 2 frames, UNet | |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | 94 |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

Adding extra disappeared labels definitely help, but the motion predicted in the disappear pixel locations are incorrect. Why?
The incorrectness at the disappear locations lower the overall motion prediction accuracy. However, the framework is more neat.

Explanation:
The incorrectness of motion prediction at the disappear locations is due to we first consider disappearance.
And if the pixel is predicted as disappear, it is possible to estimate different motions for it, rather than predict it correctly.
The solution is to use context by predicting motion for more previous frames and more later frames.
