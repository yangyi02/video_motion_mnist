local motion using fully convolutional network to predict every pixel motion
MNIST dataset
MNIST digit has motion, background has no motion
use 5 frames instead of 3 frames, the third frame is the supervision
hence use both previous 2 frames and last 2 frames to predict the middle frame
the potential benefit of this is to handle occluded pixel prediction, because if the motion is consistent, then pixels in previous and later frames can recover the middle frame 
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

input: four frames (i.e. 28x28x4)
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
| motion range = 5, unsupervised 3 frames, UNet |  |

Take Home Message:

It turns out by adding bidirection inputs, the reconstructed image looks better, but the overall flow estimation becomes worse. This is because as long as either direction estimates correctly, they can reconstruct the final signal correctly. This is a problem.
Hence why not just estimate from one direction since it already gives a very accurate prediction.
