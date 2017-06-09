local motion but using fully convolutional network to predict every pixel motion
MNIST dataset
3 motions: 2 on two MNIST digits, 1 on background
use 3 frames instead of 2 frames, the third frame is the supervision
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning
Supervised: 96%
Unsupervised: 83% (L2 loss)
Unsupervised: 94% (L1 loss)