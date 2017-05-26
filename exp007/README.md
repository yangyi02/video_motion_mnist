local motion but using fully convolutional network to predict every pixel motion
MNIST dataset
2 motions: 1 on MNIST digit, 1 on background
use 3 frames instead of 2 frames, the third frame is the supervision
adding noise to MNIST background
use L1 loss instead of L2 loss for unsupervised learning
Supervised: 97%
Unsupervised: 93% (L2 loss)
Unsupervised: 94% (L1 loss)