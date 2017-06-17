local motion using fully convolutional network to predict every pixel motion
First attempt on real world video, google robot video dataset
https://sites.google.com/site/robotprediction/
Static background: robot arm has motion, background has no motion

### Real motion on real images
The images are randomly sampled from robot dataset.
Robot contains 761250 training images, organized in a way below:
793/9/23.jpg
Image resolution: 512x640x3.
But we downsample to 64x80x3.
Add extra 1 dimension for predicting disappeared pixels, these pixels should be able to be predicted by looking at the motion and image segmentation.
At this moment, we assume only foreground moves, hence the disappeared pixels are those background close to the foreground moving direction.
motion range = 1 corresponds to 9+1 motion classes.
motion range = 2 corresponds to 25+1 motion classes.
motion range = 3 corresponds to 49+1 motion classes.
motion range = 5 corresponds to 121+1 motion classes.

input: multiple previous frames (i.e. 28x28x3x5)
output: local motion (i.e. 28x28x10) and next frame (i.e. 28x28x3)

| Local motion | Training Loss (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

First attempt failed.
Training loss are very high. After visualization I find all pixels are predicted with 0 motion.
The reason is because in the video, many have very small motion (or almost 0 motion).
And because I use discrete class labels to train, it makes much harder.
