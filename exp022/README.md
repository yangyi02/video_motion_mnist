local motion using fully convolutional network to predict every pixel motion
Third attempt mpii human pose video dataset
http://human-pose.mpi-inf.mpg.de/#download

### Real motion on real images
The images are the batch 1 from mpii human pose dataset.
Mpii batch 1 contains 41821 training images, organized in a way below:
1/013688563/00000068.jpg
Image resolution: 480xwidthx3.
But we downsample to 64xwidthx3.
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

