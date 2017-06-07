global motion but using fully convolutional network to predict every pixel motion
use 2 frames, the second frame is the supervision
does not work

### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
Image resolution: 11x11x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.

input: two frames (11x11x2)
output: motion mask (11x11x9)
unsupervised output: second frame (11x11x1)

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 2, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 3, supervised 2 frames | 0.06 | 98 |
| mask, motion range = 1, unsupervised 2 frames | 0.40 | 11 (random guess) |
| mask, motion range = 2, unsupervised 2 frames | 0.16 | 4 (random guess) |

batch size = 64 is better than batch size = 32, probably because of batch normalization
learning rate = 0.01 is better than learning rate = 0.001, probably because of Adam optimization
learning rate = 0.1 does not work
when training mostion range = 3, train epoch = 10000 is better than train epoch = 5000, probably because of fine detailed convergence
most predictions errors happend at the cornor or border of image
learning rate decay does not seem helpful, probably because of Adam optimization algorithm
