global motion but using fully convolutional network to predict every pixel motion
use 2 frames, the second frame is the supervision
does not work

### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
Image resolution: 11x11x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.

input: two frames (11x11x2)
output: motion mask (11x11x9)

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.01 | 99 |
| mask, motion range = 2, supervised 2 frames | 0.01 | 74 |
| mask, motion range = 1, unsupervised 2 frames | 0.40 | 11 (random guess) |
| mask, motion range = 2, unsupervised 2 frames | 0.30 | 4 (random guess) |
