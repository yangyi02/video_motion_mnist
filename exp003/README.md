global motion but using fully convolutional network to predict every pixel motion
use 3 frames instead of 2 frames, the third frame is the supervision
it works

### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.

input: two frames (i.e. 11x11x2)
output: motion mask (i.e. 11x11x9)
unsupervised output: third frame (i.e. 11x11x1)

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 1, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 2, supervised 2 frames | 0.00 | 100 |
| mask, motion range = 3, supervised 2 frames | 0.06 | 98 |
| mask, motion range = 1, unsupervised 3 frames | | 100 |
| mask, motion range = 2, unsupervised 3 frames | | 99 |
| mask, motion range = 3, unsupervised 3 frames | | 96 |

when training unsupervised with larger motion range = 3, image size is important

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| mask, motion range = 3, image size = 11 |  | 61 |
| mask, motion range = 3, image size = 21 |  | 94 |
| mask, motion range = 3, image size = 28 |  | 96 |
