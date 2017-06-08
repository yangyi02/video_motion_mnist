only output one scalar for every two frames

### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
Image resolution: 11x11x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.
motion range = 3 corresponds to 49 motion classes.
motion range = 5 corresponds to 121 motion classes.

input: two frames (11x11x2)
output: global motion (1x1x9)

| Global motion | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| scalar, motion range = 1, supervised 2 frames | 100 |
| scalar, motion range = 2, supervised 2 frames | 100 |
| scalar, motion range = 3, supervised 2 frames | 99 |
| scalar, motion range = 5, supervised 2 frames | 98 |
| scalar, motion range = 1, unsupervised 2 frames | 100 |
| scalar, motion range = 2, unsupervised 2 frames | 99 |
| scalar, motion range = 5, unsupervised 2 frames | 98 |

Net3 gets better results than Net2, Net2 gets better results than Net
