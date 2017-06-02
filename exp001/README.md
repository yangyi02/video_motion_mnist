### Synthetic motion on synthetic images
The images are pure random values (0-1) at each pixel location.
Image resolution: 11x11x1.
motion range = 1 corresponds to 9 motion classes.
motion range = 2 corresponds to 25 motion classes.

only output one scalar for every two frames
input: two frames
output: global motion (1x1x9)

| Global motion | Training Loss | Testing Accuracy (%) |
| ------------- | ----------- | ----------- |
| scalar, motion range = 1, supervised 2 frames | 0.00 | 100 |
| scalar, motion range = 2, supervised 2 frames | 0.00 | 100 |
| scalar, motion range = 1, unsupervised 2 frames | 0.00 | 100 |
| scalar, motion range = 2, unsupervised 2 frames | 0.00 | 100 |
