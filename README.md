## Inspired by:  https://github.com/LS-Computer-Vision/sudoku-solver-2
Most of the resources used to solve the problem has been taken from the above link.

# Sudoku Solver 

## Implementation Details
**How was the board extracted?**
This was done by finding all the contours in the grayscaled image, and from there choosing the contour with the largest area.

**How was each digit extracted?**
This was done by splitting the cropped image of the ssudoku board in 9X9 boxes. This seems to be enough, but the cell demarkations proved to be a challenge for the neural network, as it would often classify blank cells to be digit "7" or "1", due to the lines being present. So, the borders of each image had to be removed. This was achieved by fidning the largest connected **blob** aka the contour at the center of the image. This method, however fails, when the cell demarkations are clearly visible in the extracted cell image and hence are the **largest center** blob. Often clearing the border removes the enter digit, as when they are close to the border. So, many a times corrections have to be passed into the "corrections array" in test_model.py

**Why was the neural network not trained on MNIST?**
Here, printed digits on a sudoku board is being dealt with, and I didn't want to take chances. You could possibly use a MNIST trained model on this, and expect similar results. 

### Note:
Model hasn't been trained on MNIST, as most of the digits in Sudoku puzzles are in printed format. The images of the digits can be found in digits folder, which was downloaded from: https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset?resource=download
This was then converted into Numpy arrays, using a loader, which isn't included in this repo. Feel free to use your own dataset and models.
### Bugs:
Still a lot of work is to be done.
Extracting the digit isn't exactly perfect at this moment.
Corrections, have to be applied whenever digits aren't properly recognized.

### Future Updates:
I plan to implement a much more robust image extraction technique. Also, I would like to implement a feature such that the user can input their image via their webcam or mobile camera.



