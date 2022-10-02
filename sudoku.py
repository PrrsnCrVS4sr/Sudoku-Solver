from time import sleep
import cv2
import numpy as np
import inspect
import sys
import re
import operator
from model import Trainer
from solver import Solver
from skimage.segmentation import clear_border
import imutils


class Detector:
    def __init__(self):
        p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")
        self.t = Trainer()
        self.t.load_data()
        try:
            self.t.load_model()
        except:
            self.t.train()
        self.t.test()
        self.stages = list(sorted(
            map(
                lambda x: (p.fullmatch(x[0]).groupdict()[
                    'idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
                filter(
                    lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
                    inspect.getmembers(self))),
            key=lambda x: x[0]))

        # For storing the recognized digits
        self.digits = []

    # Takes as input 9x9 array of numpy images
    # Combines them into 1 image and returns
    # All 9x9 images need to be of same shape
    def makePreview(images):
        assert isinstance(images, list)
        assert len(images) > 0
        assert isinstance(images[0], list)
        assert len(images[0]) > 0
        assert isinstance(images[0], list)

        rows = len(images)
        cols = len(images[0])

        cellShape = images[0][0].shape

        padding = 10
        shape = (rows * cellShape[0] + (rows + 1) * padding,
                 cols * cellShape[1] + (cols + 1) * padding)

        result = np.full(shape, 255, np.uint8)

        for row in range(rows):
            for col in range(cols):
                pos = (
                    row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

                result[pos[0]:pos[0] + cellShape[0], pos[1]
                    :pos[1] + cellShape[1]] = images[row][col]

        return result

    # Takes as input 9x9 array of digits
    # Prints it out on the console in the form of sudoku
    # None instead of number means that its an empty cell

    def showSudoku(self,array):
        cnt = 0
        for row in array:
            if cnt % 3 == 0:
                print('+-------+-------+-------+')

            colcnt = 0
            for cell in row:
                if colcnt % 3 == 0:
                    print('| ', end='')
                print('. ' if cell == 0 else str(cell) + ' ', end='')
                colcnt += 1
            print('|')
            cnt += 1
        print('+-------+-------+-------+')

    # Runs the detector on the image at path, and returns the 9x9 solved digits
    # if show=True, then the stage results are shown on screen
    # Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
    # that the digit at (1,2) is corrected to 9
    # and the digit at (3,3) is corrected to 4
    def run(self, path='assets/sudokus/sudoku1.jpg', show=False, corrections=[]):
        self.path = path
        self.original = cv2.imread(path)

        self.run_stages(show)
        result = self.solve(corrections)

        if show:
            self.showSolved()

            cv2.waitKey(0)
            sleep(5)
            cv2.destroyAllWindows()

        return result

    # Runs all the stages
    def run_stages(self, show):
        results = [('Original', self.original)]

        for idx, name, fun in self.stages:
            image = fun().copy()
            results.append((name, image))

        if show:
            for name, img in results:
                cv2.imshow(name, img)

    # Stages
    # Stage function name format: stage_[stage index]_[stage name]
    # Stages are executed increasing order of stage index
    # The function should return a numpy image, which is shown onto the screen
    # In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
    # to create a single image out of those
    # You can pass data from one stage to another using class member variables

    def stage_1_preprocess(self):
        image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (7, 7), 3)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image = cv2.bitwise_not(image, image)
        self.image1 = image

        return image

    def stage_2_crop(self):
        contours, hierarchy = cv2.findContours(
            self.image1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(self.original,contours,-1,(0,255,0),3)
        cornersofLargestContour, maxArea = self.largest_contour(contours)
        if cornersofLargestContour.size != 0:
            cornersofLargestContour = self.arrangeCorners(
                cornersofLargestContour)
            # cv2.drawContours(self.original.copy(),cornersofLargestContour,-1,(0,255,0),10)
            pers1 = np.float32(cornersofLargestContour)
            side = max([self.distance_between(cornersofLargestContour[2], cornersofLargestContour[1]),
                        self.distance_between(
                cornersofLargestContour[0], cornersofLargestContour[3]),
                self.distance_between(
                cornersofLargestContour[1], cornersofLargestContour[3]),
                self.distance_between(cornersofLargestContour[0], cornersofLargestContour[1])])
            pers2 = np.float32(
                [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
            matrix = cv2.getPerspectiveTransform(pers1, pers2)
            imageWarp = cv2.warpPerspective(
                self.original.copy(), matrix, (int(side), int(side)))
            imageWarpGray = cv2.cvtColor(imageWarp, cv2.COLOR_BGR2GRAY)

            self.image2 = cv2.resize(imageWarpGray, (450, 450))
        # print(cornersofLargestContour)

        return self.image2

    def stage_3_split_image(self):
        image = self.image2.copy()
        cells = self.extractCells(image)
        # t.test()
        j = 0
        # i = cells[4]
        # cv2.imshow("lol", i)
        # img = self.extract_digit(i,True)
        # print(img)
        # if(img is None):
        #     print(0)
        # else:
        #     print(self.identify_num(img))
        #     cv2.imshow("lol2", img)
        row = []
        for i in cells:

            img = self.extract_digit(i)
            if(img is None):
                row.append(0)
            else:
                row.append(self.identify_num(img))

            j = j+1
            if(j == 9):
                self.digits.append(row)
                row = []
                j = 0
        return self.original

    def extractCells(self, image):
        self.cell_locations= []
        stepX = image.copy().shape[1] // 9
        stepY = image.copy().shape[0] // 9
        rows = np.vsplit(image, 9)
        cells = []
        x = 0
        y = 0
        for i in rows:
            rows_ = []
            startX = x * stepX
            endX = (x + 1) * stepX		
            columns = np.hsplit(i, 9)
            for cell in columns:
                startY = y * stepY
                endY = (y + 1) * stepY
                cells.append(cell)
                y = y+1
                # print(startX)
                rows_.append((startX,startY,endX,endY))
            x = x+1
            y = 0
            self.cell_locations.append(rows_)
        # print(self.cell_locations)
        return cells

    def extract_digit(self, cell, debug=False):
        thresh = cv2.threshold(cell, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        thresh = clear_border(thresh)
        # cv2.imwrite("img2.png",thresh)
        contours = cv2.findContours(thresh.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if len(contours) == 0:
            return None

        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype='uint8')

        cv2.drawContours(mask, [c], -1, 255, -1)

        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w*h)

        if debug:
            print(percentFilled)

        if percentFilled < 0.03:
            return None

        digit = cv2.bitwise_and(thresh, thresh, mask=mask)

        if debug:
            cv2.imshow("Digit", digit)
            cv2.waitKey(0)

        return digit

    def largest_contour(self, contours):
        largest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 30:
                perimeter = cv2.arcLength(i, True)
                approxCorners = cv2.approxPolyDP(i, 0.02*perimeter, True)
                if area > max_area and len(approxCorners) == 4:
                    largest = approxCorners
                    max_area = area
        return largest, max_area

    def arrangeCorners(self, corners):
        corners = corners.reshape((4, 2))
        reorderedCorners = np.zeros((4, 1, 2), dtype=np.int32)
        # take columnwise difference
        summation = corners.sum(1)
        difference = np.diff(corners, axis=1)
        reorderedCorners[0] = corners[np.argmin(summation)]
        reorderedCorners[3] = corners[np.argmax(difference)]
        reorderedCorners[2] = corners[np.argmax(summation)]
        reorderedCorners[1] = corners[np.argmin(difference)]
        return reorderedCorners

    def identify_num(self, image):
        pred = 0
        if(image.sum() > 25000):
            image_resize = cv2.resize(image, (28, 28))    #
            pred = self.t.predict(image_resize)
        else:
            pred = 0
        return pred

    def distance_between(self, p1, p2):
        p1 = p1.reshape(2, 1)
        p2 = p2.reshape(2, 1)
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def solve(self, corrections):
        # Only upto 3 corrections allowed
        assert len(corrections) < 3

        # Apply the corrections
        for i in corrections:
            self.digits[i[0]][i[1]] = i[2]
        # Solve the sudoku.
        # print(self.digits)
        self.answers = [[self.digits[j][i]
                         for i in range(9)] for j in range(9)]
        s = Solver(self.answers)
        if s.solve():
            self.showSudoku(self.digits)
            self.answers = s.digits
            self.showSudoku(self.answers)
            self.flipped_answers = [[self.answers[i][j]
                         for i in range(9)] for j in range(9)]
            self.showSolved()
            return s.digits

        return [[None for i in range(9)] for j in range(9)]

    # Optional
    # Use this function to backproject the solved digits onto the original image
    # Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
    # an appropriate name
    def showSolved(self):
        # self.showSudoku(self.digits)
        img = self.image2.copy()
        i = 0
        j = 0
        for cellRow,boardRow in zip(self.cell_locations
        , self.flipped_answers):
            
            for (box, digit) in zip(cellRow,boardRow):
                    # print(i, j)
                    startX,startY,endX,endY = box

                    testX = int((endX - startY)*0.33)
                    textX = int((endX - startX) * 0.33)
                    textY = int((endY - startY) * -0.2)
                    textX += startX
                    textY += endY
                    if(self.digits[j][i] == 0):
                        cv2.putText(img, str(digit), (textX, textY),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    j = j+1
            i = i+1
            j=0
        cv2.imshow("solution",img)
        cv2.waitKey(0)
        cv2.imwrite("solution.png",img)



if __name__ == '__main__':
    d = Detector()
    result = d.run('assets/sudokus/sudoku1.jpg', show=True)
    print('Recognized Sudoku:')
    Detector.showSudoku(d.digits)
    print('\n\nSolved Sudoku:')
    Detector.showSudoku(result)
