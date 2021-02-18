import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def surrounds(box_parent, box_child):
    return box_parent[0][0] < box_child[0][0] and \
        box_parent [0][1] < box_child[0][1] and \
        box_parent[1][0] > box_child[1][0] and \
        box_parent[1][1] > box_child[1][1]

def hasSurrounding(boxes, box):
    return np.count_nonzero(list(map(lambda itr_box: surrounds(itr_box, box), boxes))) != 0

im = cv2.imread('./assets/symbols.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = []

for contour in contours:
    max_x = -math.inf
    min_x = math.inf
    max_y = -math.inf
    min_y = math.inf

    for coord_wrapper in contour:
        coord = coord_wrapper[0]
        if coord[0] < min_x:
            min_x = coord[0]
        if coord[0] > max_x:
            max_x = coord[0]
        if coord[1] < min_y:
            min_y = coord[1]
        if coord[1] > max_y:
            max_y = coord[1]

    # This ignores the box around the whole image
    if min_x != 0 or min_y != 0:
        boxes.append(((min_x, min_y),(max_x,max_y)))

# surrounding boxes are removed for characters such as
# 'e' and 'g' which have contours on the interior of
# the symbol
boxes_new = []
for box in boxes:
    if not hasSurrounding(boxes, box):
        boxes_new.append(box)
boxes = boxes_new

for box in boxes:
    im = cv2.rectangle(im, box[0], box[1], (255,0,0))
imgplot = plt.imshow(im)
plt.show()

print(boxes)
print("num boxes:", len(boxes))
