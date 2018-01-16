from sklearn.feature_extraction import image
import scipy.misc
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "data/C_0029_1.LEFT_CC.jpg";
output= "out/"
img = cv2.imread(image_path)
# #
# dst = cv2.fastNlMeansDenoising(img, None, 30.0, 7, 21);
# # # print dst
# #
# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

# img = Image.open(image_path)
# im = cv2.imread(image_path)
# print (im.shape)
# one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
# print one_image.shape
# one_image = np.arange(4 * 4 * 3).reshape(im.shape)
# print img
# patches = image.extract_patches_2d(im, (256, 256) , max_patches=12);
# print patches.shape
# index = 1;
# for patch in patches:
#     print patch
#     scipy.misc.imsave(output+str(index)+'.jpg', patch)
#     index = index + 1;

# src = cv2.imread(image_path, 1)
# tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
# b, g, r = cv2.split(src)
# rgba = [b, g, r, alpha]
# dst = cv2.merge(rgba, 4)
# cv2.imwrite("test.jpg", dst)
# for
# scipy.misc.imsave('outfile.jpg', image_array)
# print(patches)

import PIL

# image = PIL.Image.open('C_0029_1.LEFT_CC.LJPEG.1.template.pgm')
# print image

import re
import numpy

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        print header
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def printMaxSubSquare(M):
    R = len(M)  # no. of rows in M[][]
    C = len(M[0])  # no. of columns in M[][]

    S = [[0 for k in range(C)] for l in range(R)]
    # here we have set the first row and column of S[][]

    # Construct other entries
    for i in range(1, R):
        for j in range(1, C):
            if (M[i][j] == 255):
                S[i][j] = min(S[i][j - 1], S[i - 1][j],
                              S[i - 1][j - 1]) + 1
            else:
                S[i][j] = 0

    # Find the maximum entry and
    # indices of maximum entry in S[][]
    max_of_s = S[0][0]
    max_i = 0
    max_j = 0
    for i in range(R):
        for j in range(C):
            if (max_of_s < S[i][j]):
                max_of_s = S[i][j]
                max_i = i
                max_j = j

    print("Maximum size sub-matrix is: ", M)
    for i in range(max_i, max_i - max_of_s, -1):
        for j in range(max_j, max_j - max_of_s, -1):
             print M[i][j] , i , j
            # print("")


if __name__ == "__main__":
    from matplotlib import pyplot
    image = read_pgm("C_0029_1.LEFT_CC.LJPEG.1.template.pgm", byteorder='<')
    # print image.shape
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()
    mask = cv2.imread('C_0029_1.LEFT_CC.LJPEG.1.template.pgm', 0)
    # print mask
    coordList = numpy.argwhere(mask == 255)
    numWhitePoints = len(coordList)

    # print coordList
    #
    # # printMaxSubSquare(mask);
    #
    # # print "Found {0} points".format(numWhitePoints)
    col = 0
    row = 0
    top_y = 5000
    leftx = 5000
    rightx = 0
    bottomy = 0
    for pixel_row in mask:
        # print "Start"
        col = 0
        for pixel in pixel_row:
            if pixel != 0 and pixel != 255:
                print pixel
            if pixel != 0:
                if top_y > row:
                    top_y = row
                if bottomy <= row:
                    bottomy = row

                if leftx > col:
                    leftx = col

                if rightx <= col:
                    rightx = col

            col = col + 1
        row = row + 1
    print top_y , "top"
    print leftx , "left"
    print rightx, "right"
    print bottomy , "bottom"
    # print "end"

    # mask = np.random.random_integers(0, 1, 48).reshape(8, 6)
    # img = np.random.random_integers(3, 9, 8 * 6 * 3).reshape(8, 6, 3)
    # chk = np.ones((8, 6, 3))
    # print mask.shape
    # print img.shape
    # mask = mask[:, :, np.newaxis]
    # res = np.where(mask == 0, chk, img)
    # pyplot.imshow(chk, pyplot.cm.gray)
    # pyplot.show()
    pyplot.imshow(coordList, pyplot.cm.gray)
    # pyplot.show()
    # miny = 100000;
    # maxy = 0;
    # minx = 100000;
    # maxx = 0;
    # for i in (coordList):
    #
    #     if(miny > i[1:]):
    #         miny = i[1:]
    #     if (maxy < i[1:]):
    #         maxy = i[1:]
    #     if (minx > i[:1]):
    #         minx = i[:1]
    #     if (maxx < i[:1]):
    #         maxx = i[:1]
    # print miny , maxy , minx , maxx
    # from PIL import Image
    #
    # img = Image.open("C_0029_1.LEFT_CC.jpg")
    img = Image.open("C_0029_1.LEFT_CC.jpg")
    width, height  = img.size
    # print width , height
    area = (leftx, top_y , rightx, bottomy)
    cropped_img = img.crop(area)
    cropped_img.show()
