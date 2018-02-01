
from PIL import Image
import cv2
import imutils
import numpy as np
import os

PIC_PATH = "C_0029_1.LEFT_CC.LJPEG.1.template.pgm";
def create_PatchByGivenCordi(img, area, jpeg_file_path ,out_file_path):
        #print area
        #print(out_file_path , area)
        cropped_img = img.crop(area)
        cropped_img.save(out_file_path)


def extrct_ROIImage(pgm_file_path, jpeg_file_path ,out_root_path, output_file_name):
        image = cv2.imread(pgm_file_path);
        img = Image.open(jpeg_file_path)
        width, height = img.size
        #print width
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edged = cv2.Canny(gray, 100, 220)

        kernel = np.ones((5,5),np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        images, contours, hier = cv2.findContours(closed.copy(), cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        prevx = 0;
        prevy = 0;
        prevw = 0;
        prevh = 0;

        for c in contours:
            # get the bounding rect
#            print(c)
            x, y, w, h = cv2.boundingRect(c)
            if  prevx != x and prevy != y and prevw != w and prevh != h:
                #print x , y , x + w , y + h

                left_rect = ((x - (((x+ w) - x) / 2 )) if (x - (((x+ w) - x) / 2 )) >= 0 else 0,
                              y , ((x + w ) - (((x+ w) - x) / 2 )) , y + h)
                top_rect = (x , (y - (((y+h) - y) / 2)) if (y - (((y+h) - y) / 2)) >= 0 else 0,
                             x + w , ((y + h) -  (((y+h) - y) / 2)))
                right_rect = ((x + (((x + w) - x) / 2)), y,
                              ((x + w) + (((x+ w) - x) / 2 )) if ((x + w) + (((x+ w) - x) / 2 )) < width else width, y + h)
                bot_rect = (x , (y + (((y+h) - y) / 2)) ,
                             x + w , ((y + h) +  (((y+h) - y) / 2)) if ((y + h) + (((y+h) - y) / 2)) < height else height)

                create_PatchByGivenCordi(img, (x, y, x+w , y+h),
                                         jpeg_file_path , os.path.join(out_root_path, output_file_name)  )
                # create_PatchByGivenCordi(img, left_rect  ,jpeg_file_path ,
                #                           os.path.join(out_root_path, "left_"+output_file_name ))
                # create_PatchByGivenCordi(img, right_rect, jpeg_file_path,
                #                          os.path.join(out_root_path, "right_" + output_file_name))
                # create_PatchByGivenCordi(img, top_rect, jpeg_file_path,
                #                          os.path.join(out_root_path, "top_" + output_file_name))
                # create_PatchByGivenCordi(img, bot_rect, jpeg_file_path,
                #                          os.path.join(out_root_path, "bott_" + output_file_name))

                #print ((x + w) + (((x+ w) - x) / 2 ))
                #print (((x+ w) - x) / 2 ) , (((y+h) - y) / 2)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #
                # # get the min area rect
                # rect = cv2.minAreaRect(c)
                # box = cv2.boxPoints(rect)
                # # convert all coordinates floating point values to int
                # box = np.int0(box)
                # # draw a red 'nghien' rectangle
                # cv2.drawContours(image, [box], 0, (0, 0, 255))
                # draw a green rectangle to visualize the bounding rect
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            prevx = x;
            prevy = y;
            prevw = w;
            prevh = h;
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)

