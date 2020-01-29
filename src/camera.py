import cv2
import numpy as np
import imutils

class Camera:

    def __init__(self, image, save_image=True):
        self.image = image
        self.save_image = save_image


    def capture_food_image(self, sens_bound, step):
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # define range of green color in HSV
        # lower_green = np.array([36, 0, 0])
        # upper_green = np.array([86, 255, 255])

        lower_green = np.array([60 - sens_bound, 100, 100])
        upper_green = np.array([60 + sens_bound, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        ## slice the green
        imask = mask > 0
        img_green = np.zeros_like(self.image, np.uint8)
        img_green[imask] = self.image[imask]

        ## save
        #cv2.imwrite("green.png", img_green)

        # print(np.array(img_green))

        # cv2.imshow('mask', mask)
        # cv2.imshow('hsv', img_hsv)
        # cv2.imshow('self', self.image)
        # cv2.imshow('green', img_green)
        # cv2.waitKey(0)

        self.img_green = img_green

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        self.contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(self.contours)

        # lengths of axes of the image
        x_ax = float(np.array(self.image).shape[1])
        y_ax = float(np.array(self.image).shape[0])


        x = 0  # bigger it is, more central it is, smaller, more off
        y = None
        neg_x = 0  #129  # bigger it is, more central it is, smaller, more off
        neg_y = y_ax  # --> smaller, more desirable

        if self.contours:
            # x,y is top left coordinate of rectangle
            # x and y are distances from object in positive axes directions

            contours = []
            for contour in self.contours:
                x, y, w, h = cv2.boundingRect(contour)
                contours.append([x,y,w,h])

            contours = np.array(contours)
            print("{} object{} detected".format(len(contours), 's' if len(contours) > 1 else ""))

            x, y, w, h = contours[np.argmax(contours[:,2] * contours[:,3])]
            # x, y, w, h = cv2.boundingRect(self.contours[0])
            print("edges: {}, {}, {}, {}".format(x, y, w, h))

            self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # distances from object in negative axes directions
            neg_x = x_ax - x - w
            neg_y = y_ax - y - h
            print('distance: {}, {}, {}, {}'.format(x, y, neg_x, neg_y))
            print('axis sum: {}, {}'.format(neg_x+x+w, neg_y+y+h))


        if self.save_image:
            path = "green_{}.png".format(step)
            save = cv2.imwrite(path, self.image)
            print('Image saved: {}'.format(save))

        return x/x_ax, neg_x/x_ax, neg_y/y_ax

    def capture_prey_image(self, sens_bound, step):
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # define range of red color in HSV
        # lower_green = np.array([36, 0, 0])
        # upper_green = np.array([86, 255, 255])

        #HSV?
        # lower_red = np.array([110 - sens_bound, 70, 50])
        # upper_red = np.array([70 + sens_bound, 255, 255])

        # #BG3
        # lower_red = np.array([47 - sens_bound, 15, 100])
        # upper_red = np.array([20 + sens_bound, 56, 200])

        lower_red1 = np.array([0, 100, 0])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 100, 0])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the RBG image to get only red colors
        mask1 = cv2.inRange(self.hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(self.hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_xor(mask1, mask2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        ## slice the green
        imask = mask > 0
        img_red = np.zeros_like(self.image, np.uint8)
        img_red[imask] = self.image[imask]

        ## save
        #cv2.imwrite("green.png", img_green)

        # print(np.array(img_green))

        # cv2.imshow('mask', mask)
        # cv2.imshow('hsv', img_hsv)
        # cv2.imshow('self', self.image)
        # cv2.imshow('green', img_green)
        # cv2.waitKey(0)

        self.img_red = img_red

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        self.contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(self.contours)

        # lengths of axes of the image
        x_ax = float(np.array(self.image).shape[1])
        y_ax = float(np.array(self.image).shape[0])


        x = 0  # bigger it is, more central it is, smaller, more off
        y = None
        neg_x = 0  #129  # bigger it is, more central it is, smaller, more off
        neg_y = y_ax  # --> smaller, more desirable

        if self.contours:
            # x,y is top left coordinate of rectangle
            # x and y are distances from object in positive axes directions

            contours = []
            for contour in self.contours:
                x, y, w, h = cv2.boundingRect(contour)
                contours.append([x,y,w,h])

            contours = np.array(contours)
            print("{} object{} detected".format(len(contours), 's' if len(contours) > 1 else ""))

            x, y, w, h = contours[np.argmax(contours[:,2] * contours[:,3])]
            # x, y, w, h = cv2.boundingRect(self.contours[0])
            print("edges: {}, {}, {}, {}".format(x, y, w, h))

            self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # distances from object in negative axes directions
            neg_x = x_ax - x - w
            neg_y = y_ax - y - h
            print('distance: {}, {}, {}, {}'.format(x, y, neg_x, neg_y))
            print('axis sum: {}, {}'.format(neg_x+x+w, neg_y+y+h))


        if self.save_image:
            #path = "green_{}.png".format(step)
            path = "red.png"
            save = cv2.imwrite(path, self.image)
            print('Image saved: {}'.format(save))

        return x/x_ax, neg_x/x_ax, neg_y/y_ax



    # def find_food_area(self):
    #     #img_gray = cv2.cvtColor(self.img_green, cv2.COLOR_BGR2GRAY)
    #
    #     food_area = np.where(self.img_green > 0, 1, 0)
    #     print('food_area.shape',food_area.shape)
    #
    #     part = int(food_area.shape[1] / 3)
    #     #print('part', part)
    #     a1 = food_area[:, 0:part + 1]
    #     print("a2 size", a1.shape)
    #     a2 = food_area[:, part + 1: 2 * part + 1]
    #     a3 = food_area[:, 2 * part + 1: 3 * part + 2]
    #
    #     self.n_a1 = (a1 == 1).sum()
    #     self.n_a2 = (a2 == 1).sum()
    #     self.n_a3 = (a3 == 1).sum()
    #
    #     self.a1_size = a1.size
    #     self.a2_size = a2.size
    #     self.a3_size = a3.size
    #
    #     return self.n_a1, self.n_a2, self.n_a3
    #
    # def detect_food_location(self):
    #
    #     if self.n_a2>self.n_a1 and self.n_a2>self.n_a3:
    #         self.obj_location = 'center'
    #     elif self.n_a1>self.n_a2 and self.n_a1>self.n_a3:
    #         self.obj_location = 'left'
    #     elif self.n_a3>self.n_a1 and self.n_a3>self.n_a2:
    #         self.obj_location = 'right'
    #     else:
    #         self.obj_location = 'no food'
    #
    #     return self.obj_location
    #
    # #def find_center_of_mass(self):
    #
    #
    # def find_food_percentage(self):
    #
    #     self.a1_obj_percentage = self.n_a1/ self.a1_size
    #     self.a2_obj_percentage = self.n_a2 / self.a2_size
    #     self.a3_obj_percentage = self.n_a3 / self.a3_size
    #
    #     return self.a1_obj_percentage, self.a2_obj_percentage, self.a3_obj_percentage
    #
    # def find_edges(self):
    #     # convert to gray
    #     img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #
    #
    #     edges = cv2.Canny(img_gray, 100, 200)
    #
    #     num_horizon = np.array(edges).shape[0]  # approximate
    #     #num_black = (edges == 0).sum()
    #     num_white = (edges == 255).sum()
    #     num_edge = num_white - num_horizon
    #
    #     return num_edge
    #
    # def find_object_area(self):
    #     img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #
    #     obj_area = np.where((img_gray > 0) & (img_gray < 156), 1, 0)
    #
    #     part = int(obj_area.shape[1] / 3)
    #     #print(part)
    #     a1 = obj_area[:, 0:part + 1]
    #     a2 = obj_area[:, part + 1: 2 * part + 1]
    #     a3 = obj_area[:, 2 * part + 1: 3 * part + 2]
    #
    #     self.n_a1 = (a1 == 1).sum()
    #     self.n_a2 = (a2 == 1).sum()
    #     self.n_a3 = (a3 == 1).sum()
    #
    #     self.a1_size = a1.size
    #     self.a2_size = a2.size
    #     self.a3_size = a3.size
    #
    #     return self.n_a1, self.n_a2, self.n_a3
    #
    # def find_object_location(self):
    #
    #     if self.n_a2>self.n_a1 and self.n_a2>self.n_a3:
    #         self.obj_location = 'center'
    #     elif self.n_a1>self.n_a2 and self.n_a1>self.n_a3:
    #         self.obj_location = 'left'
    #     elif self.n_a3>self.n_a1 and self.n_a3>self.n_a2:
    #         self.obj_location = 'right'
    #     else:
    #         self.obj_location = 'no object'
    #
    #     return self.obj_location
    #
    # def obstacle_detection(self, edges):
    #
    #     if self.obj_location=='center': #and edges>130:
    #         self.obs_existence = 'object detected'
    #     elif self.obj_location=='left': #and edges<100:
    #         self.obs_existence = 'object detected'
    #     elif self.obj_location == 'right': #and edges<100:
    #         self.obs_existence = 'object detected'
    #     elif edges<0:
    #          self.obs_existence = 'wall detected'
    #     else:
    #         self.obs_existence = 'nothing detected'
    #
    #     return self.obs_existence
    #
    # def find_object_percentage(self):
    #
    #     self.a1_obj_percentage = self.n_a1/ self.a1_size
    #     self.a2_obj_percentage = self.n_a2 / self.a2_size
    #     self.a3_obj_percentage = self.n_a3 / self.a3_size
    #
    #     return self.a1_obj_percentage, self.a2_obj_percentage, self.a3_obj_percentage
    #
    # def detect_wall(self):
    #
    #     img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #
    #     wall_detection = np.all(img_gray>207)
    #
    #     return wall_detection
