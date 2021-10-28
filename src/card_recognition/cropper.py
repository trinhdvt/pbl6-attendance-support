import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


class Cropper:
    """
    Cắt vùng thẻ sinh viên ra khỏi background ảnh đầu vào
    """

    def __init__(self, MAX_WIDTH=760):
        self.MAX_WIDTH = MAX_WIDTH

    @staticmethod
    def contour_sort_fn(contour):
        return cv2.arcLength(contour, True)

    @staticmethod
    def preprocess_threshold(blurred_img):
        #
        gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

        # _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        thresh = cv2.Canny(gray_img, 100, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        return thresh

    @staticmethod
    def preprocess_hsv(blurred_img):
        hsv = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2HSV)

        # sky-blue color
        # lower HUE, SATURATION, VALUE
        lower = np.array([70, 20, 20])

        # upper HUE, SATURATION, VALUE
        upper = np.array([180, 255, 255])

        # get the binaries image
        thresh = cv2.inRange(hsv, lower, upper)

        # erode mask to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh

    @staticmethod
    def get_contours(thresh_img):
        # find all contours
        contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # grab contours
        cnts = imutils.grab_contours(contours)

        # sort contours by sort function
        cnts = sorted(cnts, key=Cropper.contour_sort_fn, reverse=True)

        # the biggest one is the whole image, the second-largest one is what we need
        c = cnts[1]
        return c

    @staticmethod
    def get_4corners(contour):
        # calculate arc length of selected contour
        peri = cv2.arcLength(contour, True)

        # approximate the shape of contour with variety of points
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        approx = np.asarray([p[0] for p in approx])

        # sum x + y
        sum_approx = np.sum(approx, axis=1)
        # diff y - x
        diff_approx = np.diff(approx, axis=1)

        corner_point = np.array([
            # [-10, -10]: padding
            # top_left point has the smallest sum
            approx[np.argmin(sum_approx)] + [-10, -10],
            # top_right point has the smallest diff
            approx[np.argmin(diff_approx)] + [10, -10],
            # bottom_right
            approx[np.argmax(sum_approx)] + [10, 10],
            # bottom_left
            approx[np.argmax(diff_approx)] + [-10, 10],
        ], dtype="float32")

        return corner_point

    @staticmethod
    def calculate_fit_size(corner_pts):
        """
        Tính size của ảnh sau khi cắt ra

        :param corner_pts: Toạ độ 4 góc
        :return: Size tương ứng
        """

        (tl, tr, br, bl) = corner_pts
        # width = distance between top_left and top_right
        # or bottom_right and bottom_left
        width1 = np.linalg.norm(tl - tr)
        width2 = np.linalg.norm(br - bl)
        new_width = max(int(width1), int(width2))

        #
        height1 = np.linalg.norm(tl - bl)
        height2 = np.linalg.norm(tr - br)
        new_height = max(int(height1), int(height2))

        #
        return new_width, new_height

    def transform(self, origin_img):
        """
        :param origin_img: Ảnh đầu vào được đọc bằng opencv
        :return: ảnh chỉ chứa thẻ sinh viên
        """

        img = origin_img.copy()
        (h, w, c) = img.shape
        if w > self.MAX_WIDTH:
            img = imutils.resize(img, width=self.MAX_WIDTH)
        blurred_img = cv2.blur(img, (5, 5))

        # threshold images
        thresh_img = self.preprocess_hsv(blurred_img)

        # get contour
        contour = self.get_contours(thresh_img)

        # calculate 4 corner points
        corner_pts = self.get_4corners(contour)

        # calculate size for output img
        width, height = self.calculate_fit_size(corner_pts)

        # rotate img
        dst_point = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        m = cv2.getPerspectiveTransform(corner_pts, dst_point)
        transformed_img = cv2.warpPerspective(img, m, (width, height))

        #
        return transformed_img


def plot_img(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    ax1.imshow(img1)
    ax1.set_title("Origin image")

    ax2.imshow(img2)
    ax2.set_title("Crop and center aligned")

    plt.show()


if __name__ == '__main__':
    img_path = "/Volumes/MacDATA/VSCodeWorkSpace/Python/PBL6-attendance-support/image/test_img/fb_img/101180012.jpeg"
    img = cv2.imread(img_path)
    cropper = Cropper()
    rs = cropper.transform(img)
    plot_img(img, rs)
