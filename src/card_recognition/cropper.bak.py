import cv2
import matplotlib.pyplot as plt
import numpy as np


class Cropper:
    def find_four_corner(self, img):
        """
        Xác định toạ độ 4 góc của thẻ sinh viên

        :param img: Ảnh thẻ sinh viên được đọc bởi thư viện opencv
        :return: Toạ độ 4 góc thẻ (top_left -> top_right -> bottom_right -> bottom_left)
        """

        # reduce noise with Gaussian Filter
        noise_rd = cv2.GaussianBlur(img, (9, 9), 0)
        # convert to HSV
        hsv = cv2.cvtColor(noise_rd, cv2.COLOR_BGR2HSV)
        h, s, v, = cv2.split(hsv)
        # canny edge on S-channel
        canny_s = cv2.Canny(s, 0, 150)
        # make edge more sharpe
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        canny = cv2.dilate(canny_s, kernel, iterations=1)
        # find contours
        cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # sort DESC
        cnts = sorted(cnts, key=self.sort_by_area, reverse=True)

        # cv2.drawContours(img, contours, 2, (0, 255, 0), 2)
        # cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
        cv2.imshow("s", canny_s)
        cv2.waitKey(0)

        # the biggest one is our ROI
        c = cnts[0]
        # calc arc length of selected contour
        arc_length = cv2.arcLength(c, True)
        # approximate shape of contour by various points
        approx = cv2.approxPolyDP(c, 0.01 * arc_length, True)
        approx = np.asarray([p[0] for p in approx])
        # sum x + y
        sum_approx = np.sum(approx, axis=1)
        # diff x - y
        diff_approx = np.diff(approx, axis=1)
        corner_points = np.array([
            # top_left
            approx[np.argmin(sum_approx)] + [-10, -10],
            # top_right
            approx[np.argmin(diff_approx)] + [10, -10],
            # bottom_right
            approx[np.argmax(sum_approx)] + [10, 10],
            # bottom_left
            approx[np.argmax(diff_approx)] + [-10, 10]
        ], dtype='float32')
        # for p in approx:
        #     x, y = p
        #     cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return corner_points

    def transform(self, origin_img):
        """
        Cắt và xoay phần thẻ sinh viên trong ảnh đầu vào

        :param origin_img: Ảnh đầu vào được đọc bằng opencv
        :return: ảnh chỉ chứa thẻ sinh viên
        """

        img = np.copy(origin_img)
        # find four corner coordinates
        corner_pts = self.find_four_corner(img)
        # calc img size after crop
        width, height = self.calc_fit_size(corner_pts)
        # rotate img
        dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        m = cv2.getPerspectiveTransform(corner_pts, dst_points)
        transformed_img = cv2.warpPerspective(img, m, (width, height))

        return transformed_img

    @staticmethod
    def calc_fit_size(corner_pts):
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

    @staticmethod
    def sort_by_area(contours):
        (_, _, w, h) = cv2.boundingRect(contours)
        return h * w


