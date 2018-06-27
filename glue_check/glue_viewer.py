import datetime
import os
import time
import numpy as np
from threading import Thread

import cv2

from file_check.file_monitor import FileMonitor
from file_check.file_monitor_test import FileMonitorTest


class ImageProc:
    def __init__(self, image_queue, time_pause=0.01):
        self.index = 0
        self.image_queue = image_queue
        self.image = None

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        t = Thread(target=self.process, args=(), daemon=True)
        t.start()
        return self

    def process(self):
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            image_path = self.image_queue.get()

            if os.path.exists(image_path):
                self.image, result = self.proc(image_path)
                if self.image is not None:
                    if result == 0:
                        print('ok')
                    elif result == 1:
                        print('black mask area contaminated')
                    elif result == 2:
                        print('sensor area contaminated')
                    else:
                        print('unknown')

    @staticmethod
    def proc(image_path):
        ori_image = cv2.imread(image_path, 0)
        if ori_image is not None:
            image = ori_image.copy()
            # t1 = datetime.datetime.now()
            h, w = image.shape[:2]
            image = image[int(h / 5):int(4 * h / 5), int(w / 5):int(4 * w / 5)]

            h, w = image.shape[:2]
            image = cv2.medianBlur(image, 3)
            center_color = np.mean(image[int(h / 2 - 10):int(h / 2 + 10), int(w / 2 - 10):int(w / 2 + 10)]).astype(int)

            # print(center_color)
            val, roi = cv2.threshold(image, int(center_color * 0.75), 255, cv2.THRESH_BINARY)
            roi = cv2.erode(roi, None, iterations=4)

            mask = np.zeros((h + 2, w + 2), np.uint8)

            # in case the center is glue, so choose the brightest one
            anchor_point = [(h // 2, w // 2),
                            (h // 2, w // 2 - 10), (h // 2, w // 2 + 10),
                            (h // 2 - 10, w // 2), (h // 2 + 10, w // 2)]

            val = np.zeros(5)
            for i, ap in enumerate(anchor_point):
                val[i] = roi[ap]

            anchor = anchor_point[np.argmax(val)]

            _, roi, _, _ = cv2.floodFill(roi, mask, (anchor[1], anchor[0]), 128)
            roi[roi != 128] = 0
            roi[roi > 0] = 255

            roi_whole = roi.copy()

            # fix rotation
            _imgs = ImageProc.fix_rotation(image, base_roi=roi)
            roi = _imgs[0]
            image = _imgs[1]

            cnts = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts) > 0:
                cnt = cnts[0]
                x, y, w, h = cv2.boundingRect(cnt)

                # offset
                # x += 2
                # y -= 2
                # w -= 2
                # h -= 2

                if len(cnt) > 0:
                    image = image[y:y + h, x:x + w]
                    roi = image.copy()

            # roi[roi == 0] = 255
            for i in range(3):
                roi = cv2.medianBlur(roi, 3)

            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            roi_dark = cv2.bitwise_not(roi)
            coords = np.column_stack(np.where(roi_dark > 0))
            y, x, h, w = cv2.boundingRect(coords)  # noticing input is numpy row col instead x, y

            roi_dark = roi_dark[y:y + h, x:x + w]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            roi_dark = cv2.morphologyEx(roi_dark, cv2.MORPH_CLOSE, kernel, iterations=4)
            roi_dark = cv2.bitwise_not(roi_dark)
            bright_area_defects_count = np.count_nonzero(roi_dark)

            if bright_area_defects_count > 0:
                return ori_image, 1
            else:
                roi_dark_com = image[y:y + h, x:x + w]
                # roi_dark_x = cv2.Sobel(roi_dark_com, cv2.CV_32F, 2, 0, borderType=cv2.BORDER_CONSTANT)
                roi_dark_y = cv2.Sobel(roi_dark_com, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_CONSTANT)
                # roi_dark = cv2.bitwise_or(roi_dark_x, roi_dark_y)
                roi_dark = cv2.convertScaleAbs(roi_dark_y)
                h_, w_ = roi_dark.shape[:2]
                roi_dark[0:5] = 0
                roi_dark[h - 5:h] = 0
                roi_dark = cv2.medianBlur(roi_dark, 3)

                dark_area_defects_count = np.count_nonzero(roi_dark)

                if bright_area_defects_count > 0:
                    return ori_image, 2

            # roi_dark = np.hstack([roi_dark_com, roi_dark])

            # t2 = datetime.datetime.now()
            # print('time: {}'.format(t2 - t1))

            return ori_image, 0

        else:
            return None, -1

    @staticmethod
    def fix_rotation(*images, base_roi, inter=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
        coords = np.column_stack(np.where(base_roi > 0))  # get all non-zero pixel coords
        anchor, size, angle = cv2.minAreaRect(coords)  # bound them with a rotated rect
        # angle of minAreaRect is confusing, recommends to a good answer here
        # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        center = (anchor[0] + size[0] / 2, anchor[1] + size[1] / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)

        h, w = base_roi.shape[:2]
        img = cv2.warpAffine(base_roi, M, (w, h), flags=inter, borderMode=borderMode, borderValue=borderValue)

        imgs = [base_roi]
        for img in images:
            h, w = img.shape[:2]
            img = cv2.warpAffine(img, M, (w, h), flags=inter, borderMode=borderMode, borderValue=borderValue)
            imgs.append(img)
        return imgs

    @staticmethod
    def rescale(image):
        minval = np.min(image)
        maxval = np.max(image)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    fmt = FileMonitorTest('../test/aa1/EW1/2018/05/27', 1)
    fmt.start()

    fm = FileMonitor('../test/aa1/EW1/2018/05/27', 1, 100, 'glue.jpg')
    fm.start()

    IP = ImageProc(fm.image_queue)
    IP.start()
    while True:
        if IP.image is not None:
            cv2.imshow('test', IP.image)
            cv2.waitKey(1)
        else:
            time.sleep(1)
