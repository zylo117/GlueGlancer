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
                self.crop(image_path)

    def crop(self, image_path):
        image = cv2.imread(image_path)
        print(np.mean(image))
        self.image = image

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
