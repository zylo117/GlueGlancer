import os
import shutil
import time

from threading import Thread
import numpy.random as random


class FileMonitorTest:
    def __init__(self, monitor_path, time_pause=1, max_file=100):
        self.monitor_path = monitor_path
        self.time_pause = time_pause
        self.index = 0
        self.max_file = max_file

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        t = Thread(target=self.monitor, args=(), daemon=True)
        t.start()
        return self

    def monitor(self):
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            if os.path.exists(self.monitor_path):
                # do something
                try:
                    shutil.copy('../test/%d.jpg' % random.randint(0, 10),
                                self.monitor_path + '/' + '%dglue.jpg' % self.index)
                except:
                    pass

                if self.index == self.max_file - 1:
                    print('finished')
                    self.stop()
                    return

                time.sleep(self.time_pause)
                self.index += 1

            else:
                os.makedirs(self.monitor_path)

    @staticmethod
    def end_with(filename, filters):
        for f in filters:
            if f in filename:
                return filename

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    fmt = FileMonitorTest('../test/aa1/EW1/2018/05/27', 1)
    fmt.start()
    while True:
        time.sleep(1)
