import os
import time

from threading import Thread
import queue

from file_check.file_monitor_test import FileMonitorTest


class FileMonitor:
    def __init__(self, monitor_path, time_pause=1, queue_size=100, *file_filter):
        if len(file_filter) == 0:
            file_filter = 'jpg', 'jpeg', 'png', 'bmp'
        self.monitor_path = monitor_path
        self.time_pause = time_pause
        self.file_filter = file_filter
        self.old_list = []

        # output
        self.image_queue = queue.Queue(maxsize=queue_size)

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
                new_list = os.listdir(self.monitor_path)

                new_list_filtered = []
                for l in new_list:
                    l = self.end_with(l, self.file_filter)
                    if l is not None:
                        new_list_filtered.append(l)

                new_files = list(set(new_list_filtered).difference(set(self.old_list)))
                self.old_list = new_list_filtered

                # do something
                # self.new_files_path = [self.monitor_path + '/' + p for p in new_files]
                # print(self.new_files_path)

                for p in new_files:
                    new_files_path = self.monitor_path + '/' + p
                    self.image_queue.put(new_files_path)
                    # print(self.image_queue.qsize())

                time.sleep(self.time_pause)

            # else:
            #     os.makedirs(self.monitor_path)

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

    fm = FileMonitor('../test/aa1/EW1/2018/05/27', 1, 100, 'glue.jpg')
    fm.start()
    while True:
        time.sleep(1)
