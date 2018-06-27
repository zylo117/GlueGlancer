import datetime
import time

import numpy as np

# parse config
from file_check.file_monitor import FileMonitor
from glue_check.glue_viewer import ImageProc

dt = str(datetime.date.today()).split('-')

conf = open('./ip-name.csv', 'r').readlines()
conf = conf[1:]
conf = [c.strip('\n').split(',') for c in conf]
conf = np.array(conf)

addr_by_sut = []
for i in range(len(conf[:, ])):
    addr = '\\\\' + conf[:, 1][i // 4] + '\\d$\\EpocyInsp\\EW{}\\{}\\{}\\{}'.format(i % 4 + 1, dt[0], dt[1], dt[2])
    addr_by_sut.append([conf[:, 0][i // 4], i % 4 + 1, addr])
addr_by_sut = np.array(addr_by_sut)

# start N threads to monitor so many machines and process images
for addr in addr_by_sut:
    fm = FileMonitor(addr[2], 1, 100, 'glue.jpg')
    fm.start()
    print('monitoring {} SUT{}'.format(addr[0], addr[1]))

    IP = ImageProc(fm.image_queue, machine=addr[0], station=addr[1], output=True)
    IP.start()
    print('processing {} SUT{}'.format(addr[0], addr[1]))

while True:
    time.sleep(1)
