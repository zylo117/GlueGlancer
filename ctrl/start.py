import datetime

import numpy as np
import os

# parse config
dt = str(datetime.date.today()).split('-')

conf = open('./ip-name.csv', 'r').readlines()
conf = conf[1:]
conf = [c.strip('\n').split(',') for c in conf]
conf = np.array(conf)

addr_by_sut = []
for i in range(len(conf[:,])):
    addr = '\\\\' + conf[:, 1][i // 4] + '\\d$\\EpocyInsp\\EW{}\\{}\\{}\\{}'.format(i % 4 + 1, dt[0], dt[1], dt[2])
    addr_by_sut.append(addr)





print()
