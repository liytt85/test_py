import json
import numpy as np
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from scipy import misc
import matplotlib.pyplot as plt
from pylab import *

json_name = '/home/lyt/下载/run-trpo_gail.Walker2d.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001aa-tag-summary_2-True_rewards.scalar.summary_1.json'
#json_name2 = '/home/lyt/下载/run-trpo_gail.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001aaa-tag-summary_2-True_rewards.scalar.summary_1.json'
json_file = open(json_name)
#json_file2 = open(json_name2)
annos = json.load(json_file)
#annos1 = json.load(json_file2)
print(type(annos))
plt.figure()
#im = misc.imread(input_name)
# plt.imshow(im)
out_name = '/home/lyt/cfile/1111.png'
lenth = len(annos)
x = []
y = []
exper = []
x1 = []
y1 = []
for i in range(500):
    x.append(annos[i][1])
    y.append(annos[i][2])
    exper.append(2459)
'''for i in range(700):
    x1.append(annos1[i][1])
    y1.append(annos1[i][2])'''
plt.plot(x, y, color='yellow', label='model')
plt.plot(x, exper, color='blue', label="expert")

'''curr = annos['shapes']
# print curr[0]['points'][0]
for item in curr:
    if item['label'][:5] == 'Empty':
        continue
    x = item['points'][0]
    plot(x[0], x[1], 'ro')'''

# show()
plt.xlabel('steps')
plt.ylabel('reward')
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig(out_name, dpi=1000)

plt.show()

a = 1
b = a
b -= 1
print(a)
