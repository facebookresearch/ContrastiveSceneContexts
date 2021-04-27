import os
import numpy as np


# ScanNet
path = '/checkpoint/jihou/data/scannet/pointcloud/splits'
txtfile = 'scannetv2_train.txt'
ratios = [0.40, 0.80]
cross_val = 3
train_scenes = open(os.path.join(path, txtfile)).readlines()
import ipdb; ipdb.set_trace()

for i in range(cross_val):
    for ratio in ratios:
        output_txtfile = 'scannetv2_train_{}_{}.txt'.format(ratio, i)
        choice = np.random.choice(len(train_scenes), int(len(train_scenes)*ratio), replace=False)
        output_scenes = []
        for choice_ in choice:
            output_scenes.append(train_scenes[choice_])
        f = open(output_txtfile, 'w')
        f.writelines(output_scenes)
        f.close()

## Stanford
#path = '/checkpoint/jihou/data/scannet/pointcloud/splits'
#txtfile = 'scannetv2_train.txt'
#ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
#train_scenes = open(os.path.join(path, txtfile)).readlines()
#for ratio in ratios:
#    output_txtfile = os.path.join(path, 'scannetv2_train_{}.txt'.format(ratio))
#    output_scenes = train_scenes[:int(len(train_scenes) * ratio)]
#    f = open(output_txtfile, 'w')
#    f.writelines(output_scenes)
#    f.close()









