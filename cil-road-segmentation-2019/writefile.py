import os
from os import walk
import glob

dataset_path = os.path.dirname(os.path.abspath(__file__))

gt = []
img = []
test = []
print(dataset_path)
# get img
for name in glob.glob('training/images/*'):
    img.append(name)
# get gt
for name in glob.glob('training/groundtruth/*'):
    gt.append(name)
# get test
for name in glob.glob('test_images/*'):
    test.append(name)

img.sort()
gt.sort()
print("{:10d} images {:10d} groundtruth".format(len(img), len(gt)))

# train.txt
with open('train.txt', 'w') as f:
    for i in range(90):
        line = img[i] + '\t' + gt[i] + '\n'
        f.write(line)
f.close()

# val.txt
with open('val.txt', 'w') as f:
    for i in range(90, 100):
        line = img[i] + '\t' + gt[i] + '\n'
        f.write(line)
f.close()


# test.txt
with open('test.txt', 'w') as f:
    for i in range(len(test)):
        line = test[i] + '\t' + gt[i] + '\n'
        f.write(line)
f.close()
