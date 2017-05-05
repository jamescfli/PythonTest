from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

dataDir = '..'
dataType = 'train2014'

# 1) instance segmentation
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# display coco categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# print 'COCO categories: \n', ', '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
# print 'COCO supercategories: \n', ', '.join(nms)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
# print catIds
imgIds = coco.getImgIds(catIds=catIds)
# print imgIds.__len__()
# print imgIds
# # train2014.zip is not fully downloaded
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
img = coco.loadImgs(imgIds[0])[0]

# load and display
I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name']))
plt.figure()
plt.axis('off')
plt.imshow(I)
# plt.show()

# load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)    # iscrowd: anns for given crowd label
print annIds
anns = coco.loadAnns(annIds)    # list, len = 6
# [u'segmentation',
#  u'area',
#  u'iscrowd',
#  u'image_id',
#  u'bbox',
#  u'category_id',
#  u'id']
print anns
coco.showAnns(anns)
plt.show()

# 2) show person keypoints
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
coco_kps = COCO(annFile)

# load and display keypoints
plt.imshow(I)
plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)    # list, len = 6, dict
# [u'segmentation',
#  u'num_keypoints',    # new
#  u'area',
#  u'iscrowd',
#  u'keypoints',        # new
#  u'image_id',
#  u'bbox',
#  u'category_id',
#  u'id']
coco_kps.showAnns(anns)
plt.show()

# 3) caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
coco_caps = COCO(annFile)

annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)       # len = 5
# In: anns[0]
# Out:
# {u'caption': u'A person walking a dog on a leash down a street.',
#  u'id': 717924,
#  u'image_id': 379520}
# In: anns[1]
# Out:
# {u'caption': u'THIS IS A GIRL ON A SKATEBOARD WALKING HER PIT BULL',
#  u'id': 724539,
#  u'image_id': 379520}
# ...
# In: anns[4]
# Out[52]:
# {u'caption': u'Her pit bull takes the lead when she is on her skateboard.',
#  u'id': 730176,
#  u'image_id': 379520}
coco_caps.showAnns(anns)
plt.imshow(I)
plt.axis('off')
plt.show()
