import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[2]
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print 'Running demo for *{}* results.'.format(annType)

dataDir = '../'
dataType = 'val2014'
annFile = '{}/annotations/{}_{}.json'.format(dataDir, prefix, dataType)
cocoGt = COCO(annFile)
# where
#   cocoGt.getImgIds().__len__() = 40504
#   cocoGt.getAnnIds().__len__() = 291875
#   cocoGt.getCatIds().__len__() = 80       # 1~90 something missed within

# initialize COCO detection api
resFile = '{}/results/{}_{}_fake{}100_results.json'
resFile = resFile.format(dataDir, prefix, dataType, annType)
cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds     # o.w. it will evaluate whole set of images
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
