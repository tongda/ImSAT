from json import encoder

import pylab

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

dataDir = './data'
dataType = 'val2014'
algName = 'fakecap'
annFile = '%s/coco/annotations/captions_%s.json' % (dataDir, dataType)
subtypes = ['results', 'evalImgs', 'eval']
resFile = "result.json"
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()
