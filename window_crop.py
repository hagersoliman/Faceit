import imageio
import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte
from mtcnn import MTCNN
import cv2
import face_alignment
import numpy as np

def get_initial_BBox(initFrame):
  detector = MTCNN()
  boxes = detector.detect_faces(initFrame)
  conf = 0
  initBbx = None
  for box in boxes:
    if box['confidence'] > conf and box['confidence'] > 0.5 :
      initBbx = box['box']
  return initBbx 


def get_alignemnt_based_bbx(initFrame):
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
  bboxes =  extract_bbox(initFrame, fa)
  maxIou = 0
  initBbx = get_initial_BBox(initFrame)
  x, y, w, h = initBbx
  initBox = [x, y, x+w, y+h]
  selectedBbx = None
  for box in bboxes:
    bbx = [ int(dim) for dim in box]
    iou = bb_intersection_over_union(bbx, initBox)
    if  iou > maxIou:
      selectedBbx = bbx
      maxIou = iou
  x, y, w, h = selectedBbx [0], selectedBbx[1], selectedBbx[2] - selectedBbx [0], selectedBbx[3] - selectedBbx[1]
  return x, y, w, h 

def compute_bbox( left, top, width, height , frame_shape, increase_area=0.1):
    right, bot =  (left+ width, top+height)
    print("compute", left,top, right,bot)
    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left
    return left, top, w, h
    
def single_frame_tracker_based_cropping(initFrame,initBbx,  frame):
  ok, bbox = tracker.update(frame)
  x, y, w, h =  int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
  newbbx = compute_bbox(w, h, x, y , initFrame.shape, increase_area=0.1)
  x, y, w, h = tuple(newbbx)
  sub_face = frame[y:y+h, x:x+w]
  resize(sub_face, (256, 256))[..., :3]
  return sub_face

def single_frame_fixedCrop_based_cropping(initBbx):
  x, y, w, h =  int(initBbx[0]), int(initBbx[1]), int(initBbx[2]), int(initBbx[3])
  sub_face = frame[y:y+h, x:x+w]
  resize(sub_face, (256, 256))[..., :3]
  return sub_face

import numpy as np
def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        #print(frame.shape, scale_factor)
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        #print(frame.shape)
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    #print(bboxes)
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# !pip install mtcnn
# !git clone https://github.com/1adrianb/face-alignment
# cd /content/first-order-model/face-alignment
# !python /content/first-order-model/face-alignment/setup.py install


# initFrame = None
# tracker = cv2.TrackerMIL_create()
# tracker.init(initFrame,tuple(initBbx))
  
# initBbx = get_alignemnt_based_bbx(initFrame)
# newinitbbx = compute_bbox(x, y, w, h , initFrame.shape, increase_area=0.1)
# frame = None
# if opt == "tracker":
#   sub_face =  single_frame_tracker_based_cropping(initFrame,initBbx,  frame)
# elif opt == "fixed":
#   sub_face =  single_frame_fixedCrop_based_cropping(initBbx)
