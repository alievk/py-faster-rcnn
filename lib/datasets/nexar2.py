import cPickle
from collections import defaultdict
import os
import sys
import numpy as np
import pandas as pd
from datasets.imdb import imdb
from fast_rcnn.config import cfg

class nexar2(imdb):
    def __init__(self, image_set):
        # of imdb
        imdb.__init__(self, 'nexar2_' + image_set)
        # in Nexar 2 challenge vehicle classes (car, bus, etc) does not matter
        self._classes = ('__background__',
                         'car')

        # of nexar2
        self._data_path = os.path.join(cfg.DATA_DIR, 'nexar2')
        assert os.path.exists(self._data_path), 'Path {} does not exist'.format(self._data_path)
        self._anno_path = os.path.join(self._data_path, image_set + '.csv')
        assert os.path.exists(self._anno_path), 'Path {} does not exist'.format(self._anno_path)
        self._gt_roidb = []
        self._image_names = []
        self._object_class_ind = 1
        self._image_width = 1280
        self._image_height = 720
        self.load_annos()

    def load_annos(self):
        """
        Create ground truth boxes/ROIs database.

        Return list of items of the following format:
                {
                'image' : image path,
                'boxes' : Bx4 (B=num of boxes),
                'gt_classes': B,
                'gt_overlaps': None,
                'flipped' : False
                }
        """
        annos_all = pd.read_csv(self._anno_path, index_col='image_filename')
        image_names = list(set(annos_all.index.values))

        self._image_names = image_names
        self._image_index = range(len(image_names))

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._gt_roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        else:
            def fix_bbox(bbox):
                x0, y0, x1, y1 = bbox
                x0 = max(0., min(self._image_width-1., x0))
                y0 = max(0., min(self._image_height-1., y0))
                x1 = max(0., min(self._image_width-1., x1))
                y1 = max(0., min(self._image_height-1., y1))
                return [x0, y0, x1, y1]

            print "Creating nexar2 roidb..."
            for image_name in image_names:
                ann = annos_all.loc[image_name]

                # DataFrame.loc[] will return a Series instance if there is only one annotation for the image
                # otherwise it will return DataFrame
                boxes_list = []
                cls_list = []
                if ann.ndim == 1:  # Series instance
                    x0 = min(self._image_width-1, ann.x0)
                    boxes_list.append(fix_bbox([ann.x0, ann.y0, ann.x1, ann.y1]))
                    cls_list.append(self._object_class_ind)
                else:  # DataFrame instance
                    for row in ann.itertuples():
                        boxes_list.append(fix_bbox([row.x0, row.y0, row.x1, row.y1]))
                        cls_list.append(self._object_class_ind)

                num_objs = len(boxes_list)
                overlaps = np.zeros((num_objs, 2), dtype=np.float32)
                overlaps[:, 1] = 1.
                gt_classes = np.array(cls_list, dtype=np.uint32)
                self._gt_roidb.append(
                    {
                        'image': os.path.join(self._data_path, 'train', image_name),
                        'boxes': np.array(boxes_list, dtype=np.uint16),
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': gt_classes,
                        'max_overlaps': overlaps,
                        'flipped': False
                    }
                )

            with open(cache_file, 'wb') as fid:
                cPickle.dump(self._gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt roidb to {}'.format(cache_file)

    def gt_roidb(self):
        return self._gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # this should not be called -- image paths is already in the roidb
        return os.path.join(self._data_path, 'train', self._image_names[self._image_index[i]])

    def _get_widths(self):
        return [self._image_width] * self.num_images

    def evaluate_detections(self, all_boxes, output_dir=None):
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)

        # will transform this in pandas DataFrame
        det_dict = defaultdict(list)

        # ignore background (0) class
        for img_ind, det in enumerate(all_boxes[1]):
            img_name = self._image_names[img_ind]
            if det != []:
                for x0, y0, x1, y1, score in det:
                    det_dict['image_filename'].append(img_name)
                    det_dict['x0'].append(x0)
                    det_dict['y0'].append(y0)
                    det_dict['x1'].append(x1)
                    det_dict['y1'].append(y1)
                    det_dict['label'].append('car')
                    det_dict['confidence'].append(score)

        # save detections in csv file
        dt_csv = os.path.join(output_dir, 'detections.csv')
        val_det_df = pd.DataFrame(det_dict, columns=['image_filename','x0','y0','x1','y1','label','confidence'])
        val_det_df.to_csv(dt_csv, index=False)

        # evaluate detections
        import eval_challenge
        eval_challenge.DEBUG = True
        gt_csv = self._anno_path
        iou_threshold = 0.75
        print ('AP: {}'.format(eval_challenge.eval_detector_csv(gt_csv, dt_csv, iou_threshold)))