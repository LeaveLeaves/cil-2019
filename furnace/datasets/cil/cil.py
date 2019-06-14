#!/usr/bin/env python3
# encoding: utf-8
import numpy as np

from datasets.BaseDataset import BaseDataset


class Cil(BaseDataset):
    trans_labels = [0, 1]

    @classmethod
    def get_class_colors(*args):
        return [[128, 64, 128], [119, 11, 32]]

    @classmethod
    def get_class_names(*args):
        # class counting(gtFine)
        # 2953 2811 2934  970 1296 2949 1658 2808 2891 1654 2686 2343 1023 2832
        # 359  274  142  513 1646
        return ['road', 'non-road']

    @classmethod
    def transform_label(cls, pred, name):
        # should be removed
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        print('Trans', name, 'to', new_name, '    ',
              np.unique(np.array(pred, np.uint8)), ' ---------> ',
              np.unique(np.array(label, np.uint8)))
        return label, new_name