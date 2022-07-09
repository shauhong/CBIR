# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

d_type = "d1"
depth = 10
descriptors = ['correlation', 'contrast',
               'dissimilarity', 'energy', 'homogeneity', 'ASM']
distances = [1, 2, 3, 4]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]


class GLCM:
    def make_samples(self, db, normalize=True, verbose=True):
        if verbose:
            print("Counting histogram..., distance=%s, depth=%s" %
                  (d_type, depth))
        samples = []
        data = db.get_data()
        for d in data.itertuples():
            d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

            img = cv2.imread(d_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            glcm = graycomatrix(img, distances=distances,
                                angles=angles, levels=256)

            d_hist = graycoprops(glcm, "correlation")
            d_hist = d_hist.flatten()

            if normalize:
                d_hist /= d_hist.sum()

            samples.append({
                'img': d_img,
                'cls': d_cls,
                'hist': d_hist
            })

        return samples


if __name__ == "__main__":
    # evaluate database
    db = Database()
    APs = evaluate_class(db, f_class=GLCM, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls,  MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
