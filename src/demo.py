# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

from fusion import FeatureFusion
import sys
import os
import shutil

depth = 5
d_type = 'd1'
query_idx = int(sys.argv[1])

if __name__ == '__main__':
    db = Database()
    method = FeatureFusion(features=['color', 'hog', 'glcm'])
    samples = method.make_samples(db)
    query = samples[query_idx]
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)

    os.makedirs(f"result/retrieval_result/{query_idx}", exist_ok=True)

    src = query['img']
    dest = f"result/retrieval_result/{query_idx}/query.jpg"
    shutil.copy(src, dest)

    for r in result:
        src = r['img']
        img = r['img'].split("/")[-1]
        dest = f"result/retrieval_result/{query_idx}/{img}"
        shutil.copy(src, dest)
    print("Query:")
    print(f"img: {query['img']} cls: {query['cls']}")
    print("Result:")
    print(result)
