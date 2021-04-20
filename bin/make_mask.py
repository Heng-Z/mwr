#!/usr/bin/env python3

import sys
import mrcfile
args = sys.argv
from IsoNet.util.filter import maxmask,stdmask
import numpy as np
#import cupy as cp
import os

def make_mask_dir(tomo_dir,mask_dir,side = 8,percentile=30,threshold=1,surface=None):
    tomo_list = ["{}/{}".format(tomo_dir,f) for f in os.listdir(tomo_dir)]
    try:
        os.makedirs(mask_dir)
    except FileExistsError:
        import shutil
        shutil.rmtree(mask_dir)
        os.makedirs(mask_dir)
    mask_list = ["{}/{}_mask.mrc".format(mask_dir,f.split('.')[0]) for f in os.listdir(tomo_dir)]

    for i,tomo in enumerate(tomo_list):
        print('tomo and mask',tomo, mask_list[i])
        make_mask(tomo, mask_list[i],side = side,percentile=percentile,threshold=threshold,surface=surface)


def make_mask(tomo_path, mask_name,side = 5,percentile=30,threshold=1.0,surface=None):
    from scipy.ndimage.filters import gaussian_filter
    with mrcfile.open(tomo_path) as n:
        tomo = n.data
    sp=np.array(tomo.shape)
    bintomo = tomo[0:-1:2,0:-1:2,0:-1:2].astype(np.float32)
    sp2 = bintomo.shape
    gauss = gaussian_filter(bintomo, side/4)
    if percentile is not None:
        mask1 = maxmask(gauss,side=side, percentile=percentile)
    else:
        mask1 = np.ones(sp2)
    if threshold is not None:
        mask2 = stdmask(gauss,side=side, threshold=threshold)
    else:
        mask2 = np.ones(sp2)
    out_mask_bin = np.multiply(mask1,mask2)
        # out_mask_bin = mask1
    if (surface is not None) and surface < 1:
        for i in range(int(surface*sp2[0])):
            out_mask_bin[i] = 0
        for i in range(int((1-surface)*sp2[0]),sp2[0]):
            out_mask_bin[i] = 0
    out_mask = np.zeros(sp)
    out_mask[0:-1:2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,0:-1:2,1::2] = out_mask_bin
    out_mask[0:-1:2,1::2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,1::2,1::2] = out_mask_bin
    out_mask[1::2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[1::2,0:-1:2,1::2] = out_mask_bin
    out_mask[1::2,1::2,0:-1:2] = out_mask_bin
    out_mask[1::2,1::2,1::2] = out_mask_bin
    out_mask = (out_mask>0.5).astype(np.uint8)
    with mrcfile.new(mask_name,overwrite=True) as n:
        n.set_data(out_mask)

    # with mrcfile.new('./test_mask1.rec',overwrite=True) as n:
    #     n.set_data(mask1.astype(np.float32))    
    # with mrcfile.new('./test_mask2.rec',overwrite=True) as n:
    #     n.set_data(mask2.astype(np.float32))
if __name__ == "__main__":
# the first arg is tomo name the second is mask name
    make_mask(args[1],args[2])

