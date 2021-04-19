"""
Generate mask by comparing local variance and global variance
"""
import numpy as np

def maxmask(tomo, side=5,percentile=60):
    from scipy.ndimage.filters import maximum_filter
    print('maximum_filter')
    filtered = maximum_filter(-tomo, 2*side+1, mode='reflect')
    out =  filtered > np.percentile(filtered,100-percentile)
    out = out.astype(np.uint8)
    return out

def stdmask(tomo,side=10,threshold=1):
    from scipy.signal import convolve
    print('std_filter')
    tomosq = tomo**2
    ones = np.ones(tomo.shape)
    eps = 0.01
    kernel = np.ones((2*side+1, 2*side+1, 2*side+1))
    s = convolve(tomo, kernel, mode="same")
    s2 = convolve(tomosq, kernel, mode="same")
    ns = convolve(ones, kernel, mode="same")

    out = np.sqrt((s2 - s**2 / ns) / ns + eps)
    out = out>np.std(tomo)*threshold
    return out.astype(np.uint8)

# def gauss

if __name__ == "__main__":
    import sys
    import mrcfile
    args = sys.argv
    with mrcfile.open(args[1]) as n:
        tomo = n.data
    mask = stdmask_mpi(tomo,cubelen=20,cubesize=80,ncpu=20,if_rescale=True)
    with mrcfile.new(args[2],overwrite=True) as n:
        n.set_data(mask)