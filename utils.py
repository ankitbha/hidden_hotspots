

import numpy as np


# extract contiguous portions for an arrays
def contiguous(arr):
    valids = ~np.isnan(arr)
    
    count = 0
    startind = 0
    state = False
    
    for ind, ele in enumerate(valids):
        if state:
            if ele:
                count += 1
            else:
                state = False
                yield arr[startind:startind+count], np.arange(startind, startind+count)
                count = 0
        else:
            if ele:
                state = True
                startind = ind
                count = 1

    yield arr[startind:startind+count], np.arange(startind, startind+count)


# extract largest contiguous portion for list of arrays
def extract_contiguous(arr_list):

    mat = np.hstack([np.vstack(arr) for arr in arr_list])
    valids = (~np.isnan(mat)).prod(axis=1)

    count = 0
    startind = 0
    maxcount = 0
    maxstartind = 0
    
    state = False
    
    for ind, ele in enumerate(valids):
        if state:
            if ele:
                count += 1
            else:
                if count > maxcount:
                    maxcount = count
                    maxstartind = startind
                state = False
                count = 0
        else:
            if ele:
                state = True
                startind = ind
                count = 1

    if state:
        if count > maxcount:
            maxcount = count
            maxstartind = startind

    return mat[maxstartind:maxstartind+maxcount,:], np.arange(maxstartind, maxstartind+maxcount)


# compute correlation coefficient between two arrays with gaps
def nancorrcoef(x, y):

    assert len(x) == len(y)
    x = np.asarray(x)
    y = np.asarray(y)
    assert (x.ndim == 1) & (y.ndim == 1)
    
    # something wrong about the method below; not clear what.
    
    # dev_x = np.asarray(x) - np.nanmean(x)
    # dev_y = np.asarray(y) - np.nanmean(y)
    # print(np.nanmean(x), np.nanmean(y))
    # print(dev_x)
    # print(dev_y)
    # num = np.nanmean(dev_x * dev_y)
    # denom = np.sqrt(np.nanmean(dev_x**2) * np.nanmean(dev_y**2))
    # return (num / denom)

    # alternative way to handle gaps; just remove elements are those
    # positions where there is a gap in either x or y
    validpos = (~np.isnan(x)) & (~np.isnan(y))
    x = x[validpos]
    y = y[validpos]
    if len(x) == 0:
        return np.nan, 0
    else:
        return np.corrcoef(x,y)[0,1], len(x)


if __name__=='__main__':

    # x = [1,2,np.nan,-10,3,4,5,np.nan,-10,6,7]
    # y = [1,2,-10,np.nan,3,4,5,-10,np.nan,6,7]
    x = [1,np.nan,3,4]
    y = [2,3,np.nan,5]
    
    # print(extract_contiguous([x, y]))
    print(nancorrcoef(x,y))
