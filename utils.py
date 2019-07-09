

import numpy as np


# extract largest contiguous portion for list of arrays
def extract_contiguous(arr_list):

    mat = np.hstack([np.vstack(arr) for arr in arr_list])
    valids = (~np.isnan(mat)).prod(axis=1)

    count = 0
    maxcount = 0
    startind = 0
    state = False
    
    for ind, ele in enumerate(valids):
        if state:
            if ele:
                count += 1
            else:
                if count > maxcount:
                    maxcount = count
                    startind = ind - count
                state = False
                count = 0
        else:
            if ele:
                state = True
                count = 1

    return mat[startind:startind+maxcount,:]


if __name__=='__main__':

    print(extract_contiguous([[1,2,np.nan,-10,3,4,5,np.nan,-10,6,7],
                              [1,2,-10,np.nan,3,4,5,-10,np.nan,6,7]]))
        
        
