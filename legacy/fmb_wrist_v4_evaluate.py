import numpy as np
TIMES=np.arange(1,21)/10


def align(future,initial):
    source=np.r_[0,np.arange(1,future.shape[1]+1)/15]
    values=np.concatenate([initial[:,None],future],axis=1)
    return np.array([[np.interp(TIMES,source,values[n,:,a]) for a in range(3)] for n in range(24)]).transpose(0,2,1)
