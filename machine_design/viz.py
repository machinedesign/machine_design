import numpy as np

def horiz_merge(left, right):
    assert left.shape[0] == right.shape[0]
    assert left.shape[2:] == right.shape[2:]
    shape = (left.shape[0], left.shape[1] + right.shape[1],) + left.shape[2:]
    im_merge = np.zeros(shape)
    im_merge[:, 0:left.shape[1]] = left
    im_merge[:, left.shape[1]:] = right
    return im_merge

def grid_of_images(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape = None, normalize=False):
    if len(M.shape) == 3:
        M = M[:, :, :, np.newaxis]
    if M.shape[-1] not in (1, 3):
        M = M.transpose((0, 2, 3, 1))
    if M.shape[-1] == 1:
        M = np.ones((1, 1, 1, 3)) * M
    bordercolor = np.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()

    if normalize:
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    if shape is None:
        n0 = np.int(np.ceil(np.sqrt(numimages)))
        n1 = np.int(np.ceil(np.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]

    im = np.array(bordercolor)*np.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = np.concatenate((
                  np.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*np.ones((height,border,3),dtype=float)), 1),
                  bordercolor*np.ones((border,width+border,3),dtype=float)
                  ), 0)
    return im
