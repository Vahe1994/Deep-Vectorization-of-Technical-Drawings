import numpy as np

def image_coord(img, MAX_X =64, MAX_Y = 64):
    img = np.array(img)
    a = np.repeat(np.arange(0,MAX_X),MAX_X).reshape((MAX_X,MAX_Y)) * 255.0 /img.shape[1]
    im = np.zeros((3,img.shape[0],img.shape[1])) 
    mask = img/255.0
    mask[mask == 1 ] = -1
    mask[mask != -1] = 1
    mask[mask == -1] = 0
    im[0, :] = img
    im[1, :] =  a * mask
    im[2, :] = a.T * mask
    return im

def order(arr):
    mass = []
    if(np.array(arr).shape[1] == 5):
        # it's a line
        for it in arr:
            a = np.round(np.abs(np.array(it)))
            if (a[0],a[1])<=(a[2],a[3]):
                mass.append((a[0],a[1],a[2],a[3],a[4]))
            else:
                mass.append((a[2],a[3],a[0],a[1],a[4]))         
    elif(np.array(arr).shape[1] == 9):
        # it's a bezier
        mass = arr
    elif(np.array(arr).shape[1] == 4):
        # it's a arc
        mass = arr
    else:
        raise ValueError('not line,not curve,not circle')

    mass.sort()
    return mass