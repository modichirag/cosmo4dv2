import numpy as np
from datatools import *

def balanced27gridpt(pm, ftlist, base, lim, ratio = 1, clsgrid = None, halopos = None, hlim = 0, clim = 0,\
                     ratiolow = 0, classify = True, ppx='std', ppy='std', testdata=True, hposf = None):
    '''For classification
    Y = data is the value at the point on halogrid ('clsgrid')
    Training set = 27 neighboring points on ftlist at halo position supplemented with *ratio multiple of points
    above lim on base and *ratiolow multiple of points below lim
    Test = value at the all the grid points    
    '''
    if (halopos is None) & (clsgrid is None):
        print('Need atleast a class-grid or halo positions')
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'

    if hposf is None:
        #We are not going to finer halos, so just give the same(correct) classgrid. Halopos is only used
        #in case clsgrid turns out to be None. if its not None, halopos is not used.
        dind = balancepts(pm, base, blim=lim, hlim=hlim, ratiohigh=ratio, ratiolow=ratiolow, clsgrid=clsgrid, halopos=halopos)
    else:
        #Since we are going to finer halos, set clsgrid to None so that halopos is used and since halopos=hposf means it has 
        #finer halos, thus the dind will have all the indices for halos above this fine limit. The dind positions with halos
        #less than numd but above min(hposf) will be set to class 0 below when assigning classes.
        #ratiolow in this case will select more points below the blim, which is overkill if we already have finer halos 
        dind = balancepts(pm, base, blim=lim, hlim=hlim, ratiohigh=ratio, ratiolow=ratiolow, clsgrid=None, halopos=hposf)
        
    dind = tuple(dind.T)

    #Assign classes
    if clsgrid is None:
        clsgrid = np.zeros([nc, nc, nc])
        clsgrid[tuple((halopos/side).astype(int).T)] = 1

    cls = clsgrid[dind]
    if classify:
        cls[cls>clim] = 1
        cls[cls<=clim] = 0

    ndim = len(ftlist)
    print('Number of features is =',ndim)

    #Read features for training and normalize
    ft = np.zeros([dind[0].size, ndim*27])
    for boo in range(ndim):
        ft[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], ind=dind)
    
    #Normalize
    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    cls, my, sy =  normdict[ppy](cls)
    if my is None:
        print('Not Normalized')

    # Read features over the whole grid to generate data to test performance
    if testdata:
        ftt = np.zeros([pm.Nmesh[0]**3, 27*ndim])
        for boo in range(ndim):
            ftt[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], pm=pm)
        fttn = normdict[ppx](ftt, mx, sx)

    else:
        ftt, fttn = None, None
    return [ftn, mx, sx], [cls, my, sy], [fttn, dind, ftt]





#####
def balanced27gridptmsub(pm, ftlist, base, lim,  clsgridall, clsgrid, hposbelow, ratio = 1, halopos = None, hlim = 0, clim = 0,\
                         ratiolow = 0, classify = True, ppx='std', ppy='std', testdata=True):
    '''For classification
    Y = data is the value at the point on halogrid ('clsgrid')
    Training set = 27 neighboring points on ftlist at halo position supplemented with *ratio multiple of points
    above lim on base and *ratiolow multiple of points below lim
    Test = value at the all the grid points    
    '''
    if (halopos is None) & (clsgrid is None):
        print('Need atleast a class-grid or halo positions')
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'


    np.random.seed(seed=100)
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    grid = pm.mesh_coordinates(dtype='i4')
    clshbelow = pm.paint(hposbelow)[...]
    
    maskb = base >= lim
    masknob = ~maskb
    masktrueh = (clsgrid > hlim)
    #masktrueh = (clsgrid > 0)
    maskallh = (clsgridall > 0)
    maskfineh = (clshbelow > 0)

    maskh = masktrueh + maskfineh
    nh = maskh.sum()
    print('Number of halo points & number of points finer =', masktrueh.sum(), maskfineh.sum())
    #print('Number of points in maskh and not maskhall = ', (maskh ^ (maskh & maskallh)).sum())
    maskbnoh = (maskb ^ (maskb & maskallh))
    masknobnoh = (masknob ^ (masknob & maskallh))

    hindex = np.array(np.where(maskh)).T 
    bhindex = np.array(np.where(maskbnoh)).T
    blindex = np.array(np.where(masknobnoh)).T

    argsh = np.random.choice(np.arange(maskbnoh.sum()), int(ratio*nh))
    argsl = np.random.choice(np.arange(masknobnoh.sum()), int(ratiolow*nh))
    bhindex = bhindex[argsh]
    blindex = blindex[argsl]
    print('hindex(true+fine), higher, lower = ', hindex.shape[0], argsh.size, argsl.size)
    dind = np.concatenate([hindex, bhindex, blindex], axis = 0)
    np.random.shuffle(dind)
    
    dind = tuple(dind.T)    
    print('ratios of points with halo vs without halo= ', dind[0].size/masktrueh.sum())
##
    cls = clsgrid[dind]
    if classify:
        cls[cls>clim] = 1
        cls[cls<=clim] = 0

    ndim = len(ftlist)
    print('Number of features is =',ndim)

    #Read features for training and normalize
    ft = np.zeros([dind[0].size, ndim*27])
    for boo in range(ndim):
        ft[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], ind=dind)
    
    #Normalize
    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    cls, my, sy =  normdict[ppy](cls)
    if my is None:
        print('Not Normalized')

    # Read features over the whole grid to generate data to test performance
    if testdata:
        ftt = np.zeros([pm.Nmesh[0]**3, 27*ndim])
        for boo in range(ndim):
            ftt[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], pm=pm)
        fttn = normdict[ppx](ftt, mx, sx)

    else:
        ftt, fttn = None, None
    return [ftn, mx, sx], [cls, my, sy], [fttn, dind, ftt]


#####
def nonzeromask(pm, ftlist, mask, lim, datagrid, fillzero=None, ppx='std', ppy='std', testdata=True):
    '''For regression
    Y = data is the value at the point on halogrid ('base')
    Training set = Y at the positions where mask>lim
    Test = value at the all the grid points
    '''
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'

    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    ndim = len(ftlist)
    pos = np.where(mask > lim)
    ft = np.zeros([pos[0].size, ndim])
    for foo in range(ndim):
        ft[:, foo] = ftlist[foo][pos]
    yy = (datagrid[pos]).reshape(-1, 1)
    if fillzero is not None:
        maskz = yy == 0
        yy[maskz] = (fillzero[pos]).reshape(-1, 1)[maskz]
        yy[yy <= 0] = 0

    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    yn, my, sy =  normdict[ppy](yy)
    if my is None:
        print('Nor Normalized')

    # Test data
    if testdata:
        ftt = np.zeros([nc**3, ndim])
        #pos = pm.mesh_coordinates(dtype='i4')
        for foo in range(ndim):
            #ftt[:, foo] = ftlist[foo][tuple(pos.T)]
            ftt[:, foo] = ftlist[foo].flatten()
        fttn = normdict[ppx](ftt, mx, sx)
    else:
        fttn = None

    return [ftn, mx, sx], [yn, my, sy], [fttn, pos]

#####
def nonzeromask27(pm, ftlist, mask, lim, datagrid, fillzero=None, ppx='std', ppy='std', testdata=True):
    '''For regression
    Y = data is the value at the point on halogrid ('datagrid')
    Training set = Y at the positions where mask>lim
    Test = value at the all the grid points
    '''
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'

    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    ndim = len(ftlist)
    dind = np.where(mask > lim)

    ndim = len(ftlist)
    print('Number of features is =',ndim)

    #Read features for training and normalize
    ft = np.zeros([dind[0].size, ndim*27])
    for boo in range(ndim):
        ft[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], ind=dind)
    yy = (datagrid[dind]).reshape(-1, 1)
    if fillzero is not None:
        maskz = yy == 0
        yy[maskz] = (fillzero[dind]).reshape(-1, 1)[maskz]
        yy[yy <= 0] = 0

    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    yn, my, sy =  normdict[ppy](yy)
    if my is None:
        print('Nor Normalized')

    # Test data
    # Read features over the whole grid to generate data to test performance
    if testdata:
        ftt = np.zeros([pm.Nmesh[0]**3, 27*ndim])
        for boo in range(ndim):
            ftt[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], pm=pm)
        fttn = norm(ftt, mx, sx)        
    else:
        fttn = None
    return [ftn, mx, sx], [yn, my, sy], [fttn]



######                                                                                                                                                      
def abovelimgridpt(pm, ftlist, clsgrid, base, lim, hlim=0, clim=0, ratio=5, ratiolow=0, ppx='std', ppy='std', testdata=True, classify=True, hposf = None):
    '''For classification
    Training set = all the grid points above the limit = lim in array = base                                                                            
    Class = 1 if the grid point is the one closest to halo,0 otherwise                                                                                     
    Test = value at all the grid points of the grid                                                                                                        
    '''
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    ndim = len(ftlist)

    if hposf is None:
        pos = balancepts(pm, base, blim=lim, hlim=hlim, ratiohigh=ratio, ratiolow=ratiolow, clsgrid=clsgrid, halopos=None)
    else:
        pos = balancepts(pm, base, blim=lim, hlim=hlim, ratiohigh=ratio, ratiolow=ratiolow, clsgrid=None, halopos=hposf)
        
    pos = tuple(pos.T)
    #pos = np.where(base > lim)

    ft = np.zeros([pos[0].size, ndim])
    for foo in range(ndim):
        ft[:, foo] = ftlist[foo][pos]

    cls = clsgrid[pos]
    if classify:
        cls[cls>clim] = 1
        cls[cls<=clim] = 0

    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    cls, my, sy =  normdict[ppy](cls)
    if my is None:
        print('Not Normalized')

    # Test data
    if testdata:
        ftt = np.zeros([nc**3, ndim])
        #pos = pm.mesh_coordinates(dtype='i4')
        for foo in range(ndim):
            #ftt[:, foo] = ftlist[foo][tuple(pos.T)]
            ftt[:, foo] = ftlist[foo].flatten()
        fttn = normdict[ppx](ftt, mx, sx)
    else:
        fttn = None

    return [ftn, mx, sx], [cls, my, sy], [fttn, pos]


def all27gridpts(pm, ftlist, base, lim, clsgrid = None, halopos = None, clim = 0,\
                     ppx='std', ppy='std', testdata=True):
    '''For classification
    Y = data is the value at the point on halogrid ('clsgrid')
    Training set = 27 neighboring points on ftlist at halo position supplemented with *ratio multiple of points
    above lim on base and *ratiolow multiple of points below lim
    Test = value at the all the grid points    
    '''
    if (halopos is None) & (clsgrid is None):
        print('Need atleast a class-grid or halo positions')
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'

    #Assign classes
    if clsgrid is None:
        clsgrid = np.zeros([nc, nc, nc])
        clsgrid[tuple((halopos/side).astype(int).T)] = 1

    cls = clsgrid.flatten()
    cls[cls>clim] = 1
    cls[cls<=clim] = 0

    ndim = len(ftlist)
    print('Number of features is =',ndim)

    #Read features for training and normalize
    ft = np.zeros([pm.Nmesh[0]**3, 27*ndim])
    for boo in range(ndim):
        ft[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], pm=pm)
    
    #Normalize
    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    cls, my, sy =  normdict[ppy](cls)
    if my is None:
        print('Not Normalized')

    # Read features over the whole grid to generate data to test performance
    if testdata:
        ftt = np.zeros([pm.Nmesh[0]**3, 27*ndim])
        for boo in range(ndim):
            ftt[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], pm=pm)
        fttn = normdict[ppx](ftt, mx, sx)

##        if ppx == 'std':
##            fttn = norm(ftt, mx, sx)
##        elif ppx == 'minmax':
##            fttn = minmax(ftt, mx, sx)
##        else:
##            print('Test data not normalized')
##            fttn = ftt.copy()
    else:
        ftt, fttn = None, None
    return [ftn, mx, sx], [cls, my, sy], [fttn, None, ftt]




#########                                                                                                                                                 
###def abovefinergridpt(pm, hposf, ftlist, clsgrid, base, lim, hlim=0, ratio=5, ppx='std', ppy='std', testdata=True, classify=True):
###    '''For classification
###    Training set = all the grid points above the limit = lim in array = base                                                                            
###    Class = 1 if the grid point is the one closest to halo,0 otherwise                                                                                     
###    Test = value at all the grid points of the grid                                                                                                        
###    '''
###    if ppx is None:
###        ppx = 'none'
###    if ppy is None:
###        ppy = 'none'
###
###    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
###    ndim = len(ftlist)
###
###    
###    pos = balancepts(pm, base, blim=lim, hlim=hlim, ratiohigh=ratio, ratiolow=0, clsgrid=None, halopos=hposf)
###    pos = tuple(pos.T)
###    #pos = np.where(base > lim)
###
###    ft = np.zeros([pos[0].size, ndim])
###    for foo in range(ndim):
###        ft[:, foo] = ftlist[foo][pos]
###
###    cls = clsgrid[pos]
###    if classify:
###        cls[cls>0] = 1
###        cls[cls<=0] = 0
###
###    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
###    ftn, mx, sx = normdict[ppx](ft)
###    cls, my, sy =  normdict[ppy](cls)
###    if my is None:
###        print('Not Normalized')
###
###    # Test data
###    if testdata:
###        ftt = np.zeros([nc**3, ndim])
###        #pos = pm.mesh_coordinates(dtype='i4')
###        for foo in range(ndim):
###            #ftt[:, foo] = ftlist[foo][tuple(pos.T)]
###            ftt[:, foo] = ftlist[foo].flatten()
###        fttn = normdict[ppx](ftt, mx, sx)
###    else:
###        fttn = None
###
###    return [ftn, mx, sx], [cls, my, sy], [fttn, pos]
###
###

##########################################################################################################
##########################################################################################################
##########################################################################################################
def balanced27gridpt_old(pm, ftlist, base, lim, ratio = 1, clsgrid = None, halopos = None):

    if (halopos is None) & (clsgrid is None):
        print('Need atleast a class-grid or halo positions')
    #halomass, halopos = self.halomass, self.halopos
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    grid = pm.mesh_coordinates(dtype='i4')
    
    #dind selects positions on 'base' grid above 'lim'
    #Idea is to select positions which are above some density threshold
    dind = np.where(base > lim)

    #Assign classes
    if clsgrid is None:
        zeros = np.zeros([nc, nc, nc])
        zeros[tuple((halopos/side).astype(int).T)] = 1
        cls = zeros[dind]
    else:
        cls = clsgrid[dind]
        cls[cls > 0] = 1
        cls[cls <=0] = 0

    #balance number of zeros here
    cls, dind = subsize_zeros(cls, np.array(dind).T, ratio = ratio)
    dind = tuple(dind.T)

    ndim = len(ftlist)
    print('Number of features is =',ndim)
    ftextlist = []
    #Extend dim of every grid and copy 0-side to N+1 to make things periodic
    for foo in range(ndim):
        ftextlist.append(periodic(ftlist[foo]))

    #Read features for training
    ft = np.zeros([dind[0].size, ndim*27])
    counter = 0
    for boo in range(ndim):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    rpos = [dind[0] + x, dind[1] + y, dind[2] + z]
                    ft[:, counter] = ftextlist[boo][tuple(rpos)]
                    counter +=1
    ftn, mx, sx = norm(ft)

    # Read features over the whole grid to generate data to test performance
    counter = 0
    ftt = np.zeros([nc**3, 27*ndim])
    for boo in range(ndim):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    rpos = [grid[:, 0] + int(x), grid[:, 1] + int(y), grid[:, 2] +int(z)]
                    ftt[:, counter] = ftextlist[boo][tuple(rpos)]
                    counter +=1
    fttn = norm(ftt, mx, sx)

    return [ftn, mx, sx], [cls], [fttn]


