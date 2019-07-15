import numpy as np
import matplotlib.pyplot as plt
from pmesh.particlemesh import ParticleMesh
from pmesh.pm import ParticleMesh as PMnew
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import yaml

import sys, os
#sys.path.append('/global/homes/c/chmodi/Programs/Py_codes/modules/')
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
#import mytools as tools
import mymass_function as mass_function
#import mycosmology
#import mysigmoid as sigmoid

sys.path.append("/global/homes/c/chmodi/Programs/cosmo4d/")
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/train_nets')
import features as ft
import datalib as lib
import datatools as dtools
import nettools as ntools
import sigtools as stools
import diagnostic as dg

scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
train = scratch + 'train/'
data = scratch + 'data/'

pfile = "/global/homes/c/chmodi/Data/ics_matterpow_0.dat"
klin, plin = np.loadtxt(pfile, unpack=True)
mf = mass_function.Mass_Func(pfile, 0.3175)

from cosmo4d import report as dgrep
import yaml

########


with open(scratch + 'train/models.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
train = scratch + 'train/'

abund=True
bs, nc, numd, seed = 400, 128, 1e-3, 100
doexp = False
mexp, cc = 0.5, 4.6
stellar = False
regression = False
Rsm = 3

num = int(numd*bs**3)
fine = 4
pm = PMnew(BoxSize=bs, Nmesh=[nc]*3)
#mbinsm = np.linspace(6.5, 11.5, 11)[::-1]
#
#if stellar:
#    mbinsm = 10**np.linspace(6, 11, 12)[::-1]
#    msave = [1e15] + list(mbinsm)
#
#else:
#    mbinsm = 10**np.linspace(7.5, 11.5, 12)[::-1]
#    msave = [1e15] + list(mbinsm)
#    if doexp:
#        mbinsm = mf.fmexp(mbinsm, mexp, cc)
#        msave = [1e15] + list(mbinsm)
#



ptpath = train + cfg['%s-%s'%(bs,nc)][0][numd]['ppath']
if regression:
    mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['rpath']
else:
    mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['mpath']
#if mexp is not None and mexp !=1:
#    mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['spath%02d'%(mexp*100)]
#elif stellar:
#    if regression:
#        mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['strpath']
#    else:
#        mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['stpath']
#else:
#    mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['mpath']
#

ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
if regression:
    mtup, mftname, mlocal, mdict = ntools.setupmassreg(mtpath )
else:
    mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
R1, R2 = pdict['R1'], pdict['R2']

#model
meshdict, dummy = ntools.readfiles(pm, scratch + '/data/L%04d_N%04d_S%04d_05step/'%(bs, nc, seed), R1=R1, R2=R2, abund=abund, 
                                   doexp=doexp, mexp=mexp, cc=cc, stellar=stellar)
ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
mftt = ntools.createdata(pm, meshdict, mftname, mlocal)
nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
if regression:
    nnmass = ntools.applymassreg(mftt, mtup).reshape(nc, nc, nc)
else:
    nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
if doexp:
    nnmass = mf.fmexp(ntools.relu(nnmass), mexp, cc)
predict = pm.create(mode ='real')
predict[...] = nnpred*nnmass
predictR = ft.smooth(predict, Rsm, 'fingauss')


#data

#hdictf = ntools.gridhalos(pm, scratch +'/data/L%04d_N%04d_S%04d_40step/'%(bs, 4*nc, seed), rank=num, R1=R1, R2=R2, pmesh=True)
#datapt = pm.create(mode='real', zeros=True)
#datapt[...] = hdictf[0]['halomesh']
hdictf = ntools.gridhalos(pm, scratch +'/data/L%04d_N%04d_S%04d_40step/'%(bs, fine*nc, seed), R1=R1, R2=R2, pmesh=True, abund=abund, 
                        doexp=doexp, mexp=mexp, cc=cc, stellar=stellar)[1]
datap = pm.paint(hdictf['position'][:num], hdictf['mass'][:num])
datapR = ft.smooth(datap, Rsm, 'fingauss')

#bins = np.linspace(-3, 3, 300)
func = dg.normal
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'k']

if doexp:
    mms =  [10**8, 10**8.5, 10**9, 10**9.2, 10**9.8, 10**10]

elif stellar:
    mms = [1e6, 1e7, 1e8, 1e9, 1e10]
else:
    mms = [1e8, 1e9, 1e10, 1e11, 1e12]

if doexp:
    mtpath = mtpath + 'Mexp/'
    if mexp is not None:
        mtpath = mtpath[:-1] + '%02d/'%(mexp*100)
    try:
        os.makedirs(mtpath)
    except Exception as e: 
        print ('The following exception occured \n', e, '\n')
###########################################
for M0 in mms:
    print('For M0 = %0.2e'%M0)

    if doexp:
        print(np.log10(M0)-0.5, np.log10(mf.fmexp(10**min(12.5, np.log10(M0) + 4), mexp, cc)))
        mbinsm = np.linspace(np.log10(M0)-0.5, np.log10(mf.fmexp(10**min(12.5, np.log10(M0) + 4), mexp, cc)), 12)[::-1]
    else:
        mbinsm = np.linspace(np.log10(M0)-0.5, min(13.5, np.log10(M0) + 4), 10)[::-1]
        #mbinsm = np.arange(np.log10(M0)-0.5, min(13.5, np.log10(M0) + 4), 0.3)[::-1]

    mbinsm = 10**mbinsm
    msave = [mbinsm[0]*100] + list(mbinsm)

    bins = np.zeros((mbinsm.size, 1000 ))
    for i in range(0, bins.shape[0]):
        bins[i] = np.linspace(-3*mbinsm[i]/M0, 2*mbinsm[i]/M0, bins.shape[1])

    print('mbinsm -- ', mbinsm)
    print(bins[:, 0], bins[:, -1])

    ####
    fig, ax = plt.subplots(3, 3, figsize = (14, 12))
    fit0 = dg.plot_mnoise(datapR.value, predictR.value, M0=M0, binfit=bins, c='k', axin=ax, func=func, mbin=mbinsm, retfit=True, lsf='--')[0]
    
    fits = []
    sgs = [0.1, 0.20]
    for i, sg in enumerate(sgs):
        hmass, hpos = dg.scatter_catalog(hdictf['mass'], hdictf['position'], sg)
        datasg = pm.paint(hpos[:num], hmass[:num])
        datasgR = ft.smooth(datasg, Rsm, 'fingauss')
        fitt = dg.plot_mnoise(datasgR.value, predictR.value, M0=M0, binfit=bins, c=colors[i], axin=ax, func=func, mbin=mbinsm, retfit=True)[0]
        fits.append(fitt)

    fig.savefig(mtpath + 'noisehist_M%02d_mass.png'%(10*np.log10(M0)))

    tosave = []
    for i, res in enumerate(fit0):
        #print(res)
        tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
    fpath = mtpath + 'hist_M%02d_mass.txt'%(10*np.log10(M0))
    np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), b, Mass function not matched for data')

    for j, sg in enumerate(sgs):
        tosave = []
        for i, res in enumerate(fits[j]):
            tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
        fpath = mtpath + 'hist_M%02d_sg%02d_mass.txt'%(10*np.log10(M0), sg*100)
        np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), b, Mass function not matched for data')
