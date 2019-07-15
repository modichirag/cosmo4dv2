##Generate noise files with a scatter drawn from a uniform distribution
##
import numpy as np
import matplotlib.pyplot as plt
from pmesh.particlemesh import ParticleMesh
from pmesh.pm import ParticleMesh as PMnew
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import yaml

import sys
sys.path.append('/global/homes/c/chmodi/Programs/Py_codes/modules/')
#import mytools as tools
import mymass_function as mass_function
import mycosmology
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

abund=False
bs, nc, numd, seed = 200, 64, 5e-4, 100
num = int(numd*bs**3)
fine = 2
bpath = scratch + 'output/L%04d_N%04d_05step-fof/lhd_S%04d/n%02d'%(bs, nc, seed, numd*1e4)
pm = PMnew(BoxSize=bs, Nmesh=[nc]*3)
#mbinsm = np.linspace(6.5, 11.5, 11)[::-1]
mbinsm = np.linspace(7.5, 11.5, 12)[::-1]
print(mbinsm)
mbinsm = 10**mbinsm
msave = [1e15] + list(mbinsm)


ptpath = train + cfg['%s-%s'%(bs,nc)][0][numd]['ppath']
mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['mpath']
#ptpath = train + 'L0400_N0128_05step-n10/width_5/Wts_50_1/r3rf1/hlim-20_clim-20_nreg-40_batch-100/'
#mtpath = ptpath + '/eluWts-20_1/blim-13_nreg-33_batch-1000/'

ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
R1, R2 = pdict['R1'], pdict['R2']

#model
meshdict, dummy = ntools.readfiles(pm, scratch + '/data/L%04d_N%04d_S%04d_05step/'%(bs, nc, seed), R1=R1, R2=R2, abund=abund)
ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
mftt = ntools.createdata(pm, meshdict, mftname, mlocal)
nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
predict = pm.create(mode ='real')
predict[...] = nnpred*nnmass
predictR = ft.smooth(predict, 3, 'fingauss')


#data

#hdictf = ntools.gridhalos(pm, scratch +'/data/L%04d_N%04d_S%04d_40step/'%(bs, 4*nc, seed), rank=num, R1=R1, R2=R2, pmesh=True)
#datapt = pm.create(mode='real', zeros=True)
#datapt[...] = hdictf[0]['halomesh']
hdictf = ntools.gridhalos(pm, scratch +'/data/L%04d_N%04d_S%04d_40step/'%(bs, fine*nc, seed), R1=R1, R2=R2, pmesh=True, abund=abund)[1]
datap = pm.paint(hdictf['position'][:num], hdictf['mass'][:num])
datapR = ft.smooth(datap, 3, 'fingauss')

bins = np.linspace(-4, 4, 200)
func = dg.normal
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'c', 'k']


def scatter_cat(hmass, hpos, seed=100, smin=0.1, smax=0.2):
    logl = np.log10(hmass)
    rng = np.random.RandomState(seed)
    t = rng.normal(scale=rng.uniform(smin, smax, size=len(logl)))
    logl = logl + t
    halomass2 = 10**logl
    sort2 = np.argsort(halomass2)[::-1]
    halomass2 = halomass2[sort2]
    halopos2 = hpos[sort2]
    return halomass2, halopos2

smin, smax = 0.1, 0.2
for M0 in [1e8, 1e9, 1e10, 1e11, 1e12]:
    fig, ax = plt.subplots(3, 3, figsize = (14, 12))
    fit0 = dg.plot_noise(datapR.value, predictR.value, M0=M0, binfit=bins, c='k', axin=ax, func=func, mbin=mbinsm, retfit=True, lsf='--')[0]
    
    fits = []
    sgs = [0.2]
    for i, sg in enumerate(sgs):

        hmass, hpos = dg.scatter_catalog(hdictf['mass'], hdictf['position'], sigma=sg)
        datasg = pm.paint(hpos[:num], hmass[:num])
        datasgR = ft.smooth(datasg, 3, 'fingauss')
        fitt = dg.plot_noise(datasgR.value, predictR.value, M0=M0, binfit=bins, c='k', axin=ax, func=func, mbin=mbinsm, retfit=True)[0]
        fits.append(fitt)
    

    for i in range(3):
        hmass, hpos = scatter_cat(hdictf['mass'], hdictf['position'], seed=100*i + 123, smin=smin, smax=smax)
        datasg = pm.paint(hpos[:num], hmass[:num])
        datasgR = ft.smooth(datasg, 3, 'fingauss')
        fitt = dg.plot_noise(datasgR.value, predictR.value, M0=M0, binfit=bins, c=colors[i+4], axin=ax, func=func, mbin=mbinsm, retfit=True)[0]

    fig.savefig(mtpath + 'noisehist_M0%d_na_ran%02d%02d.png'%(np.log10(M0), 100*smin, 100*smax))

    tosave = []
    for i, res in enumerate(fitt):
        print(res)
        tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
    fpath = mtpath + 'hist_M0%d_na_ran%02d%02d.txt'%(np.log10(M0), 100*smin, 100*smax)
    np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), sigma\nMass function not matched for data\nuniform scatter between %0.2f and %0.2f'%(smin, smax))

#
#    for j, sg in enumerate(sgs):
#        tosave = []
#        for i, res in enumerate(fits[j]):
#            tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
#        fpath = mtpath + 'hist_M0%d_sg%02d_na.txt'%(np.log10(M0), sg*100)
#        np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), b, Mass functionnot matched for data')
