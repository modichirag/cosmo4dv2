#Make noise histograms for difference in overdensity

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

#scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
scratch = '/project/projectdirs/astro250/chmodi/cosmo4d/'
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


bs, nc, numd, seed = 400, 128, 1e-3, 100
zz = 0.0

abund = False
Rsm = 3


num = int(numd*bs**3)
sfine = 4
pm = PMnew(BoxSize=bs, Nmesh=[nc]*3)


ptpath = train + cfg['%s-%s'%(bs,nc)][zz][numd]['ppath']
mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][zz][numd]['mpath']
print('Loaded and saved in \n%s'%mtpath)


ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
R1 = pdict['R1']
try : R2 = pdict['R2']
except :
    sfac = pdict['sfac']
    R2 = R1*sfac


#model
print('Reading Files')
meshdict, dummy = ntools.readfiles(pm, scratch + '/data/z%02d/L%04d_N%04d_S%04d_05step/'%(zz*10, bs, nc, seed), R1=R1, R2=R2, abund=abund)

ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
mftt = ntools.createdata(pm, meshdict, mftname, mlocal)
nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
predict = pm.create(mode ='real', value=nnpred*nnmass)
predictR = ft.smooth(predict, Rsm, 'fingauss')

#data

print('Generating data')

hdictf = ntools.gridhalos(pm, dpath=scratch +'/data/z%02d/L%04d_N%04d_S%04d_40step/'%(zz*10, bs, sfine*nc, seed), R1=R1, R2=R2, pmesh=False, abund=abund)[1]
datap = pm.paint(hdictf['position'][:num], mass = hdictf['mass'][:num])
datapR = ft.smooth(datap, Rsm, 'fingauss')

print('Data generated')

func = dg.normal
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'k']


###########################################

mbins = np.logspace(10, 13, 16)[::-1]
msave = [mbins[0]*100] + list(mbins)
print('mbins -- ', mbins)

####

mms = [1e10, 1e11, 1e12]
for M0 in mms:
    print('For M0 = %0.2e'%M0)
    #normp, normd = predictR.cmean(), datapR.cmean()
    #scatter = dg.gridscatter(np.log((datapR[...]+M0)/(normd)), np.log((predictR[...]+M0)/(normp)), mbins, datapR[...])
    normd, normp = np.log((datapR[...]+M0)).mean(), np.log((predictR[...]+M0)).mean()
    scatter = dg.gridscatter(np.log((datapR[...]+M0))/normd, np.log((predictR[...]+M0))/normp, mbins, datapR[...])

    tosave = []
    #bins = np.linspace(-0.5, 0.7)
    fig, ax = plt.subplots(4, 4, figsize = (15, 15))
    for i in range(16):
        mean, std = (scatter[i][1] - scatter[i][0]).mean(), (scatter[i][1] - scatter[i][0]).std()
        bins = np.linspace(mean - 3*std, mean + 3*std)
        #bins = np.linspace(-10/(i+1), 10/(i+1))
        axis = ax.flatten()[i]
        axis.hist(scatter[i][1] - scatter[i][0], histtype='step', bins=bins, normed=True, color='C%d'%(i%9), label='%0.2e'%mbins[i])
        xx, yy, res = dg.fitpdf(scatter[i][1] - scatter[i][0], dg.normal, bins=bins, normed=True, x0=[10, mean, std])
        axis.plot(xx, dg.normal(xx, *res.x), 'k--', label='%0.2f, %0.2f'%(res.x[1], res.x[2]))
        tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
        axis.legend()

    ff = mtpath + 'noisehist_ovdlog_M%02d_na.png'%(10*np.log10(M0))
    if Rsm !=3 : ff = ff[:-4] + '_R%d.png'%Rsm
    fig.savefig(ff)

    fpath = mtpath + 'hist_ovdlog_M%02d_na.txt'%(10*np.log10(M0))
    if Rsm !=3 : fpath = fpath[:-4] + 'R%d.txt'%Rsm
    np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), \sigma, Mass function not matched for data')


    

