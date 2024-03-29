import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)

from cosmo4d.lab import mapfinal as map
from cosmo4d.lab import dg
from cosmo4d.lab import objectives

from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog
import os, json
from solve import solve

objfunc = objectives.SmoothedObjective
#########################################

#Set parameters here
bs, nc = 400., 128
seed = 100
truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
comm = truth_pm.comm
rank = comm.rank
nsteps = 5
zz = 0
aa = 1/(1+zz)
B = 1
iseed = 999
smoothings = [0, 1] 

noisevar = 0.01
smooth = None


#
mainfolder = '/global/cscratch1/sd/chmodi/cosmo4d/'
dfolder = mainfolder + '/data/z%02d/L%04d_N%04d_S%04d_%dstep/'%(zz, bs, nc, seed, 40)

ofolder = mainfolder + '/v2/z%02d/L%04d_N%04d_S%04d/pm/'%(zz, bs, nc, seed)

prefix = 'test-sm2'
fname = 's%03d_%s'%(iseed, prefix)
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0:
    print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass


#initiate

klin, plin = numpy.loadtxt('ics_matterpow_0.dat', unpack = True)
pk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)

#########################################
#dynamics
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
dynamic_model = NBodyModel(cosmo, truth_pm, B=B, steps=stages)

#noise
#Artifically low noise since the data is constructed from the model
truth_noise_model = map.NoiseModel(truth_pm, None, noisevar*(truth_pm.BoxSize/truth_pm.Nmesh).prod(), 1234)
mock_model = map.MockModel(dynamic_model)

#Create and save data if not found

dyn = BigFileCatalog(dfolder + 'dynamic/1')
s_truth = BigFileMesh(dfolder + '/mesh/', 's').paint()
data_p = mock_model.make_observable(s_truth)
layout = truth_pm.decompose(dyn['Position'])
data_p.mapp = truth_pm.paint(dyn['Position'], layout=layout)
data_p.save(optfolder+'datap/')


data_n = truth_noise_model.add_noise(data_p)
data_n.save(optfolder+'datan/')

fit_p = mock_model.make_observable(s_truth)
fit_p.save(optfolder+'fitp/')

##



################################################
#Optimizer  
##

#Initialize
inpath = None
for ir, r in enumerate(smoothings):
    if os.path.isdir(optfolder + '/%d-%0.2f/best-fit'%(nc, r)): 
        inpath = optfolder + '/%d-%0.2f//best-fit'%(nc, r)
        sms = smoothings[:ir][::-1]
        lit = 100
        if r == 0:
            if rank == 0:print('\nAll done here already\nExiting\n')
            sys.exit()
    else:
        for iiter in range(100, -1, -20):
            path = optfolder + '/%d-%0.2f//%04d/fit_p/'%(nc, r, iiter)
            if os.path.isdir(path): 
                inpath = path
                sms = smoothings[:ir+1][::-1]
                lit = 100 - iiter
                break
    if inpath is not None:
        break

if inpath is not None:
    if rank == 0: print(inpath)
    s_init = BigFileMesh(inpath, 's').paint()
else:
    s_init = truth_pm.generate_whitenoise(iseed, mode='complex')\
        .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*0.001
    sms = smoothings[::-1]
    lit = 100


x0 = s_init
N0 = nc
C = x0.BoxSize[0] / x0.Nmesh[0]

for Ns in sms:
    if truth_pm.comm.rank == 0: print('\nDo for cell smoothing of %0.2f\n'%(Ns))
    #x0 = solve(N0, x0, 0.005, '%d-%0.2f'%(N0, Ns), Ns)
    sml = C * Ns
    rtol = 0.01
    maxiter = 100
    run = '%d-%0.2f'%(N0, Ns)
    if Ns == sms[0]:
        if inpath is not None:
            run += '-nit_%d-sm_%.2f'%(iiter, smoothings[ir])
            maxiter = lit
    if maxiter > 0:
        obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=pk, sml=sml)
        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, truth_pm, optfolder, \
                   saveit=20, showit=5, title=None, maxiter=maxiter)    
