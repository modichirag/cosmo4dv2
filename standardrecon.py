import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)

#from cosmo4d.lab import maplhd as map
from cosmo4d.lab import standardrecon as model
from cosmo4d.lab import mapfof as fmap
from cosmo4d.lab import dg
from cosmo4d.lab import mymass_function as MF
from cosmo4d.iotools import save_map, load_map


from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import ArrayCatalog
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog
from solve import solve
import os, json, yaml, sys


scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
project = '/project/projectdirs/m3058/chmodi/cosmo4d/'
#########################################
###
###CHECK FRESHRUN, INFOLDER

cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


#open up dictionaries
for i in cfg.keys(): locals()[i] = cfg[i]
for i in basep.keys(): locals()[i] = basep[i]
#Mesh
pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
rank = truth_pm.comm.rank 
   

#########################################
#Folder and file names
ofolder = scratch + 'output/version2/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/'%(bs, nc, nsteps, seed, numd*1e4)
#hofolder = scratch + 'output/L%04d_N%04d_%02dstep-fof/'%(bs, nc, nsteps) + foldersp['hofolder']%(fine, seed)
hofolder = project + 'data/L%04d_N%04d_S%04d_%02dstep/'%(bs, int(nc*fine), seed, nfsteps)

#This needs to be updated!
if zz:
    ofolder = ofolder + 'z%02d/'%(zz*10)
    hofolder = hofolder[:-1] + '_z%02d/'%(zz*10)
    if not rank: print(hofolder)

if scatterp['sigma'] is not None: 
    sigma = scatterp['sigma']
    foldername = foldername + 'sg%d/'%(100*sigma)


optfolder = ofolder + '/standard/'
optfolder = optfolder + foldername + 'R%d/'%Rsm
if masswt: optfolder = optfolder[:-1] + '-mass/'
if rsd: optfolder = optfolder[:-1] + '-rsd/'

try: os.makedirs(optfolder)
except: pass
if rank == 0 : print(optfolder)

#logging
if not os.path.isfile(optfolder + 'config.yml'):
    with open(optfolder + 'config.yml', 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
        

#########################################

pfile = 'ics_matterpow_0.dat'
klin, plin = numpy.loadtxt(pfile, unpack = True)
pk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
aa = 1/(1 + basep['zz'])
mf = MF.Mass_Func(pfile, cosmo.Om0)
if not rank: print('Redshift and Scale factor at final point are = %0.2f, %0.2f'%(basep['zz'], aa))
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
dynamic_model = NBodyModel(cosmo, truth_pm, B=2, steps=stages)

s_truth = truth_pm.generate_whitenoise(seed, mode='complex')\
        .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()

fofcat = fmap.make_finer(bs, fine*nc, seed, nsteps=nfsteps, cosmo=cosmo, pk=pk, ofolder=hofolder, zz=zz)
fof_model = fmap.FoFModel(dynamic_model, fofcat, numd, mf = mf, sigma=scatterp['sigma'], abund=abund, ofolder=optfolder, 
                          seed=scatterp['rseed'], zz=zz, mexp=None, cc=0, stellar=False, rsd=rsd, truemass=False)
data_m = fof_model.make_observable(s_truth)
data_m.save(optfolder+'datam/')

fofdata = BigFileCatalog(optfolder + 'FOFd', header='Header')
layout = pm.decompose(fofdata['PeakPosition'])
hpmesh = pm.paint(fofdata['PeakPosition'], layout=layout)
save_map(data_m.s, optfolder + 'datap', 's')
save_map(data_m.d, optfolder + 'datap', 'd')
save_map(hpmesh, optfolder + 'datap', 'mapp')

#setup MODEL
pfile = 'ics_matterpow_0.dat'
klin, plin = numpy.loadtxt(pfile, unpack = True)
pk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
aa = 1/(1 + basep['zz'])
if not rank: print('Redshift and Scale factor at final point are = %0.2f, %0.2f'%(basep['zz'], aa))


recon, hpshift, rshift = model.standard(pm=pm, fofcat=fofdata, datap=data_m, mf=mf, kb=kbias, Rsm=Rsm, rsd=rsd, zz=zz, mass=masswt)

save_map(recon, optfolder + 'best-fit', 's')
save_map(hpshift, optfolder + 'best-fit', 'mapp')
save_map(rshift, optfolder + 'best-fit', 'd')

#########################################

