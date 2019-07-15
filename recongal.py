import warnings
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
#warnings.filterwarnings("ignore")
if rank!=0: warnings.filterwarnings("ignore")


import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)

#from cosmo4d.lab import maplhd as map
from cosmo4d import lab
from cosmo4d.lab import mapfof as fmap
from cosmo4d.lab import mapnoise #as noisemap
from cosmo4d.lab import objectives
from cosmo4d.lab import dg
from cosmo4d.lab import mymass_function as MF
from cosmo4d.iotools import save_map, load_map

#from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog
from solve import solve
import os, json, yaml, sys

scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
proj = '/project/projectdirs/m3035/chmodi/cosmo4d/'
#########################################
###
###CHECK FRESHRUN, INFOLDER

cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


#open up dictionaries
for i in cfg.keys(): locals()[i] = cfg[i]
for i in basep.keys(): locals()[i] = basep[i]
Mdown = [float(i) for i in Mdown]

#Mesh
truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
rank = truth_pm.comm.rank 
   
if mods['objective'].find('Smoothed') == 0: 
    if rank == 0 : print('Objective = %s'%mods['objective'][len('Smoothed'):])
    objfunc = getattr(objectives, mods['objective'])
    #objfunc = objectives.SmoothedObjective
    #likelihood = getattr(objfunc, mods['objective'][len('Smoothed'):])
else:
    print('Smoothed objective is needed')
    sys.exit()

map = getattr(lab, mods['map'])
#noisemod = getattr(noisemap, modes['noisemap'])

if basep['position'] : 
    if rank == 0: print('\nERROR : Position is not set up yet\n')
    sys.exit()
#########################################
#Folder and file names
ofolder = scratch + 'output/version2/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/'%(bs, nc, nsteps, seed, numd*1e4)
if mods['map'] == 'mapmass': ofolder += 'nngalaxies/'
elif mods['map'] == 'mapgal': ofolder += 'galaxies/'
else:
    if rank == 0: print('Map is wrong, should be mapgal or mapmass but is = ', mods['map'])
    sys.exit()

if mods['objective'] == 'SmoothedOvdObjective': ofolder += 'overdensity/'
elif mods['objective'] == 'SmoothedLogOvdObjective': ofolder += 'logoverdensity/'
elif mods['objective'] == 'SmoothedLogTotObjective': ofolder += 'logtotal/'

dpath = proj + 'data/z%02d/L%04d_N%04d_S%04d_%02dstep/'%(zz, bs, nc*fine, seed, 40)

#This needs to be updated!
if zz:
    ofolder = ofolder + 'z%02d/'%(zz*10)
    hofolder = hofolder[:-1] + '_z%02d/'%(zz*10)
    if not rank: print(hofolder)

#Read in model from somwhere
modfile = modelp['cfgfilepath']

with open(modfile, 'r') as ymlfile: models = yaml.load(ymlfile)
ppath = proj + 'train/' + models['%s-%s'%(bs, nc)][zz][numd]['ppath']
if mods['map'] == 'mapmass': mpath = ppath + models['%s-%s'%(bs, nc)][zz][numd]['gpath']
elif mods['map'] == 'mapgal': mpath = ppath + models['%s-%s'%(bs, nc)][zz][numd]['mpath']
if not rank: print('\nNN paths are \nppath = %s\nmpath = %s\n'%(ppath, mpath))

with open(ppath + '/pinfo.json') as fp: pdict = json.load(fp)
with open(mpath + '/minfo.json') as fp: mdict = json.load(fp)

#
#Modify file name here
prefix = foldername
if noisep['smooth'] is not None: prefix = prefix + 'sm%d'%noisep['smooth']
if noisep['noisevar']: prefix = prefix + 'v%02d'%(noisep['noisevar']*100)
if noisep['noised'] == 3: prefix = prefix + 'n3d'
if noisep['offset']: prefix = prefix +'off'
if noisep['L1'] : prefix = prefix + 'L1'
#if noisep['mlim'] : prefix = prefix + 'mlim' + 'n%02d'%numpy.log10(noisep['ninf'])
if noisep['mlim'] : prefix = prefix + 'mlim' + 'n%02d'%(100*numpy.log10(noisep['ninf']))
#if noisep['mlim'] : prefix = prefix + 'mlim'
if noisep['ninf'] == 'auto': prefix = prefix + 'nauto'
elif noisep['ninf'] is not None: prefix = prefix + 'n%02d'%(100*numpy.log10(noisep['ninf']))
if scatterp['sigma'] is not None: prefix = prefix + '-sg%d'%(100*scatterp['sigma'])
if optp['maxwidthonly']: prefix = prefix + '-now'
if optp['nomultismooth']: prefix = prefix + '-nomul'

##
fname = 's%d_%s'%(initp['seed'], prefix)
optfolder = ofolder + 'opt_%s/'%fname

if rank == 0: print('\nOutput Folder is %s\n'%optfolder)

for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except: pass

#logging
if not freshrunp['isit']:
    for i in freshrunp.keys(): locals()[i] = freshrunp[i]

if not rank:
    if not os.path.isfile(optfolder + 'config.yml'):
        with open(optfolder + 'config.yml', 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
    if freshrunp['isit'] or not os.path.isfile(optfolder + 'log.txt'):
        with open(optfolder + 'log.txt', 'w') as outfile:
            outfile.write(note)
            outfile.write('\n')
            outfile.write('\n Postion Network \n')
            outfile.write('Built from %s\n'%ppath)
            for key in pdict.keys(): outfile.write('%s - %s\n'%(key, pdict[key]))
            outfile.write('\n Mass Network \n')
            outfile.write('Built from %s\n'%mpath)
            for key in mdict.keys(): outfile.write('%s - %s\n'%(key, mdict[key]))
    else:
        ff = open(optfolder + 'log.txt', 'a')    
        ff.write('\Continuing a previous run from M0 = %0.1f, Ns = %0.2f, W = %0.2f! \n'%(Ms, N0, W0))
        ff.write(note)
        ff.write('\n')
        ff.close()
        

#########################################

#setup MODEL
pfile = 'ics_matterpow_0.dat'
klin, plin = numpy.loadtxt(pfile, unpack = True)
pk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
aa = 1/(1 + basep['zz'])
if not rank: print('Redshift and Scale factor at final point are = %0.2f, %0.2f'%(basep['zz'], aa))
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
dynamic_model = NBodyModel(cosmo, truth_pm, B=2, steps=stages)
mf = MF.Mass_Func(pfile, cosmo.Om0)
mock_model = map.MockModel(dynamic_model, ppath = ppath, mpath = mpath)



#########################################
#setup DATA
#depends if its a freshrun or are we continuing from a previous run
if freshrunp['isit']:
    if not rank :print('\nDoing a fresh run! \n')

    s_truth = truth_pm.generate_whitenoise(seed, mode='complex')\
        .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*itf

    galcat = BigFileCatalog(dpath + '/galaxies_n%02d/galcat'%(numd*1e4), header='Header')

    fof_model = fmap.GalModel(dynamic_model, galcat, mf = mf, ofolder=optfolder,  zz=zz,  rsd=rsd)
    data_p = fof_model.make_observable(s_truth)
    fof_model_mass = fmap.GalModel(dynamic_model, galcat, mf = mf, ofolder=optfolder,  zz=zz,  rsd=rsd)
    data_m = fof_model_mass.make_observable(s_truth)

    #Noise and perf
    fit_p = mock_model.make_observable(s_truth)

    #Save
    data_p.save(optfolder+'datap/')
    fit_p.save(optfolder+'fitp/')

    if initp['infolder'] is not None:
        if truth_pm.comm.rank == 0:
            print('\nStaring from the folder \n %s \n'%initp['infolder'])
            with open(optfolder + 'log.txt', 'a') as outfile:
                outfile.write('\nStaring from the folder \n %s \n'%initp['infolder'])

        s_init = BigFileMesh(initp['infolder'], 's').paint()
    else:
        s_init = initp['power']* truth_pm.generate_whitenoise(initp['seed'], mode='complex')\
                              .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
                              .c2r()

    Nmesh = nc

else:
    #Load
    #
    if not rank: print('\Continuing a previous run from Ns = %0.2f, W = %0.2f! \n'%(N0, W0))

    if sfolder is not None:
        if truth_pm.comm.rank == 0:
            print('\nStaring from the folder \n %s \n'%sfolder)
            with open(optfolder + 'log.txt', 'a') as outfile:
                outfile.write('\nStaring from the folder \n %s \n'%sfolder)
        s_init = BigFileMesh(sfolder , 's').paint()
        data_p = map.Observable.load(dfolder + '/datap')
        fit_p = map.Observable.load(dfolder + '/fitp')
        data_p.save(optfolder+'datap/')
        fit_p.save(optfolder+'fitp/')
    else:
        if bestfit: sinitfolder = 'best-fit'
        else: sinitfolder = '/%04d/fit_p'%nit
        if not rank: print('Start from folder iteration = %s'%sinitfolder)
        try: s_init = BigFileMesh(optfolder + 'M0_%02d/%d-%02d-%0.2f/%s'%(Ms, Nmesh, W0, N0, sinitfolder), 's').paint()
        except: s_init = BigFileMesh(optfolder + 'M0_%02d/%d-%02d-%0.2f/%s'%(Ms*10, Nmesh, W0, N0, sinitfolder), 's').paint()
        data_p = map.Observable.load(optfolder + 'datap')
        fit_p = map.Observable.load(optfolder + 'fitp')
        if basep['position']: data_m = map.Observable.load(optfolder + 'datam')
        else: data_m = data_p


if truth_pm.comm.rank == 0:
    print('data_p, data_n created')
    print('max and min of data is %0.2e, %0.2e'%(data_p.mapp[...].max(), data_p.mapp[...].min()))


#########################################
#NOISE
truth_noise_model = mapnoise.NoiseModel(truth_pm, None, noisep['noisevar']*(truth_pm.BoxSize/truth_pm.Nmesh).prod(), 1234)
data_n = truth_noise_model.add_noise(data_p)
if freshrunp['isit']: data_n.save(optfolder+'datan/')

#########################################

##
##
#SETUP OPTIMIZER  
x0 = s_init
Nmesh = Nmesh
C = x0.BoxSize[0] / x0.Nmesh[0]

maxwidth = pdict['width']
if optp['maxwidthonly']: widths = [maxwidth]
else:  widths = [1, 2, 3, 5]

if optp['nomultismooth']: Nss = [0.0]
else: Nss = optp['Nss']

Nsub = optp['Nsub']
tol = float(optp['tol'])
saveit = int(saveit)


def modifynoise(M):
    noise_model = truth_noise_model
    noisemlim = 10**M
    #if noisep['noisefile'] is not None:                        
    if noisep['offset'] or noisep['noised']==3:
        try: 
            noisefile = mpath + noisep['noisefile']%(10*M)
        except TypeError : noisefile = mpath + noisep['noisefile']
        except: 
            if not rank: print('\nERROR: Need a noise file for the offset & 3D noise\n')
            sys.exit()
        if not rank: print('Noise file is \n%s'%noisefile)

        truth_noise_model.create_off3d(data_m.mapp, noisefile, smooth=noisep['smooth'])
        save_map(truth_noise_model.offset, optfolder + 'noise_M%02d'%(10*M), 'offset')
        truth_noise_model.create_ivar3d(data_m.mapp, noisefile, noisep['noisevar'], smooth=noisep['smooth'])    

    if noisep['mlim']: truth_noise_model.suppress_noise(data_m.mapp, mlim=noisemlim, ninf=float(noisep['ninf']), smooth=noisep['smooth'])
    if noisep['noised'] == 3: save_map(truth_noise_model.ivar3d, optfolder + 'noise_M%02d'%(10*M), 'ivar3d')
    return noise_model




##Loop it !
if freshrunp['isit']:
    for M in Mdown:
        if not rank: print(M)
        if M < Mdown[0]:
            widths = [maxwidth]
            Nss = Nss[-Nsub:]
        for Ns in Nss:
            for width in widths:
                tol = tol#max(0.001, 0.005*Ns/width)
                if width <= maxwidth:
                    if truth_pm.comm.rank == 0:
                        print('\n Do for M0 = %0.1f, width = %0.2f and cell smoothing of %0.2f \n\n'%(M, width, Ns))

                    
                    mock_model_width = map.MockModel(dynamic_model, ppath = ppath, mpath = mpath, pwidth = width)
                    noise_model = modifynoise(M)
                    sml = truth_pm.BoxSize[0] / Nmesh * Ns
                    obj = objfunc(mock_model_width, noise_model, data_p, prior_ps=pk, sml=sml, \
                                                M0=10**M, smooth=noisep['smooth'], noised=noisep['noised'], L1=noisep['L1'], offset=noisep['offset'])

                    fprefix = 'Me%dsm%d'%(M, noisep['smooth'])
                    title = prefix
                    x0 = solve(Nmesh, x0, rtol=tol, run='M0_%02d/%d-%02d-%0.2f'%(M*10, Nmesh, width, Ns), Nsm=Ns, prefix =fprefix, \
                               mock_model=mock_model_width, obj=obj, data_p=data_p, truth_pm=truth_pm, optfolder=optfolder, saveit=saveit, title=title)
        

else:
    
    for M in Mdown:
        if M > Ms:
            continue
        if M < Mdown[0]:
            widths = [maxwidth]
            Nss = Nss[-Nsub:]
        for Ns in Nss:
            for width in widths:
                if Ns > N0: continue
                elif Ns == N0 and width < W0: continue
                elif Ns == N0 and width == W0 and bestfit:
                    N0 = 10
                    W0 = 0
                    continue

                N0 = 10
                W0 = 0
                tol = tol

                if width <= maxwidth:
                    if truth_pm.comm.rank == 0:
                        print('\n Do for M0 = %0.1f, width = %0.2f and cell smoothing of %0.2f \n\n'%(M, width, Ns))

                    mock_model_width = map.MockModel(dynamic_model, ppath = ppath, mpath = mpath, pwidth = width)
                    noise_model = modifynoise(M)
                    sml = truth_pm.BoxSize[0] / Nmesh * Ns
                    obj = objfunc(mock_model, noise_model, data_p, prior_ps=pk, sml=sml, \
                                                M0=10**M, smooth=noisep['smooth'], noised=noisep['noised'], L1=noisep['L1'], offset=noisep['offset'])

                    fprefix = 'Me%dsm%d'%(M, noisep['smooth'])
                    title = prefix
                    if not bestfit and nit != 0:
                        x0 = solve(Nmesh, x0, rtol=tol, run='M0_%02d/%d-%02d-%0.2f-init%d'%(M*10, Nmesh, width, Ns, nit), Nsm=Ns, prefix =fprefix, \
                                   mock_model=mock_model_width, obj=obj, data_p=data_p, truth_pm=truth_pm, optfolder=optfolder, saveit=saveit)
                        nit = 0 
                    else:
                        x0 = solve(Nmesh, x0, rtol=tol, run='M0_%02d/%d-%02d-%0.2f'%(M*10, Nmesh, width, Ns), Nsm=Ns, prefix =fprefix, \
                                   mock_model=mock_model_width, obj=obj, data_p=data_p, truth_pm=truth_pm, optfolder=optfolder, saveit=saveit, title=title)




