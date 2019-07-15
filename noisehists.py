import sys
sys.path.append('/global/u1/c/chmodi/Programs/cosmo4d/notebooks/')
from lab import *

nsteps, seed = 5, 100

proj = '/project/projectdirs/m3058/chmodi/cosmo4d/'
outpath = scratch + 'output/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/version2/'
outpathold = scratch + 'output/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/'

bs, nc = 400, 128
numd = 1e-3

with open(modfile, 'r') as ymlfile: models = yaml.load(ymlfile)
ppath = proj + 'train/' + models['%s-%s'%(bs, nc)][zz][numd]['ppath']
mpath = ppath + models['%s-%s'%(bs, nc)][zz][numd]['mpath']

noisefile = ''

