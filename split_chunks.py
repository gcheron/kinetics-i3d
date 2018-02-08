# split chunk files to one file per time frame
# save the i-th chunk in FEATDIR/subRGB/{FEATNAME}_C_N_i (there are N chunks in the video and one chunk each C frames)
# split_chunks.py FEATDIR FEATNAME IMDIR
import numpy as np
#import subprocess
import glob
import os
import ipdb
import tstools.utils as tsutils

FEATDIR='/sequoia/data2/gcheron/UCF101/I3D'
IMDIR='/sequoia/data2/gcheron/UCF101/images'

FEATDIR='/sequoia/data2/gcheron/DALY/I3D'
IMDIR='/sequoia/data2/gcheron/DALY/images'

#FEAT = 'RGB'
FEAT = 'OPF'

Chunk_each=4 # one chunk is extracted each N frames
CHUNKEACH = 4

if FEAT == 'RGB':
   FEATNAME ='I3D_features_RGB'
   SAVESUBDIRNAME ='subRGB'
elif FEAT == 'OPF':
   FEATNAME ='I3D_features_OPF'
   SAVESUBDIRNAME ='subOPF'

FVAR = {'chunkeach': Chunk_each, 'opfpat': FEATNAME}

LOADFULLDIRNAME='full'
PREFIXSAVENAME='chunk'


vidcount=0
for vidname in os.listdir( '%s/%s' % (FEATDIR,LOADFULLDIRNAME) ):
   # get number of frames
   #cmd='ls %s/%s/*.jpg | wc -l' % (IMDIR,vidname)
   #nframes=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
   #nframes=int(nframes.stdout.read().rstrip())
   nframes = len(glob.glob('%s/%s/*.jpg' % (IMDIR,vidname) ))
   
   # get the feature
   fpath = '%s/%s/%s' % (FEATDIR,LOADFULLDIRNAME,vidname)
   feat = tsutils.merge_subvid_chunks(fpath, **FVAR)
   if feat.ndim == 4:
      feat = np.expand_dims(feat,0)
   bs,chunkNum,H,W,ch=feat.shape
   assert bs==1

   # check padding (leads to additional detections)
   # to match the track length, we would need to remove PADDING frame scores
   # nframes = Chunk_each*chunkNum - PADDING

   padding=Chunk_each*chunkNum-nframes
   assert padding<Chunk_each and padding>=0

   vidcount+=1
   print 'VID%d: %d -> %d (%f) -- %d' % (vidcount,nframes,chunkNum,float(nframes)/chunkNum,padding)

   savedir='%s/%s/%s' % (FEATDIR,SAVESUBDIRNAME,vidname)
   if not os.path.exists(savedir):
      os.makedirs(savedir)
   for t in range(chunkNum):
      # save each chunk in independent files
      savename='%s/%s_%d_%d_%d.npy' % (savedir,PREFIXSAVENAME,Chunk_each,chunkNum,t+1)
      sub=np.transpose(feat[0,t,:,:,:],(2,0,1)) # save in C x H x W (torch format)
      #print(savename,sub.shape)
      np.save(savename,sub)
