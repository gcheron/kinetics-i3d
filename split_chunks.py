# split chunk files to one file per time frame
# save the i-th chunk in FEATDIR/subRGB/{FEATNAME}_C_N_i (there are N chunks in the video and one chunk each C frames)
# split_chunks.py FEATDIR FEATNAME IMDIR
import numpy as np
import subprocess
import os

FEATDIR='/sequoia/data2/gcheron/UCF101/I3D'
IMDIR='/sequoia/data2/gcheron/UCF101/images'
FEATNAME='I3D_features_RGB.npy'
SAVESUBDIRNAME='subRGB'
LOADFULLDIRNAME='full'
PREFIXSAVENAME='chunk'

Chunk_each=4 # one chunk is extracted each N frames

vidcount=0
for vidname in os.listdir( '%s/%s' % (FEATDIR,LOADFULLDIRNAME) ):
   # get number of frames
   cmd='ls %s/%s/*.jpg | wc -l' % (IMDIR,vidname)
   nframes=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
   nframes=int(nframes.stdout.read().rstrip())
   
   # get the feature
   feat=np.load('%s/%s/%s/%s' %(FEATDIR,LOADFULLDIRNAME,vidname,FEATNAME))
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
      sub=feat[0,t,:,:,:]
      #print(savename,sub.shape)
      np.save(savename,sub)
