import numpy as np
from skimage.transform import resize
import ipdb
import os
import glob
import re
import numpy as np

def padding(X, new_h, new_w, getValues=False):
   if type(X) == tuple:
      # when we just need theoretical values
      # we can pass (h, w) instead of X tensor
      assert getValues == True
      h, w = X
   else:
      h, w = X.shape[:-1]

   new_ar = float(new_w)/new_h
   ar = float(w)/h

   if not getValues:
      new_X = np.zeros((new_h, new_w, 3),dtype='float32')
   if new_ar > ar: # need to add more W
      # scale by setting H to the wanted size
      sc = float(new_h)/h
      _w = int(round(sc * w))
      pad_w = new_w - _w
      pad_h = 0
      if not getValues:
         X = resize(X, (new_h, _w))
         st_pad = int(round(float(pad_w)/2))
         en_pad = st_pad + _w
         new_X[:,st_pad:en_pad,:] = X

   else: # need to add more H
      # scale by setting W to the wanted size
      sc = float(new_w)/w
      _h = int(round(sc * h))
      pad_w = 0
      pad_h = new_h - _h
      if not getValues:
         X = resize(X, (_h, new_w))
         st_pad = int(round(float(pad_h)/2))
         en_pad = st_pad + _h
         new_X[st_pad:en_pad,:,:] = X

   # check farest
   w_f = float(max(w ,new_w))/min(w, new_w)
   h_f = float(max(h, new_h))/min(h, new_h)

   assert (pad_h >= 0) and (pad_w >= 0)

   if getValues:
      return sc, pad_h, pad_w
   else:
      return new_X

def merge_subvid_chunks(vidpath, rgbpat = None, opfpat = None, chunkeach=-1):
   # return the merged RGB and OPF given the video dir path as input
   # if chunkeach > 0, we are dealing with chunks extracted each "chunkeach" frames

   assert not ( (rgbpat is None) and (opfpat is None) )
   paths = {}
   pat = {}
   n_out = 0
   if not rgbpat is None:
      paths['rgb'] = []
      pat['rgb'] = rgbpat
      checkpat = rgbpat
      n_out += 1
   if not opfpat is None:
      paths['opf'] = []
      pat['opf'] = opfpat
      checkpat = opfpat
      n_out += 1

   outputs = {'rgb': None, 'opf': None }
   frame_st_end = [] # info about sub chunks start and end

   if not os.path.exists('%s/%s.npy' % (vidpath, checkpat)):
      # get the number of sub videos
      checkname = glob.glob('%s/%s_sub*' % (vidpath, checkpat))[0]
      subnum = int(re.match('.*/%s_sub[0-9]*-([0-9]*)[^/]*.npy' % checkpat, checkname).group(1))
      feat_ov = -1 # overlap between two sub chunks

      for sv in range(subnum): # load all sub vid in the right orde
         for feat in paths:
            sub_ = glob.glob('%s/%s_sub%d-%d*' % (vidpath, pat[feat], sv+1, subnum))
            assert len(sub_) == 1
            sub_ = sub_[0]
            paths[feat].append(sub_)
            rres = re.match('.*/[^/]*_sub[0-9]*-[0-9]*-([0-9]*)-([0-9]*).npy',sub_)
            st_ = int(rres.group(1))
            en_ = int(rres.group(2))
            if len(frame_st_end) >= sv+1: # add only for the first feat
               assert frame_st_end[sv] == (st_,en_)
            else:
               frame_st_end.append((st_,en_))
               if sv > 0:
                  cov = frame_st_end[-2][1] - frame_st_end[-1][0] + 1
                  if sv == 1:
                     feat_ov = cov
                     assert feat_ov % 2 == 0
                  else:
                     assert feat_ov == cov, 'all sub chunks must have the same ov'

      assert frame_st_end[0][0] == 1, 'frame number must start at 1!'
      nbframes = frame_st_end[-1][1]
      
      min_h = feat_ov/2 # minimum wanted frame history in the sub chunk to get "accurate" value
      for feat in paths:
         for sv in range(subnum):
            cchunk = np.load(paths[feat][sv])
            if cchunk.shape[0] == 1:
               cchunk = cchunk.squeeze(0)

            st_, en_ = frame_st_end[sv]
            vlen = en_ - st_ + 1

            cshape0 = cchunk.shape[0]

            if chunkeach > 0:
               assert cshape0 == vlen/chunkeach or cshape0 == vlen/chunkeach + 1
            else:
               assert cshape0 == vlen

            # get buff idx considering minimum hist and starting at 0
            i_st = st_ + min_h - 1
            i_en = en_ - min_h - 1

            c_st = min_h
            c_en = -min_h 

            if sv == 0:
               nshape = [nbframes] + list(cchunk.shape[1:])
               if chunkeach > 0:
                  nshape[0] = (nshape[0] - 1)/chunkeach + 1
               outputs[feat] = np.zeros(nshape,dtype='float32')
               i_st = 0 # at chunk 1, start from first frame
               c_st = 0
            else:
               assert last_en == i_st - 1
               if sv == subnum - 1:
                  i_en = nbframes - 1 # at last chunk, go til the end
                  c_en = cshape0 * chunkeach if chunkeach > 0 else cshape0
                  if chunkeach > 0: c_en += chunkeach

            last_en = i_en

            strprint = "add frames %d to %d (%d --> %d)(%s)" % (st_+c_st, en_+c_en, c_st, c_en, cchunk.shape)

            if chunkeach > 0:
               i_st /= chunkeach
               i_en /= chunkeach
               assert (c_st % chunkeach == 0) and (c_en % chunkeach == 0 or sv == subnum - 1)
               c_st /= chunkeach
               c_en /= chunkeach
               corr_st = (st_-1)/chunkeach + 1 + c_st
               corr_en = (en_-1)/chunkeach + 1 + c_en
               strprint += "\n   (chunks %d to %d)(%d --> %d)" % (corr_st, corr_en, c_st, c_en)
               if sv > 0:
                  assert last_en_c == i_st - 1
               last_en_c = i_en

            cchunk = cchunk[c_st:c_en] # shorten the current sub chunk
            outputs[feat][i_st:i_en+1] = cchunk

            if sv == subnum - 1:
               # for last sub chunks we need to take up to the end
               assert i_en + 1 == outputs[feat].shape[0]
               assert c_en >= cshape0

            print strprint
            #if chunkeach > 0:ipdb.set_trace()

   else: # simply load the (full) chunks
      for feat in paths:
         outputs[feat] = np.load('%s/%s.npy' % (vidpath, pat[feat])).squeeze(0)

   if n_out == 1:
      if not outputs['rgb'] is None:
         return outputs['rgb']
      else:
         return outputs['opf']
   else:
      return outputs['rgb'], outputs['opf']


if __name__ == "__main__":
   new_h, new_w = 360,640
   
   X = np.zeros((1920, 1080, 3))
   print padding(X,new_h, new_w, True)
   print padding(X,new_h, new_w, False).shape
   X = np.zeros((19, 108, 3))
   print padding(X,new_h, new_w, True)
   print padding(X,new_h, new_w, False).shape
   X = np.zeros((360, 640, 3))
   print padding(X,new_h, new_w, True)
   print padding(X,new_h, new_w, False).shape
