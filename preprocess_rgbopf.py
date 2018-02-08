import numpy as np
import os
import glob
from PIL import Image
import sys
import skimage
from skimage.transform import resize
import re
import ipdb
from tstools import utils
import math

# command
# python preprocess_rgbopf.py /sequoia/data1/gcheron/code/torch/lstm_time_detection/dataset/splitlistsDALY/sub/all_vidlist_sub29.txt opf

def normalization(im):
    im = im-128
    return im/128

def center_crop(im):
    h,w = im.shape

    center = w/2
    start = center - h/2
    end = center + h/2

    return im[:,start:end]

def rescaling(im, dim=224,keep_ar=True):
      h, w = im.shape[:2]
      if keep_ar:
       if h > w:
         new_h,new_w = dim*h/w,dim
       else:
         new_h,new_w = dim,dim*w/h
      else: new_h,new_w = dim,dim

      if im.shape[:-1]==(new_h,new_w): # just send to [0,1]
         im=skimage.img_as_float(im)
      else:
         im=resize(im,(new_h,new_w)) # this also sends to [0,1] and convert
      return im

#movies='/sequoia/data2/gcheron/UCF101/detection/OF_vidlist_all.txt'
movies = sys.argv[1]
featstr = sys.argv[2]

dataset = 'DALY'
if dataset == 'UCF101':
   if featstr == 'rgb':
      root_dir = '/sequoia/data2/gcheron/UCF101/images/'
   elif featstr == 'opf':
      root_dir = '/sequoia/data2/gcheron/UCF101/OF_closest/'
   res_dir ='/sequoia/data2/gcheron/UCF101/I3D/full'
   _h,_w=240,320
elif dataset == 'DALY':
   if featstr == 'rgb':
      root_dir = '/sequoia/data2/gcheron/DALY/images/'
   elif featstr == 'opf':
      root_dir = '/sequoia/data2/gcheron/DALY/OF_closest/'
   res_dir ='/sequoia/data2/gcheron/DALY/I3D/full'
   #_h,_w=360,640
   _h,_w=240,320

str_pattern = 'image-%05d.jpg'
minSize=min(_h,_w)
keepAR=True

maxlen = 1000 # max len to send to GPU for feature extraction
extend_len = 100 # extend before and after (of this size) not to take chunks at boundaries which are inaccurate
reliable_len = maxlen - extend_len * 2 # chunk len which is accurate enough (middle of the chunk) (this len is different at vid bounds)
assert reliable_len > extend_len and reliable_len % extend_len == 0

with open(movies) as f:
    movie_list = f.readlines()
movie_list = [re.sub(' .*','',x.strip()) for x in movie_list]

for vidn in movie_list:
        #vidn=movie_list[427]
        path=os.path.join(root_dir,vidn)

    #if not(os.path.isfile(output_name)):
        rdi=os.path.join(res_dir,vidn)
        if not os.path.exists(rdi): os.makedirs(rdi)
        all_files = glob.glob(os.path.join(path, '*.jpg'))
        n_files = len(all_files)
        print path

        n_sub = 1 # first sub vid
        fst_len = maxlen - extend_len # first sub length (do not extend forward)
        remain_len = max(0, n_files - fst_len)
        n_sub += int(math.ceil(float(remain_len)/reliable_len)) # number of sub vids
        ml = min(n_files, reliable_len)

        for i_sub in range(n_sub):
            # frame bounds (starting at 1)
            i_st = 1 + i_sub * ml
            i_en = i_st + ml - 1

            # extend to get some dummy chunks before
            i_st = i_st - extend_len
            # extend backward
            i_en = i_en + extend_len

            # clamp
            i_st = max(1, i_st)
            i_en = min(n_files, i_en)

            if n_sub > 1:
               output_name = '%s/%s/I3D_%s_sub%d-%d-%d-%d.npy' % (res_dir, vidn, featstr, i_sub+1, n_sub, i_st, i_en)
            else:
               output_name = '%s/%s/I3D_%s.npy' % (res_dir, vidn, featstr)

            cur_n = i_en - i_st + 1

            if featstr == 'rgb':         
               _tensor = np.zeros((1,cur_n,_h,_w,3),dtype='float32')
            elif featstr == 'opf':
               _tensor = np.zeros((1,cur_n,_h,_w,2),dtype='float32')

            if n_sub > 1:
               print 'sub: %d/%d, f: %d/%d (%d fs) (%s)' % (i_sub+1, n_sub, i_st, i_en, cur_n, output_name)

            assert not ((i_sub == 0 ) ^ (i_st == 1))
            assert not ((i_sub == n_sub -1 ) ^ (i_en == n_files))
            assert i_en <= n_files

            for i in range(cur_n):
                i_im = i_st + i
                impath = os.path.join(path,str_pattern % (i_im) )
                im=np.array(Image.open(impath))
                if i == 0:
                  strinfo = 'load from %s' % impath
                elif i == cur_n - 1:
                  strinfo += '\n       to %s' % impath
    
                if dataset == 'DALY':
                   im = utils.padding(im,_h,_w) # pad and send to [0,1]
                else:
                   im=rescaling(im,minSize,keepAR) # rescale and send to [0,1]
                assert im.max() <= 1
                assert im.min() >= 0
   
                if featstr == 'opf':
                  im=im[:,:,:2] # drop last channel
 
                _tensor[0,i,:,:,:] = im
            print strinfo
            _tensor=_tensor*2-1 # [0,1] --> [0,2] --> [-1,1]
    
            np.save(output_name,_tensor)
