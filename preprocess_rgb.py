import numpy as np
import os
import glob
from PIL import Image
import sys
import skimage
from skimage.transform import resize
import re
import pdb

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

root_dir = '/sequoia/data2/gcheron/UCF101/images/'
res_dir ='/sequoia/data2/gcheron/UCF101/I3D/'
str_pattern = 'image-%05d.jpg'
_h,_w=240,320
#_h,_w=224,224
minSize=min(_h,_w)
keepAR=True

#movies='/sequoia/data2/gcheron/UCF101/detection/OF_vidlist_all.txt'
movies = sys.argv[1]

with open(movies) as f:
    movie_list = f.readlines()
movie_list = [re.sub(' .*','',x.strip()) for x in movie_list]

for vidn in movie_list:
    #vidn=movie_list[427]
    path=os.path.join(root_dir,vidn)
    output_name = os.path.join(res_dir,vidn,'I3D_rgb.npy')


    if not(os.path.isfile(output_name)):
        rdi=os.path.join(res_dir,vidn)
        if not os.path.exists(rdi): os.makedirs(rdi)
        all_files = glob.glob(os.path.join(path, '*.jpg'))
        n_files = len(all_files)
        print path

        _tensor = np.zeros((1,n_files,_h,_w,3),dtype='float32')
        for i in range(n_files):
            im=np.array(Image.open(os.path.join(path,str_pattern%(i+1))))
            im=rescaling(im,minSize,keepAR) # rescale and send to [0,1]
            _tensor[0,i,:,:,:] = im

        _tensor=_tensor*2-1 # [0,1] --> [0,2] --> [-1,1]

        np.save(output_name,_tensor)
