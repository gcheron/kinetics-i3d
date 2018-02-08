import pickle
import re
import math

dpath = "/sequoia/data2/gcheron/DALY/DALY_videoResolution.txt"
res = {}
rat = {}
with open(dpath) as f:
   content = f.readlines()

for r in content:
   m = re.match('([^ ]*) *([^ ]*) *([^ ]*)',r)
   h = int(m.group(2).strip())
   w = int(m.group(3).strip())
   ratio = math.ceil(10*float(w)/h)/10

   if not (h,w) in res:
      res[(h,w)] = 1
   else:
      res[(h,w)] += 1
   if not ratio in rat:
      rat[ratio] = 1
   else:
      rat[ratio] += 1
for i in res:
   print i,res[i]
print
for i in rat:
   print i,rat[i]
