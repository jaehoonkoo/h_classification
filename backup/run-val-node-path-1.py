import sys, os, time
import numpy as np
from datetime import datetime

file_name  = str(sys.argv[1]) # 'command-hc-node-val.txt'

try:
    a = open('/home/jkoo/code-tf/hc/'+file_name)
except FileNotFoundError:
    a = open('/scratch/jkoo/code-tf/hc/'+file_name)
    
f = a.readlines()
a.close()

n_file = len(f)
for i in range(n_file):    
    print (f[i])
    os.system(f[i])
    time.sleep(60*180)

