## node val time ---> 30 mins
## path val time ---> 20 mins
## test inference time ---> 165 mins
## node te time ---> less 10 min
## path te time ---> less 10 min
## nohup python run-scripts.py command-hc-node-val.txt > ./nohup/command-hc-node-val-att-1106 & 
## nohup python run-scripts.py command-hc-infer-test-0.txt > ./nohup/command-hc-infer-test-0-att-1108 & 
## nohup python run-scripts.py command-hc-path-val-0.txt > ./nohup/command-hc-path-val-0-att-1110 &

import sys, os, time
import numpy as np
from datetime import datetime

#time.sleep(60*60*3.6)

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
    time.sleep(60*60)
