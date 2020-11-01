## Re produce predictions for path 
import numpy as np

tree = {
'0'	:	[	0],
'1'	:	[	1],
'2'	:	[	0,	2],
'3'	:	[	0,	3],
'4'	:	[	1,	4],
'5'	:	[	1,	5],
'6'	:	[	0,	2,	6	],
'7'	:	[	0,	2,	7	],
'8'	:	[	0,	2,	8	],
'9'	:	[	0,	2,	9	],
'10'	:	[	0,	2,	10	],
'11'	:	[	0,	2,	11	],
'12'	:	[	0,	2,	12	],
'13'	:	[	0,	2,	13	],
'14'	:	[	0,	2,	14	],
'15'	:	[	0,	2,	15	],
'16'	:	[	0,	3,	16	],
'17'	:	[	0,	3,	17	],
'18'	:	[	0,	3,	18	],
'19'	:	[	0,	3,	19	],
'20'	:	[	0,	3,	20	],
'21'	:	[	0,	3,	21	],
'22'	:	[	1,	4,	22	],
'23'	:	[	1,	4,	23	],
'24'	:	[	1,	4,	24	],
'25'	:	[	1,	4,	25	],
'26'	:	[	1,	4,	26	],
'27'	:	[	1,	4,	27	],
'28'	:	[	1,	5,	28	],
'29'	:	[	1,	5,	29	]}

def encoding_prd_path(prd, threshold):
    
    n_cls = 30
    ## create every possible paths from nodes 
    # actual node is same as actual prd 
    
    prd_path = np.zeros((len(prd),n_cls))
    ## turn on 
    prd = prd > threshold
    
    ## create label paths from prd 
    for i in range(len(prd)):
        for p in range(n_cls):
            act_path_idx = tree[str(p)]
            #print (prd[i][act_path_idx])
            prd_temp = np.sum(prd[i][act_path_idx])
            if len(act_path_idx) == prd_temp:
                prd_path[i][p] = 1.0
    return prd_path


import glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

###################################################### val #######################################################
home_dir = '/home/jkoo/code-tf/hc/tmp/'
mod = str(sys.argv[1])

if mod == 'val':
 model_dir  = str(sys.argv[2]) ###############################################
 Best_epoch = int(sys.argv[3]) ###############################################

 Start = Best_epoch - 1
 End   = Start + 1

 def fscore(act,prd):
    f1     = []
    pr     = []
    recall = []
    scale = 0.01
    End   = int(1/scale)+1
    for p in range(0,End):
        threshold = p*scale
        prd_path  = encoding_prd_path(prd,threshold)
        pr_temp, recall_temp, f1_temp,_ = precision_recall_fscore_support(act, prd_path, average='weighted')        
        pr.append(pr_temp)
        recall.append(recall_temp)
        f1.append(f1_temp)
    F1 = np.array(f1)    
    #print (idx_dir)
    #print ('Best F1:', np.max(F1), 'with threshold', np.argmax(F1)*scale)
    print ('Best VAL F1:',np.max(F1),'with threshold',np.argmax(F1)*scale)
    print (np.max(F1),np.argmax(F1)*scale)

 for ep in range(Start,End):
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_val/node/'
    print (idx_dir)

 for ep in range(Start,End):    
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_val/node/'

    act_name = sorted(glob.glob(home_dir+idx_dir+'/act_*.npy'))
    prd_name = sorted(glob.glob(home_dir+idx_dir+'/prd_*.npy'))    
    act = np.load(act_name[0])
    prd = np.load(prd_name[0])
    for i in range(1,len(prd_name)):
        act = np.r_[act,np.load(act_name[i])]
        #print (np.load(act_name[i]).shape)
        prd = np.r_[prd,np.load(prd_name[i])]
    print (act.shape,prd.shape)    
    fscore(act,prd)

#import glob, os, sys
#import numpy as np
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import precision_recall_fscore_support

else:
###################################################### test #######################################################

 model_dir  = str(sys.argv[2]) ###############################################
 Best_epoch = int(sys.argv[3]) ###############################################   
 Best_thr   = np.float32(sys.argv[4]) ###############################################     
 Start  = Best_epoch
 End    = Start+1
 THRESH = np.float32(Best_thr)

 def fscore_test(act,prd,THRESH):
    prd_path  = encoding_prd_path(prd,THRESH)
    pr_temp, recall_temp, f1_temp,_ = precision_recall_fscore_support(act, prd_path, average='weighted')        
    print ('Best TEST F1:',f1_temp,'with threshold',THRESH)
    print (f1_temp,THRESH)

 for ep in range(Start,End):
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_test/node/'
    print (idx_dir)

 for ep in range(Start,End):   
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_test/node/'

    act_name = sorted(glob.glob(home_dir+idx_dir+'/act_*.npy'))
    prd_name = sorted(glob.glob(home_dir+idx_dir+'/prd_*.npy'))
    act = np.load(act_name[0])
    prd = np.load(prd_name[0])
    for i in range(1,len(prd_name)):
        act = np.r_[act,np.load(act_name[i])]
        prd = np.r_[prd,np.load(prd_name[i])]
    fscore_test(act,prd,THRESH)    
