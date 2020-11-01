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

 Start = -1
 End   = Best_epoch

 def fscore(act,prd):
    f1     = []
    pr     = []
    recall = []
    scale = 0.01
    End   = int(1/scale)+1
    for p in range(0,End):
        pr_temp, recall_temp, f1_temp,_ = precision_recall_fscore_support(act, prd>np.float32(p*scale), average='weighted')
        pr.append(pr_temp)
        recall.append(recall_temp)
        f1.append(f1_temp)
    F1 = np.array(f1)    
    #print (idx_dir)
    #print ('Best F1:', np.max(F1), 'with threshold', np.argmax(F1)*scale)
    print (np.max(F1),np.float32(np.argmax(F1)*scale))

 for ep in range(Start,End):
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_val/node/'
    print (idx_dir)

 for ep in range(Start,End):    
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_val/node/'

    act_name = sorted(glob.glob(home_dir+idx_dir+'/act_node_*.npy'))
    prd_name = sorted(glob.glob(home_dir+idx_dir+'/prd_*.npy'))    
    act = np.load(act_name[0])
    prd = np.load(prd_name[0])
    for i in range(1,len(act_name)):
        act = np.r_[act,np.load(act_name[i])]
        prd = np.r_[prd,np.load(prd_name[i])]
    fscore(act,prd)
    
else:
    
 model_dir  = str(sys.argv[2]) ###############################################
 Best_epoch = int(sys.argv[3]) ###############################################   
 Best_thr   = np.float32(sys.argv[4]) ############################################### 
    
 Start  = Best_epoch
 End    = Start+1
 THRESH = np.float32(Best_thr)

 def fscore(act,prd):
    f1     = []
    pr     = []
    recall = []
    scale = 0.01
    End   = int(1/scale)+1
    for p in range(0,End):
        pr_temp, recall_temp, f1_temp,_ = precision_recall_fscore_support(act, prd>np.float32(p*scale), average='weighted')
        pr.append(pr_temp)
        recall.append(recall_temp)
        if np.float32(p*scale) == THRESH:
           print (f1_temp,np.float32(p*scale))
        f1.append(f1_temp)
        
    F1 = np.array(f1)    
    #print (idx_dir)
    #print ('Best F1:', np.max(F1), 'with threshold', np.argmax(F1)*scale)
    print (np.max(F1),np.float32(np.argmax(F1)*scale))

 for ep in range(Start,End):
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_test/node/'
    print (idx_dir)

 for ep in range(Start,End):    
    idx_dir = model_dir+'-infer-ep-'+ str(ep) +'_test/node/'

    act_name = sorted(glob.glob(home_dir+idx_dir+'/act_node_*.npy'))
    prd_name = sorted(glob.glob(home_dir+idx_dir+'/prd_*.npy'))    
    act = np.load(act_name[0])
    prd = np.load(prd_name[0])
    for i in range(1,len(act_name)):
        act = np.r_[act,np.load(act_name[i])]
        prd = np.r_[prd,np.load(prd_name[i])]
    fscore(act,prd)









