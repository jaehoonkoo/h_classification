'''
ref 1: bidirectional_rnn.py
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
ref 2: slim resnet
https://github.com/tensorflow/models/tree/master/slim
https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py

# server: allstate, nu
# mode1: vgg-16, resnet-50, hc-rnn-res-alt, hc-rnn-vgg-alt, hc-rsrnn-res-alt, hc-rsrnn-vgg-alt
# preprocessing: res & vgg
# things to change:
# ## server: celje, lj, dd0: folder names, loading models.
# ## data: cifar10, ilsvrc65: define weights, tree, label files, beam search

# updates
# oi-v1-0 01042019
## open-image added

# old updates
# Replace RNN to seq2seq
# 1120: train on slim-preprocessing
# 11202017 transform converted cnn features for seq2seq
# 1121: clean script
# 1201: implement on imagenet without crash
# v1-0 04082018
# v2-0 04202018
## 1) allstate part updated
## 2) vgg related options
## 3) hc1 to hc
## 4) Resnet_finetuned to cnn_finetuned
'''

from __future__ import print_function

import os, sys, argparse, glob
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import rnn
import random as rn
# Import slim
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import resnet_v1
#from tensorflow.contrib.slim.nets import vgg
#from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

# SE nets
#slim_path = "/home/jkoo/code-tf/slim/models/research/slim/"
slim_path = "/home/jkoo/code-tf/CBAM-tensorflow-slim/"
sys.path.append(slim_path)

#import tensorflow.contrib.slim as slim
from nets import resnet_v1
from nets import vgg
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


# Define options
parser = argparse.ArgumentParser()
parser.add_argument("-server","-server", dest='server',type=str,
                    help="Select which server", default="allstate")
parser.add_argument("-model","-model_name", dest='model',type=str,
                    help="Select model option", default="resnet-50")
parser.add_argument("-base","-cnn_base", dest='base',type=str,
                    help="Select cnn base option", default="res")
parser.add_argument("-residual_opt","-resi", dest='resi',type=int,
                    help="Select residual of rnn option", default=0)
parser.add_argument("-index","-idx", dest='idx',type=str,
                    help="Select idx to save checkpoint", default="idx-test")
parser.add_argument("-lr","-lr", dest='lr',type=float,
                    help="Select learning rate", default=0.1)
parser.add_argument("-epoch","-epoch", dest='epoch',type=int,
                    help="Select max epoch", default=100)
parser.add_argument("-bach_size","-batch", dest='batch',type=int,
                    help="Select batch size", default=32)
parser.add_argument("-optimizer","-opt", dest='opt',type=str,
                    help="Select optimizer option", default="adam")
parser.add_argument("-RNNInputDim", "-input", dest='input', type=int,
                    help="Select RNN input dimension", default=1)
parser.add_argument("-RNNhidden","-hidden", dest='hidden',type=str,
                    help="Select RNN hidden state", default="1")
parser.add_argument("-option1","-conversion", dest='option1',type=float,
                    help="Select conversion option", default=1)
parser.add_argument("-option2","-average", dest='option2',type=str,
                    help="Select layer averaging option", default="1;2;3;4")
parser.add_argument("-beam","-k", dest='beam',type=int,
                    help="Select beam threshold", default=2)
parser.add_argument("-restore","-restore", dest='restore',type=str,
                    help="Select restoring model", default="")
parser.add_argument("-print","-prt", dest='prt',type=str,
                    help="Select printing option", default="full")
parser.add_argument("-mode","-mode", dest='mode',type=str,
                    help="Select printing option", default="train")
parser.add_argument("-weight","-weight", dest='weight',type=str,
                    help="Select tree layer weights option", default="1;1;1;1")
parser.add_argument("-alt1","-alt1", dest='alt1',type=int,
                    help="Select alt option", default=-1)
parser.add_argument("-alt2","-alt2", dest='alt2',type=int,
                    help="Select alt option", default=5)
parser.add_argument("-check_dir","-check_dir", dest='check_dir',type=str,
                    help="Select check_dir", default="/home/jkoo/code-tf/hc/")
parser.add_argument("-seed","-seed", dest='seed',type=int,
                    help="Select seed", default=1234)
parser.add_argument("-attention","-attention", dest='attention',type=str,
                    help="Select cnn_attention", default=None)
args = parser.parse_args()

#
def create_checks(infer_dir):
    check_dir = args.check_dir
    f_name    = check_dir + infer_dir.split('/')[-2] + '.txt'
    f         = open(f_name,'w')
    currentDT = datetime.now()
    f.write('Finished at '+str(currentDT) +'\n')
    f.close()

# set random seed
def set_random_seed(run_seed):
    #run_0 = 1234
    #run_1 = 2468
    #run_2 = 1357    
    rn.seed(run_seed)
    np.random.seed(run_seed)
    tf.set_random_seed(run_seed)
    return

set_random_seed(args.seed)

# print arg_options
if args.model == "resnet-50" or args.model == "vgg-16":
   print( "Server: {}, Model Name: {}, Index: {}, N_epoch: {}, Learning rate: {}, Batch size: {}, Optimizer: {}, cnn_attention: {}".format(args.server, args.model, args.idx, args.epoch, args.lr, args.batch, args.opt, args.attention))
elif args.model == "hc":
   print( "Server: {}, Model Name: {}, Index: {}, N_epoch: {}, Learning rate: {}, Batch size: {}, Optimizer: {}, RNN Input Dimension: {}, RNN hidden state: {}, Conversion Option: {}, Layer Averaging Option: {}, Beam k: {}, Base: {}, alt1: {}, alt2: {}, cnn_ttention: {}".format(args.server, args.model, args.idx, args.epoch, args.lr, args.batch, args.opt, args.input, args.hidden, args.option1, args.option2, args.beam, args.base, args.alt1, args.alt2, args.attention))
else:
  print("Wrong model choice!!")
  sys.exit()

# Load check point
if args.server == "nu":
   #import hickle as hkl
   #home_dir = "/home/lab.analytics.northwestern.edu/jkoo/" ## deepdish
   home_dir = "/home/jkoo/"                                 ## lj and celje
   #data_dir = "/home/public/"                              ## deepdish
   #data_dir = "/scratch/jkoo/data/open-images/"            ## lj
   #save_dir = "/scratch/jkoo/"                             ## lj
   #data_dir = "/mnt/jkoo/"                                 ## celje
   #save_dir = "/mnt/jkoo/"                                 ## celje
   data_dir = "/home/jkoo/"             ## osp/wcdl0
   save_dir = "/home/jkoo/"                              ## osp/wcdl0
   checkpoints_dir = home_dir + "code-tf/weights/"
   if args.model == "hc":
      train_log_dir = save_dir + "code-tf/hc/tmp/hc_finetuned/" + args.idx + "/"
   else:
      train_log_dir = save_dir + "code-tf/hc/tmp/cnn_finetuned/" + args.idx + "/"
   if not os.path.exists(train_log_dir):
       os.makedirs(train_log_dir)
   if args.base == "res":
      input_dir = data_dir + "arrays/tr_npy_b256_slim/*.npy"
      label_dir = data_dir + "arrays/labels/tr_list_label_path.npy"
      input_dir_val = data_dir + "arrays/val_npy_b256_slim/*.npy"
      label_dir_val = data_dir + "arrays/labels/val_list_label_path.npy"
#      input_dir_te  = input_dir_val
#      label_dir_te  = label_dir_val  
      input_dir_te = data_dir + "arrays/test_npy_b256_slim/*.npy"
      label_dir_te = data_dir + "arrays/labels/te_list_label_path.npy"
   else: # vgg-16
      input_dir = data_dir + "arrays/tr_npy_b256_slim/*.npy"
      label_dir = data_dir + "arrays/labels/tr_list_label_path.npy"
      input_dir_val = data_dir + "arrays/val_npy_b256_slim/*.npy"
      label_dir_val = data_dir + "arrays/labels/val_list_label_path.npy"
#      input_dir_te  = input_dir_val
#      label_dir_te  = label_dir_val         
      input_dir_te = data_dir + "arrays/test_npy_b256_slim/*.npy"
      label_dir_te = data_dir + "arrays/labels/te_list_label_path.npy"
else:
   home_dir = "/data/jkooa/"
   data_dir = "/data/jkooa/"
   checkpoints_dir = home_dir + "Git/tf/slim/weights/"
   if args.model == "hc":
      train_log_dir = home_dir + "code/hc-tf/hc/tmp/hc_finetuned/" + args.idx + "/"
   else:
      train_log_dir = home_dir + "code/hc-tf/hc/tmp/cnn_finetuned/" + args.idx + "/"
   if not os.path.exists(train_log_dir):
       os.makedirs(train_log_dir)
   if args.base == "res":
      input_dir = data_dir + "preprocessing-hc/labeled_set/train_npy_b256_b_256/*"
      label_dir = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_tr_0-skew_label.npy"
      input_dir_val = data_dir + "preprocessing-hc/labeled_set/val_npy_b256_b_256/*"
      label_dir_val = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_te_0_label.npy"
      input_dir_te = data_dir + "preprocessing-hc/labeled_set/test_npy_b256_b_256/*"
      label_dir_te = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_test_1_label.npy"
   else: # vgg-16
      input_dir = data_dir + "preprocessing-hc/labeled_set/train_npy_b256_b_256-slim/*"
      label_dir = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_tr_0-skew_label.npy"
      input_dir_val = data_dir + "preprocessing-hc/labeled_set/val_npy_b256_b_256-slim/*"
      label_dir_val = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_te_0_label.npy"
      input_dir_te = data_dir + "preprocessing-hc/labeled_set/test_npy_b256_b_256-slim/*"
      label_dir_te = data_dir + "preprocessing-hc/labeled_set/labels/crop_hc_test_1_label.npy"

def get_init_fn(cnn_base):
    """Returns a function run by the chief worker to warm-start the training."""
    if cnn_base == "res":
       checkpoint_exclude_scopes= ["resnet_v1_50/logits", "predictions"]
       print("...Loading check point...", 'resnet_v1_50.ckpt')
    else: # cnn_base = "vgg":
       checkpoint_exclude_scopes= ['vgg_16/fc6','vgg_16/fc7','vgg_16/fc8'] # ['vgg_16/fc8']
       print("...Loading check point...", 'vgg_16.ckpt')
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

# Initialize arg options
model_name = args.model
rnn_dim = args.input
con_option = args.option1
avg_0 = args.option2.split(";")
avg_option = []
for i in range(len(avg_0)):
    avg_option.append([int(j) for j in avg_0[i].split(" ")])
w = [float(q) for q in args.weight.split(";")]
n_hidden = [int(q) for q in args.hidden.split(";")]    # args.hidden # hidden layer num of features
rnn_n_layer = int(len(n_hidden))
beam_k = args.beam
cnn_attention  = args.attention

# Parameters
learning_rate = args.lr
batch_size = args.batch
n_epoch = args.epoch
n_repeat = 1

# Network Parameters
if args.base == "res":
   n_height = 224                     # Resized image height
   n_width = 224                      # Resized image width
else:
   n_height = 224                     # Resized image height
   n_width = 224                      # Resized image width
n_channels = 3                        # Resized image channel
n_input = n_height*n_width*n_channels # Imagenet data input (img shape: 224*224)
n_steps = len(avg_option)+1           # timesteps

# how many parent nodes
if args.server == "nu":
   node_added = 0                     # oi s2s    
   #node_added = 7                     # ilsvrc65 s2s
else:
   node_added = 2                     # allstate s2s
   # node_added = 10                  # allstate rnn
# how many leaf nodes
if args.server == "nu":
   n_classes = 30                     ###### oi s2s    
   #n_classes = 33                    ###### ilsvrc65 s2s
   #n_classes = 57                    ###### ilsvrc65
   #n_classes = 1000                  # ilsvrc1k
   resnet_label_scale = 0             ###### oi s2s
   #resnet_label_scale = 7             ###### ilsvrc65
else:
   n_classes = 13                     ###### allstate s2s
   #n_classes = 18                     ###### allstate rnn

if args.base == "res":
   mean_r = 123.68  # ilsvrc1k-123.68  | cifar100-129.30417
   mean_g = 116.779 # ilsvrc1k-116.779 | cifar100-124.06996
   mean_b = 103.935 # ilsvrc1k-103.935 | cifar100-112.43405
else:
   mean_r = 123.68 # ilsvrc1k-123.68  | cifar100-129.30417
   mean_g = 116.779 # ilsvrc1k-116.779 | cifar100-124.06996
   mean_b = 103.935 # ilsvrc1k-103.935 | cifar100-112.43405
std_r = 1 # | cifar100-68.14695
std_g = 1 # | cifar100-65.37863
std_b = 1 # | cifar100-70.40022

# tf Graph input
x = tf.placeholder("float", [None, n_height, n_width, n_channels])
if args.server == 'nu':
   y      = tf.placeholder("float", [None, n_classes+node_added]) ## oi ilsvrc65 s2s
   y_path = tf.placeholder("float", [None, n_classes+node_added]) ## oi s2s
else:
   if model_name == "hc":
      y = tf.placeholder("float", [None, n_classes+node_added]) ## allstate s2s
   else: ## allstate cnn-s2s
      y = tf.placeholder("float", [None, n_classes]) ## allstate s2s
y_encoded = tf.placeholder("float", [None, n_steps, n_classes+node_added+2]) ## s2s: +2 is added.

if model_name == "hc":
 if args.base == "res":
   # Define dimension of 4 blocks
   bl1_dim = [28,28,256] #
   bl2_dim = [14,14,512] #
   bl3_dim = [7,7,1024]  #
   bl4_dim = [7,7,2048]  #
   # Define polling conversion dimension
   l1_scale = 12544 #
   l2_scale = 8192  #
   l3_scale = 16384 #
   l4_scale = 8192  #
 elif args.base == "vgg":
   # Define dimension of 4 blocks (VGG)
   bl1_dim = [14,14,512] #
   bl2_dim = [14,14,512] #
   bl3_dim = [14,14,512] #
   bl4_dim = [7,7,512]   #
   # Define polling conversion dimension
   l1_scale = 8192  #
   l2_scale = 8192  #
   l3_scale = 8192  #
   l4_scale = 8192  #
 else:
   print ('wrong CNN base!')
   sys.exit()

 with tf.name_scope('HC_CONV'):
  weights = {
    ## this is for output from lstm
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    #'out0': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added]) * tf.sqrt(2.0/(2*n_hidden[-1]))),
    #'out1': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added]) * tf.sqrt(2.0/(2*n_hidden[-1]))),
    #'out2': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added]) * tf.sqrt(2.0/(2*n_hidden[-1]))),
    # 'out3': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added],mean=0.0))
    'W' : tf.Variable(tf.random_uniform([2*n_hidden[-1], n_classes+node_added+2], -1, 1), dtype=tf.float32) ## s2s
  }
  biases = {
    ## this is for output from lstm
    #'out0': tf.Variable(tf.zeros([n_classes+node_added])),
    #'out1': tf.Variable(tf.zeros([n_classes+node_added])),
    #'out2': tf.Variable(tf.zeros([n_classes+node_added])),
    # 'out3': tf.Variable(tf.random_normal([n_classes+node_added],mean=0.0))
    'b' : tf.Variable(tf.zeros([n_classes+node_added+2]), dtype=tf.float32)
  }
  if con_option == 1:
    weights['wd1'] = tf.Variable(tf.random_normal([bl1_dim[0]*bl1_dim[1]*bl1_dim[2], rnn_dim]) * tf.sqrt(2.0/(bl1_dim[0]*bl1_dim[1]*bl1_dim[2])))
    weights['wd2'] = tf.Variable(tf.random_normal([bl2_dim[0]*bl2_dim[1]*bl2_dim[2], rnn_dim]) * tf.sqrt(2.0/(bl2_dim[0]*bl2_dim[1]*bl2_dim[2])))
    weights['wd3'] = tf.Variable(tf.random_normal([bl3_dim[0]*bl3_dim[1]*bl3_dim[2], rnn_dim]) * tf.sqrt(2.0/(bl3_dim[0]*bl3_dim[1]*bl3_dim[2])))
    weights['wd4'] = tf.Variable(tf.random_normal([bl4_dim[0]*bl4_dim[1]*bl4_dim[2], rnn_dim]) * tf.sqrt(2.0/(bl4_dim[0]*bl4_dim[1]*bl4_dim[2])))
    biases['bd1'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['bd2'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['bd3'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['bd4'] = tf.Variable(tf.zeros([rnn_dim]))
  elif con_option == 2:
    weights['rwc1'] = tf.Variable(tf.random_normal([bl1_dim[0], bl1_dim[1], bl1_dim[2], rnn_dim]))
    weights['rwc2'] = tf.Variable(tf.random_normal([bl2_dim[0], bl2_dim[1], bl2_dim[2], rnn_dim]))
    weights['rwc3'] = tf.Variable(tf.random_normal([bl3_dim[0], bl3_dim[1], bl3_dim[2], rnn_dim]))
    weights['rwc4'] = tf.Variable(tf.random_normal([bl4_dim[0], bl4_dim[1], bl4_dim[2], rnn_dim]))
    biases['rbc1'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['rbc2'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['rbc3'] = tf.Variable(tf.zeros([rnn_dim]))
    biases['rbc4'] = tf.Variable(tf.zeros([rnn_dim]))

  elif con_option == 3.1 or con_option == 3.2 :
    weights['wd1p'] = tf.Variable(tf.random_normal([l1_scale, rnn_dim]) * tf.sqrt(2.0/(l1_scale)))
    weights['wd2p'] = tf.Variable(tf.random_normal([l2_scale, rnn_dim]) * tf.sqrt(2.0/(l2_scale)))
    weights['wd3p'] = tf.Variable(tf.random_normal([l3_scale, rnn_dim]) * tf.sqrt(2.0/(l3_scale)))
    weights['wd4p'] = tf.Variable(tf.random_normal([l4_scale, rnn_dim]) * tf.sqrt(2.0/(l4_scale)))

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
def maxpool2d_1(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k-1, k-1, 1],
                          padding='SAME')
def avgpool2d(x, k=2):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
def rconv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    #x = tf.nn.bias_add(x, b)
    #return tf.nn.relu(x)
    return x

def BiRNN(x, weights, biases, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer, cnn_base, cnn_attention):

    # random cropping
    if model_name == "resnet-50" or cnn_base == "res":
       x = tf.random_crop(x,[batch_size,224,224,3])

    if model_name == "resnet-50":
       # Load resnet-50 model
       with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net_, end_points_ = resnet_v1.resnet_v1_50(x, n_classes+node_added, is_training=True, attention_module=cnn_attention)
       return net_
       #return net_[:,0,0,:]

    elif model_name == "vgg-16":
       # Load vgg-16 model
       with slim.arg_scope(vgg.vgg_arg_scope()):
            net_, end_points_ = vgg.vgg_16(x, n_classes+node_added, is_training=True)
       return net_

    # return hc
    elif model_name == "hc":
       # Load resnet-50 or vgg-16
       if cnn_base == "res":
          with slim.arg_scope(resnet_v1.resnet_arg_scope()):
               net_, end_points_ = resnet_v1.resnet_v1_50(x, n_classes+node_added, is_training=True, attention_module=cnn_attention)
          var_names_cnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1')]
          for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
              var_names_cnn.append(i)
          # Define 4 blocks
          bl1_name_ = 'resnet_v1_50/block1'
          bl2_name_ = 'resnet_v1_50/block2'
          bl3_name_ = 'resnet_v1_50/block3'
          bl4_name_ = 'resnet_v1_50/block4'

       else: # cnn_base == "vgg"
          with slim.arg_scope(vgg.vgg_arg_scope()):
               net_, end_points_ = vgg.vgg_16(x, n_classes+node_added, is_training=True)
          var_names_cnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')]
          for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
              var_names_cnn.append(i)
          # Define 4 blocks
          bl1_name_ = 'vgg_16/conv5/conv5_1'
          bl2_name_ = 'vgg_16/conv5/conv5_2'
          bl3_name_ = 'vgg_16/conv5/conv5_3'
          bl4_name_ = 'vgg_16/pool5'

       blocks = {
             'bl1': end_points_[bl1_name_],
             'bl2': end_points_[bl2_name_],
             'bl3': end_points_[bl3_name_],
             'bl4': end_points_[bl4_name_]
       }
       conv1 = blocks['bl1']
       conv2 = blocks['bl2']
       conv3 = blocks['bl3']
       conv4 = blocks['bl4']
       if con_option == 1:
          print('Conversion con_option 1: Linear conversion')
          # 2-1. Trainable incidence matrix
          l1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
          ##l1 =  tf.add(tf.matmul(l1, weights['wd1']), biases['bd1'])
          l1 = tf.matmul(l1, weights['wd1'])
          #l1 = tf.nn.relu(l1)
          l2 = tf.reshape(conv2, [-1, weights['wd2'].get_shape().as_list()[0]])
          l2 = tf.matmul(l2, weights['wd2'])
          #l2 = tf.nn.relu(l2)
          l3 = tf.reshape(conv3, [-1, weights['wd3'].get_shape().as_list()[0]])
          l3 = tf.matmul(l3, weights['wd3'])
          #l3 = tf.nn.relu(l3)
          l4 = tf.reshape(conv4, [-1, weights['wd4'].get_shape().as_list()[0]])
          l4 = tf.matmul(l4, weights['wd4'])
          #l4 = tf.nn.relu(l3)
       elif con_option == 2:
          print('Conversion con_option 2: Convolutonal conversion')
          # 2-4. Trainable conv conversion
          l1 = rconv2d(conv1, weights['rwc1'], biases['rbc1'])
          #l1 = maxpool2d(l1, k=1)
          l2 = rconv2d(conv2, weights['rwc2'], biases['rbc2'])
          #l2 = maxpool2d(l2, k=1)
          l3 = rconv2d(conv3, weights['rwc3'], biases['rbc3'])
          #l3 = maxpool2d(l3, k=1)
          l4 = rconv2d(conv4, weights['rwc4'], biases['rbc4'])
          #l4 = maxpool2d(l4, k=1)
          #l1 = tf.reshape(l1, [-1,weights['wd1'].get_shape().as_list()[1]])
          #l2 = tf.reshape(l2, [-1,weights['wd2'].get_shape().as_list()[1]])
          #l3 = tf.reshape(l3, [-1,weights['wd3'].get_shape().as_list()[1]])
          #l4 = tf.reshape(l4, [-1,weights['wd4'].get_shape().as_list()[1]])
          l1 = tf.reshape(l1, [-1,weights['rwc1'].get_shape().as_list()[3]])
          l2 = tf.reshape(l2, [-1,weights['rwc2'].get_shape().as_list()[3]])
          l3 = tf.reshape(l3, [-1,weights['rwc3'].get_shape().as_list()[3]])
          l4 = tf.reshape(l4, [-1,weights['rwc4'].get_shape().as_list()[3]])
       elif con_option == 3.1:
          print('Conversion con_option 3-1: Pooling conversion - pooling and linear')
          # 2-2. conversion avg pooled cnn layer
          def dim_calc(W,D,rnn_dim):
              #return np.max([W*np.sqrt(D/rnn_dim),1])
              return np.max([W*np.sqrt(D/float(rnn_dim)),1])
          ## block1: 28 * 28 * 256 = 200704
          l1 = maxpool2d(conv1, k=2) # 28 * 28 * 256 -> k = 2 -> 14 * 14 * 256 = 50176
          l1 = maxpool2d(l1, k=2)    # 14 * 14 * 256 -> k = 2 -> 7 * 7 * 256 = 12544
          #l1 = maxpool2d(l1, k=2)    # 7 * 7 * 256 -> k = 2 -> 4 * 4 * 256 = 4096
          # block2: 14 * 14 * 512 = 100352
          l2 = maxpool2d(conv2, k=2) # 14 * 14 * 512 -> k = 2 -> 7 * 7 * 512 = 25088
          l2 = maxpool2d(l2, k=2)    # 7 * 7 * 512 -> k = 2 -> 4 * 4 * 512 = 8192
          #if cnn_base == "vgg":
          #   l2 = maxpool2d(l2, k=2) ################################################################################# VGG
          # block3: 7 * 7 * 1024 = 50176
          l3 = maxpool2d(conv3, k=2) # 7 * 7 * 1024 -> k = 2 -> 4 * 4 * 1024 = 16384
          #l3 = maxpool2d(l3, k=2)    # 4 * 4 * 1024 -> k = 2 -> 2 * 2 * 1024 = 4096
          if cnn_base == "vgg":
             l3 = maxpool2d(l3, k=2) ################################################################################# VGG
          # block4: 7 * 7 * 2048 = 100352
          l4 = maxpool2d(conv4, k=2) # 7 * 7 * 2048 -> k = 2 -> 4 * 4 * 2048 = 32768
          if cnn_base == "res":
             l4 = maxpool2d(l4, k=2) ################################################################################# RES
          #l4 = maxpool2d(l4, k=2)    # 4 * 4 * 2048 -> k = 2 -> 2 * 2 * 2048 = 8192
          l1 = tf.reshape(l1, [-1,weights['wd1p'].get_shape().as_list()[0]])
          l1 = tf.matmul(l1, weights['wd1p'])
          l2 = tf.reshape(l2, [-1,weights['wd2p'].get_shape().as_list()[0]])
          l2 = tf.matmul(l2, weights['wd2p'])
          l3 = tf.reshape(l3, [-1,weights['wd3p'].get_shape().as_list()[0]])
          l3 = tf.matmul(l3, weights['wd3p'])
          l4 = tf.reshape(l4, [-1,weights['wd4p'].get_shape().as_list()[0]])
          l4 = tf.matmul(l4, weights['wd4p'])
          if cnn_base == 'vgg':
             l1 = tf.nn.sigmoid(l1)
             l2 = tf.nn.sigmoid(l2)
             l3 = tf.nn.sigmoid(l3)
             l4 = tf.nn.sigmoid(l4)
       elif con_option == 3.2:
          print('Conversion con_option 3-2: Pooling conversion - pooling and p-max')
          # 2-2. conversion avg pooled cnn layer
          def dim_calc(W,D,rnn_dim):
              return np.max([W*np.sqrt(D/float(rnn_dim)),1]) #return np.max([W*np.sqrt(D/rnn_dim),1]) #return int(W + 1 - np.sqrt(D/float(rnn_dim)))
          # block1: 28 * 28 * 256 = 200704
          l1 = maxpool2d(conv1, k=2) # 28 * 28 * 256 -> k = 2 -> 14 * 14 * 256 = 50176 \\\
          l1 = maxpool2d(l1, k=2)    # 14 * 14 * 256 -> k = 2 -> 7 * 7 * 256 = 12544
          l1 = maxpool2d(l1, k=2)    # 7 * 7 * 256 -> k = 2 -> 4 * 4 * 256 = 4096
          # block2: 14 * 14 * 512 = 100352
          l2 = maxpool2d(conv2, k=2) # 14 * 14 * 512 -> k = 2 -> 7 * 7 * 512 = 25088
          l2 = maxpool2d(l2, k=2)    # 7 * 7 * 512 -> k = 2 -> 4 * 4 * 512 = 8192
          # block3: 7 * 7 * 1024 = 50176
          l3 = maxpool2d(conv3, k=2) # 7 * 7 * 1024 -> k = 2 -> 4 * 4 * 1024 = 16384
          l3 = maxpool2d(l3, k=2)    # 4 * 4 * 1024 -> k = 2 -> 2 * 2 * 1024 = 4096
          # block4: 7 * 7 * 2048 = 100352
          l4 = maxpool2d(conv4, k=2) # 7 * 7 * 2048 -> k = 2 -> 4 * 4 * 2048 = 32768
          l4 = maxpool2d(l4, k=2)    # 4 * 4 * 2048 -> k = 2 -> 2 * 2 * 2048 = 8192
          l1 = tf.reshape(l1, [-1,l1.get_shape().as_list()[1]*l1.get_shape().as_list()[2]*l1.get_shape().as_list()[3]]) #
          l1, void = tf.nn.top_k(l1,rnn_dim)
          l2 = tf.reshape(l2, [-1,l2.get_shape().as_list()[1]*l2.get_shape().as_list()[2]*l2.get_shape().as_list()[3]]) #
          l2, void = tf.nn.top_k(l2,rnn_dim)
          l3 = tf.reshape(l3, [-1,l3.get_shape().as_list()[1]*l3.get_shape().as_list()[2]*l3.get_shape().as_list()[3]]) #
          l3, void = tf.nn.top_k(l3,rnn_dim)
          l4 = tf.reshape(l4, [-1,l4.get_shape().as_list()[1]*l4.get_shape().as_list()[2]*l4.get_shape().as_list()[3]]) #
          l4, void = tf.nn.top_k(l4,rnn_dim)
       elif con_option == 4:
          # 2-3. Max p operator
          print('Conversion con_option 4: p-max')
          l1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
          l1, void = tf.nn.top_k(l1,rnn_dim)
          l2 = tf.reshape(conv2, [-1, weights['wd2'].get_shape().as_list()[0]])
          l2, void = tf.nn.top_k(l2,rnn_dim)
          l3 = tf.reshape(conv3, [-1, weights['wd3'].get_shape().as_list()[0]])
          l3, void = tf.nn.top_k(l3,rnn_dim)
          l4 = tf.reshape(conv4, [-1, weights['wd4'].get_shape().as_list()[0]])
          l4, void = tf.nn.top_k(l4,rnn_dim)
       else:
          print ("Wrong conversion operation!")
          sys.exit()
       # Averaging converted cnn layers for rnn
       rnn_input = []
       rnn_inputs = []
       rnn_inputs.append(l4)
       rnn_inputs.append(l3)
       rnn_inputs.append(l2)
       rnn_inputs.append(l1)
       for i in range(len(avg_option)):
           rnn_input.append(tf.add_n([rnn_inputs[u-1] for u in avg_option[i]])/len(avg_option[i]))

       ## Seq2seq
       rnn_input_ = tf.stack([rnn_input[0],rnn_input[1],rnn_input[2]]) ## tensor of [(batch_size,rnn_input),(batch_size,rnn_input),(batch_size,rnn_input)]

       PAD = 0
       EOS = 1
       vocab_size = rnn_dim           ## rnn input dimension at each time step
       input_embedding_size = rnn_dim ## hidden layer dimension
       vocab_size_target = n_classes+node_added+2

       encoder_hidden_units = n_hidden[0] #
       decoder_hidden_units = n_hidden[0] * 2

       encoder_inputs_length = tf.ones((batch_size,),dtype=tf.int32)
       encoder_inputs_embedded = rnn_input_                                         #### [n_steps, None, rnn_input]

       # encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
       if cnn_base == 'res':
          encoder_cell = LSTMCell(encoder_hidden_units, initializer=tf.orthogonal_initializer(), forget_bias=1.0, activation=tf.nn.relu)
       else:
          encoder_cell = LSTMCell(encoder_hidden_units, initializer=tf.orthogonal_initializer(), forget_bias=1.0, activation=tf.nn.relu)
       ((encoder_fw_outputs,encoder_bw_outputs),(encoder_fw_final_state,
         encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True,
                                    scope='HC_RNN_encoder'))
       encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
       encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
       encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
       encoder_final_state = LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
       if cnn_base == 'res':
          decoder_cell = LSTMCell(decoder_hidden_units, initializer=tf.orthogonal_initializer(), forget_bias=1.0, activation=tf.nn.relu)
       else:
          decoder_cell = LSTMCell(decoder_hidden_units, initializer=tf.orthogonal_initializer(), forget_bias=1.0, activation=tf.nn.relu)
       encoder_max_time = tf.ones((1,1),dtype=tf.int32)*3
       decoder_lengths = encoder_inputs_length + 3

       W = weights['W']
       b = biases['b']

       assert EOS == 1 and PAD == 0
       eos_val = np.zeros((batch_size,vocab_size_target), dtype=np.float32)
       eos_val[:,0] = 1
       eos_step_embedded = tf.constant(value=eos_val) # rnn_input_[0,:,:] # it should be [(batch_size,rnn_input),(batch_size,rnn_input),(batch_size,rnn_input)]

       def loop_fn_initial():
           initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
           initial_input = eos_step_embedded
           initial_cell_state = encoder_final_state
           initial_cell_output = None
           initial_loop_state = None  # we don't need to pass any additional information
           return (initial_elements_finished,
                   initial_input,
                   initial_cell_state,
                   initial_cell_output,
                   initial_loop_state)

       def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
           def get_next_input():
               output_logits = tf.add(tf.matmul(previous_output, W), b)
               # prediction = tf.argmax(output_logits, axis=1)
               next_input = tf.nn.softmax(output_logits) # rnn_input_[1,:,:]
               # next_input = tf.nn.embedding_lookup(embeddings, prediction)
               return next_input

           elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                         # defining if corresponding sequence has ended
           finished = tf.reduce_all(elements_finished) # -> boolean scalar
           input = get_next_input()
           state = previous_state
           output = previous_output
           loop_state = None

           return (elements_finished,
                   input,
                   state,
                   output,
                   loop_state)

       def loop_fn(time, previous_output, previous_state, previous_loop_state):
           if previous_state is None:    # time == 0
              assert previous_output is None and previous_state is None
              return loop_fn_initial()
           else:
              return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

       decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn, scope='HC_RNN_decoder')

       var_names_rnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_RNN_encoder')]
       for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_RNN_decoder'):
           var_names_rnn.append(i)
       for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
           var_names_rnn.append(i)

       decoder_outputs = decoder_outputs_ta.stack()

       decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
       decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
       decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
       outputs = tf.reshape(decoder_logits_flat, (4, batch_size, n_classes+node_added+2))

       # Linear activation, using rnn inner loop last output
       if args.resi:
          ### (old) softmax(fc(sig(cnn_input)+sig(lstm output)))
          # return [i+tf.nn.sigmoid(j) for i,j in zip(rnn_input, outputs)], var_names_cnn, var_names_rnn
          ### (new) softmax(fc(cnn_input+A*(lstm output)))
          return [tf.add(i,j) for i,j in zip(rnn_input, outputs)], var_names_cnn, var_names_rnn #
       else:
          return outputs, var_names_cnn, var_names_rnn

    ## return error message
    else:
       print (".......Wrong model choice!!.......")
       sys.exit()

if model_name == 'hc':
   pred, var_names_cnn, var_names_rnn = BiRNN(x, weights, biases, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer, args.base, cnn_attention)
else:
   pred = BiRNN(x, None, None, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer, args.base, cnn_attention)

# encoding function
pre_path = {
'0'	:	[	1,	31,	31	],
'1'	:	[	2,	31,	31	],
'2'	:	[	1,	3,	31	],
'3'	:	[	1,	4,	31	],
'4'	:	[	2,	5,	31	],
'5'	:	[	2,	6,	31	],
'6'	:	[	1,	3,	7	],
'7'	:	[	1,	3,	8	],
'8'	:	[	1,	3,	9	],
'9'	:	[	1,	3,	10	],
'10'	:	[	1,	3,	11	],
'11'	:	[	1,	3,	12	],
'12'	:	[	1,	3,	13	],
'13'	:	[	1,	3,	14	],
'14'	:	[	1,	3,	15	],
'15'	:	[	1,	3,	16	],
'16'	:	[	1,	4,	17	],
'17'	:	[	1,	4,	18	],
'18'	:	[	1,	4,	19	],
'19'	:	[	1,	4,	20	],
'20'	:	[	1,	4,	21	],
'21'	:	[	1,	4,	22	],
'22'	:	[	2,	5,	23	],
'23'	:	[	2,	5,	24	],
'24'	:	[	2,	5,	25	],
'25'	:	[	2,	5,	26	],
'26'	:	[	2,	5,	27	],
'27'	:	[	2,	5,	28	],
'28'	:	[	2,	6,	29	],
'29'	:	[	2,	6,	30	]}

def preprocessing(batch_x_name, cnn_base, server):
    if cnn_base == "res" or model_name == "resnet-50":
       if server == 'nu':
          #batch_x = hkl.load(batch_x_name) ## ilsvrc65
          batch_x = np.load(batch_x_name)  ## oi 
       else: ## allstate
          batch_x = np.load(batch_x_name)
       # batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
    else: ## cnn_base == "vgg"
       batch_x = np.load(batch_x_name)
    batch_x[:,:,:,0] -= mean_r
    batch_x[:,:,:,1] -= mean_g
    batch_x[:,:,:,2] -= mean_b
    batch_x[:,:,:,0] /= std_r
    batch_x[:,:,:,1] /= std_g
    batch_x[:,:,:,2] /= std_b
    return batch_x

## hc-rnn encoding
def encode(pre_path,batch_y,batch_size,n_steps,n_classes,node_added):
    path = np.zeros((batch_size,n_steps,n_classes+node_added) ,dtype=int)
    for i in range(batch_size):
        label = np.argmax(batch_y[i])
        for j in range(n_steps):
            path[i,j,pre_path[str(label)][j]] = 1
    return path

## hc-s2s encoding ### oi
def encode_s2s(pre_path,batch_y,batch_size,n_steps,n_classes,node_added):
    cnn_path = {
'0'	:	[	0,	0,	0	],
'1'	:	[	1,	1,	1	],
'2'	:	[	0,	2,	2	],
'3'	:	[	0,	3,	3	],
'4'	:	[	1,	4,	4	],
'5'	:	[	1,	5,	5	],
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
    
    label_vec_node = np.zeros((batch_size,n_classes),dtype=int)
    for i in range(batch_size):
        label_temp_idx = np.argwhere(batch_y[i]>0)
        for j in range(len(label_temp_idx)):
            temp = label_temp_idx[j][0]
            label_vec_temp = cnn_path[str(temp)]
            for vec_temp in label_vec_temp:
                label_vec_node[i,int(vec_temp)] = 1 
    # hc label vectore
    path_target = np.zeros((batch_size,n_steps,n_classes+node_added+2), dtype=int) ## oi ilsvrc65 s2s    
    path_target[:,n_steps-1,-1] = 1  # n_step = 4 / 012 are in tree, 3 is END
    level_1 = range(0,2)
    level_2 = range(2,6)
    level_3 = range(6,30)
    for i in range(batch_size):
        # check if there is END at each layer 
        if np.sum(label_vec_node[i,level_3]) == 0.0:
            path_target[i,2,-1] = 1
        elif np.sum(label_vec_node[i,level_2]) == 0.0:   
            path_target[i,1,-1] = 1
        elif np.sum(label_vec_node[i,level_1]) == 0.0:
            print ('Error: sample is not labeled')
            sys.exit()
        # encoding each layer    
        nonzero = np.argwhere(label_vec_node[i]>0)
        for l in level_1:
            if l in nonzero:
                path_target[i,0,l+1] = 1
        for l in level_2:
            if l in nonzero:
                path_target[i,1,l+1] = 1
        for l in level_3:
            if l in nonzero:
                path_target[i,2,l+1] = 1
    return batch_y, label_vec_node, path_target  ## batch_y=path, label_vec_node=node, path_target=node only for hc

## hc-s2s encoding #### ilsvrc65 and allstate
def encode_s2s_old(pre_path,batch_y,batch_size,n_steps,n_classes,node_added):
    label_target = np.zeros((batch_size,n_classes+node_added), dtype=int) ## oi ilsvrc65 s2s
    ## label_target = np.zeros((batch_size,n_classes), dtype=int)   ## allstate s2s
    path_target = np.zeros((batch_size,n_steps,n_classes+node_added+2), dtype=int) ## oi ilsvrc65 s2s    
    path_target[:,n_steps-1,-1] = 1  # n_step = 4 / 012 are in tree, 3 is END
    for i in range(batch_size):
        label = np.argmax(batch_y[i])  ## label = {0,...,17} allstate and ilsvrc65
        label_target[i,pre_path[str(label)][3]-1] = 1  ## ilsvrc65 s2s
        ## label_target[i,pre_path[str(label)][3]] = 1  ## allstate s2s
        for j in range(n_steps-1):
            path_target[i,j,pre_path[str(label)][j]] = 1
    return label_target, path_target  ## label_target = for cnn, path_target = for s2s

## CNN encoding
def encode_s2s_(pre_path,batch_y,batch_size,n_steps,n_classes,node_added):
    label_target = np.zeros((batch_size,n_classes+node_added), dtype=int) ## ilsvrc65 s2s
    ## label_target = np.zeros((batch_size,n_classes), dtype=int)   ## allstate s2s
    path_target = np.zeros((batch_size,n_steps,n_classes+node_added+2), dtype=int)
    path_target[:,n_steps-1,-1] = 1  # n_step = 4 / 012 are in tree, 3 is END
    for i in range(batch_size):
        label = np.argmax(batch_y[i]) + 7
        label_target[i,pre_path[str(label)][3]-1] = 1
        label = np.argmax(batch_y[i]) + 7 ## ilsvrc65
        label_target[i,pre_path[str(label)][3]-1] = 1  ## ilsvrc65
        ## label = np.argmax(batch_y[i]) ## allstate
        ## label_target[i,pre_path[str(label)][3]] = 1  ## allstate
        for j in range(n_steps-1):
            path_target[i,j,pre_path[str(label)][j]] = 1
    return label_target, path_target  ## label_target = for cnn, path_target = for s2s

if args.model == "resnet-50"  or args.model == "vgg-16":
   # Define loss and optimizer
   cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)) # multi-label oi
   pred_cnn = tf.nn.sigmoid(pred)
   # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # multi-class
   if args.opt == "adam":
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   elif args.opt == "sgd":
      optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)
   else:
      print("Choose proper Optimizer!!")
      sys.exit()
   # Evaluate model
   # correct nodes - Recall of all correct node
   indices_node = tf.where(tf.greater(y, 0)) 
   act_gather_node = tf.gather_nd(y, indices_node)
   prd_gather_node = tf.gather_nd(tf.round(tf.nn.sigmoid(pred)), indices_node)
   correct_pred_node = tf.equal(prd_gather_node,act_gather_node)
   accuracy = tf.reduce_mean(tf.cast(correct_pred_node, tf.float32)) ### node accuracy
   #accuracy_path_ = tf.reduce_mean(tf.cast(correct_pred_path_, tf.float32))     
   #correct_pred = tf.equal(tf.round(tf.nn.sigmoid(pred)), y) # multi-label oi
   #correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred),1), tf.argmax(y,1)) # multi-class 
   # correct path 1 - Recall of path assuming remark 1 is true
   indices = tf.where(tf.greater(y_path, 0)) 
   act_gather = tf.gather_nd(y_path, indices)
   prd_gather = tf.gather_nd(tf.round(tf.nn.sigmoid(pred)), indices)
   correct_pred_path = tf.equal(prd_gather,act_gather)
   accuracy_path = tf.reduce_mean(tf.cast(correct_pred_path, tf.float32)) 
   # correct path 2 - Recall of all correct node
   #indices_ = tf.where(tf.greater(y, 0)) 
   #act_gather_ = tf.gather_nd(y, indices_)
   #prd_gather_ = tf.gather_nd(tf.round(tf.nn.sigmoid(pred)), indices_)
   #correct_pred_path_ = tf.equal(prd_gather_,act_gather_)
   #accuracy_path_ = tf.reduce_mean(tf.cast(correct_pred_path_, tf.float32)) 

   def beam_cnn(sample,pred,y_path,y_node,n_classes):
       cnn_path_ = {
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
       indices = np.argwhere(y_path[sample]>0)
       correct_count, path_count = 0, 0
       for temp in indices:
           path_idx   = cnn_path_[str(temp[0])]
           act_gather = y_node[sample,path_idx].astype(np.float32)
           prd_gather = np.round(pred[sample,path_idx])
           correct_pred_path = np.equal(prd_gather,act_gather)
           if np.sum(correct_pred_path) == len(path_idx):
              correct_count += 1 
           path_count += 1 
       return path_count, correct_count

elif args.model == "hc": # ce
   # Define loss and optimizer
   pred_seq = []       ## output
   pred_seq_soft = []  ## softmax of output
   for i in range(n_steps):
       #pred_seq.append(tf.matmul(pred[i], weights['out'+str(i)])+biases['out'+str(i)])
       pred_seq.append(pred[i]) ## s2s
       if i == 0:
          cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:]) * tf.ones([1,1], tf.float32)*w[i]
          #cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:]) * tf.ones([1,1], tf.float32)*w[i]
          pred_seq_soft.append(tf.nn.sigmoid(pred_seq[i]))
          #pred_seq_soft.append(tf.nn.softmax(pred_seq[i]))
       else:
          cost += tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:]) * tf.ones([1,1], tf.float32)*w[i]
          #cost += tf.nn.softmax_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:]) * tf.ones([1,1], tf.float32)*w[i]
          pred_seq_soft.append(tf.nn.sigmoid(pred_seq[i]))
          #pred_seq_soft.append(tf.nn.softmax(pred_seq[i]))
   cost = tf.reduce_mean(cost)
   if args.opt == "adam":
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
      optimizer_rnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=var_names_rnn)
      optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=var_names_cnn)
   elif args.opt == "sgd":
      optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)
      optimizer_rnn = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost,var_list=var_names_rnn)
      optimizer_cnn = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost,var_list=var_names_cnn)
   else:
      print("Choose proper Optimizer!!")
      sys.exit()
   # Evaluate model
   correct_preds = []
   accuracys = []
   for i in range(n_steps):
       correct_preds.append(tf.equal(tf.round(tf.nn.sigmoid(pred_seq[i])), y_encoded[:,i,:])) # s2s        
       #correct_preds.append(tf.equal(tf.argmax(tf.nn.softmax(pred_seq[i]),1), tf.argmax(y_encoded[:,i,:],1))) # s2s
       accuracys.append(tf.reduce_mean(tf.cast(correct_preds[i], tf.float32)))
   
   #### Inference##
##### beam search ################################################################################################
   def beam(sample,k,pred_seq,y_path,y_node,n_classes,node_added):
       cnn_path_ = {
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
       # define index at each label ############################### for ilsvrc65 s2s
       level_1 = range(0,2)
       level_2 = range(2,6)
       level_3 = range(6,30)
       level_1_ = range(1,3)
       level_2_ = range(3,7)
       level_3_ = range(7,31)
       pred_n = n_classes+node_added  ## 30 + 0
       # create joint prob at each level             # idx 42 ~ 46 = depth 3's END
       pred_node = np.zeros((pred_n,), dtype=np.float32) 
       pred_node[level_1] = pred_seq[0][sample,level_1_]
       pred_node[level_2] = pred_seq[1][sample,level_2_]    
       pred_node[level_3] = pred_seq[2][sample,level_3_]

       # calculate node accuracy 
       indices = np.argwhere(y_node[sample]>0) 
       act_gather = y_node[sample,indices]
       prd_gather = np.round(pred_node[indices])
       correct_pred_node = np.equal(prd_gather,act_gather)
       correct_count_node = np.sum(correct_pred_node)
       node_count = len(indices)
        
       # calculate path accuracy 
       indices = np.argwhere(y_path[sample]>0)
       correct_count, path_count = 0, 0
       for temp in indices:
           path_idx   = cnn_path_[str(temp[0])]    
           act_gather = y_node[sample,path_idx].astype(np.float32)
           prd_gather = np.round(pred_node[path_idx]) 
           correct_pred_path = np.equal(prd_gather,act_gather)
           if np.sum(correct_pred_path) == len(path_idx):
              correct_count += 1 
           path_count += 1
       return path_count, correct_count, node_count, correct_count_node #pred_node

else:
   # kl of Bernoulli distribution
   # to be removed
   print("No more use!!")
   sys.exit()

# save final label for heatmap
#def save_prediction(train_log_dir,y_prd,y_true,epoch,step):
def save_prediction(infer_dir,inference,epoch):
    outfile_name = infer_dir + "infer_" + str(epoch) + ".npy"
    np.save(outfile_name,inference)
    return

# save predicted_seq and true path
def save_pred_seq(infer_dir,seq_soft,batch_y_val,batch_y_val_node,batch_idx):     #### save pred, original, node  
    infer_node_dir = infer_dir + 'node/'
    if not os.path.exists(infer_node_dir):
           os.makedirs(infer_node_dir)
    outfile_name_prd = infer_node_dir + "prd_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_prd,seq_soft)
    outfile_name_act = infer_node_dir + "act_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act,batch_y_val)
    outfile_name_act_node = infer_node_dir + "act_node_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act_node,batch_y_val_node)
    return

# save prediction and true label
def save_pred_resnet(infer_dir,pred_soft,batch_y_val,batch_y_val_node,batch_idx): #### save pred, original, node  
    infer_node_dir = infer_dir + 'node/'
    if not os.path.exists(infer_node_dir):
           os.makedirs(infer_node_dir)
    outfile_name_prd = infer_node_dir + "prd_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_prd,pred_soft)
    outfile_name_act = infer_node_dir + "act_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act,batch_y_val)
    outfile_name_act_node = infer_node_dir + "act_node_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act_node,batch_y_val_node)    
    return

def save_pred_hc(pred_seq,n_classes,node_added,batch_size):
    # define index at each label ############################### for ilsvrc65 s2s
    level_1 = range(0,2)
    level_2 = range(2,6)
    level_3 = range(6,30)
    level_1_ = range(1,3)
    level_2_ = range(3,7)
    level_3_ = range(7,31)
    pred_n = n_classes+node_added  ## 30 + 0
    # create joint prob at each level             # idx 42 ~ 46 = depth 3's END
    pred_node = np.zeros((batch_size,pred_n), dtype=np.float32) 
    pred_node[:,level_1] = pred_seq[0][:,level_1_]
    pred_node[:,level_2] = pred_seq[1][:,level_2_]    
    pred_node[:,level_3] = pred_seq[2][:,level_3_]
    return pred_node

# Initializing the variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=10)

# Load data
train_filenames = sorted(glob.glob(input_dir))
val_filenames = sorted(glob.glob(input_dir_val))
te_filenames = sorted(glob.glob(input_dir_te))

###############################################################################
####### hc-s2s

# Configuration of the session
#session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

if args.model == "hc":
# Launch the graph for hc
  with tf.Session(config=config) as sess:
    if args.base == "res" and args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
              get_init_fn(args.base))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess)
       init_epoch = 0
    elif args.base == "vgg" and args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
              get_init_fn(args.base))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess)
       init_epoch = 0
    else:
       # if args.restore.split(".")[3].split('_')[-3].split('/')[-1] == args.model: ## dd
       if args.restore.split(".")[0].split('_')[-3].split('/')[-1] == args.model: ## celje
          print("Restore from...",args.restore)
          # initialize sess
          sess.run(init)
          # restore model
          saver.restore(sess,args.restore)
          try:
              #init_epoch = int(args.restore.split(".")[3].split('_')[-1]) + 1  ## dd
              init_epoch = int(args.restore.split(".")[0].split('_')[-1]) + 1  ## celje, lj
          except ValueError:
              print ('Model loading falied!!')
              sys.exit()
       else:
          print("New training")
          # Define init_fn
          init_fn = slim.assign_from_checkpoint_fn(
                 args.restore,
                 get_init_fn(args.base))
          # initialize sess
          sess.run(init)
          # Call init_fn
          init_fn(sess)
          init_epoch = 0
    # Load label Train and val
    #### oi 
    batch_Y     = np.load(label_dir)     ##### s2s
    batch_Y_val = np.load(label_dir_val) ##### s2s
    batch_Y_te  = np.load(label_dir_te)  ##### s2s  
    #### ilsvrc65
    #NN = 57 + 7  ## s2s
    #batch_Y = np.eye(NN)[np.load(label_dir)] # n_classes+node_added)[np.load(label_dir)]             ## s2s
    #if args.base == "vgg":
    #   batch_Y = np.r_[batch_Y,batch_Y]
    #batch_Y_val = np.eye(NN)[np.load(label_dir_val)] # n_classes+node_added)[np.load(label_dir_val)] ## s2s
    #batch_Y_te = np.eye(NN)[np.load(label_dir_te)] # n_classes+node_added)[np.load(label_dir_val)]   ## s2s
    #### cifar100
    #batch_Y = np.eye(n_classes+node_added)[tr_y[:,0]+node_added]
    #batch_Y_val = np.eye(n_classes+node_added)[val_y[:,0]+node_added]
    #batch_Y_te = np.eye(n_classes+node_added)[te_y[:,0]+node_added]
    #### allstate
    #NN = 18  ## s2s
    #batch_Y = np.eye(NN)[np.load(label_dir)]
    #batch_Y_val = np.eye(NN)[np.load(label_dir_val)]
    #batch_Y_te = np.eye(NN)[np.load(label_dir_te)]
    epoch = init_epoch
    tr_now = "RNN"
    while (epoch < n_epoch):
        # Print out time and current learnining rate
        print(datetime.now())
        print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(train_filenames), len(val_filenames)))
        # only for inference
        def infer_test():
           print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(te_filenames), len(te_filenames)))
           if args.server == 'nu':
              infer_dir = save_dir + "code-tf/hc/tmp/hc_finetuned/" + args.idx + "-infer-ep-" + str(epoch-1) + "_test/"
           else:
              infer_dir = save_dir + "code/hc-tf/hc/tmp/hc_finetuned/" + args.idx + "-infer-ep-" + str(epoch-1) + "_test/"
           if not os.path.exists(infer_dir):
              os.makedirs(infer_dir)
           # test
           for batch_idx in range(len(te_filenames)):
               batch_x_te = preprocessing(te_filenames[batch_idx], args.base, args.server)
               batch_y_te_ = batch_Y_te[batch_idx*len(batch_x_te):(batch_idx+1)*len(batch_x_te)]
               batch_y_path_te, batch_y_te, batch_y_en_te = encode_s2s(pre_path,batch_y_te_,len(batch_x_te),n_steps,n_classes,node_added)
               repeat = 0
               print("....Test...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(te_filenames)-1))
               for step in range(0,int(len(batch_x_te)/batch_size)):
                   batch_x_temp_te = batch_x_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_temp_te = batch_y_en_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_node_temp_te = batch_y_te[(step*batch_size):(step+1)*batch_size]                    
                   batch_y_path_temp_te = batch_y_path_te[(step*batch_size):(step+1)*batch_size]
                   # accuracy at each layer out of n_classes + node_added
                   acc = []
                   for t in range(n_steps):
                       acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_te, y_encoded: batch_y_temp_te}))
                   # accuracy of 18 paths
                   loss, pred_s = sess.run([cost, pred_seq_soft], feed_dict={x: batch_x_temp_te, y_encoded: batch_y_temp_te})
                   p_count, c_count, n_count, c_count_node = 0,0,0,0 
                   for sample in range(batch_size):
                       p_count_temp, c_count_temp, n_count_temp, c_count_node_temp = beam(sample,beam_k,pred_s,batch_y_path_temp_te,batch_y_node_temp_te,n_classes,node_added)
                       p_count += p_count_temp
                       c_count += c_count_temp
                       n_count += n_count_temp
                       c_count_node += c_count_node_temp                            
                   # calculate path and node accuracy 
                   acc_p = float(c_count/p_count)
                   node_acc = float(c_count_node/n_count)
                   ## save pred_s for each minibatch
                   if step == 0:
                      seq_soft = save_pred_hc(pred_s,n_classes,node_added,batch_size)
                   else:
                      seq_soft = np.r_[seq_soft,save_pred_hc(pred_s,n_classes,node_added,batch_size)]
                   pred_s = None     
                   ''' 
                   ## save beam acc
                   temp = np.c_[np.argmax(np.eye(n_classes+node_added)[final_label].astype(np.float32),1),np.argmax(batch_y_path_temp_te,1)]
                   if batch_idx == 0:
                      inference = temp
                   else:
                      inference = np.r_[inference,temp]
                   '''   
                   print("Test ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                          .format(acc_p*100,0*100,0*100,node_acc*100,loss))
                   '''
                   ## save pred_s for each minibatch
                   if step == 0:
                      seq_soft_0 = np.expand_dims(pred_s[0],axis=1)
                      seq_soft_1 = np.expand_dims(pred_s[1],axis=1)
                      seq_soft_2 = np.expand_dims(pred_s[2],axis=1)
                      seq_soft = np.r_['1',seq_soft_0,seq_soft_1]
                      seq_soft = np.r_['1',seq_soft,seq_soft_2]
                   else:
                      seq_soft_0 = np.expand_dims(pred_s[0],axis=1)
                      seq_soft_1 = np.expand_dims(pred_s[1],axis=1)
                      seq_soft_2 = np.expand_dims(pred_s[2],axis=1)
                      seq_soft_temp = np.r_['1',seq_soft_0,seq_soft_1]
                      seq_soft_temp = np.r_['1',seq_soft_temp,seq_soft_2]
                      seq_soft = np.r_[seq_soft,seq_soft_temp]
                   '''
               ## save pred_seq
               save_pred_seq(infer_dir,seq_soft,batch_y_path_te,batch_y_te,batch_idx)
           #save_prediction(infer_dir,inference,epoch-1)
           print("Inference is done!")
           create_checks(infer_dir) 
           sys.exit()

        # only for inference
        if args.mode == "infer":
           infer_test()

########################################################################################################################
        # Alternatin: Train and validation
        # decide switch
        if epoch > args.alt1:
           tr_now = "entire"
        elif (epoch % args.alt2 == 0) and (epoch != 0):
           if tr_now == "RNN":
              tr_now = "CNN"
           else:
              tr_now = "RNN"
        # Training
        print("Training ..epoch {} parts.. {}".format(epoch, tr_now))
        # RNN train for one epoch
        for batch_idx in range(len(train_filenames)):
            batch_x = preprocessing(train_filenames[batch_idx], args.base, args.server)
            batch_y_ = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]
            batch_y_path, batch_y, batch_y_en = encode_s2s(pre_path,batch_y_,len(batch_x),n_steps,n_classes,node_added)
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(train_filenames)-1))
                for step in range(0,int(len(batch_x)/batch_size)):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y_en[(step*batch_size):(step+1)*batch_size]   ## path: Start, 0 ~ 63, END
                    batch_y_node_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    batch_y_path_temp = batch_y_path[(step*batch_size):(step+1)*batch_size] ## this is final label: 0 ~ 63
                    if tr_now == "RNN":
                       sess.run(optimizer_rnn, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    elif tr_now == "CNN":
                       sess.run(optimizer_cnn, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    else:
                       sess.run(optimizer, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    if args.prt == "full":
                       # accuracy at each layer out of n_classes + node_added
                       acc = []
                       for t in range(n_steps):
                           acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp, y_encoded: batch_y_temp}))
                       # accuracy of 18 paths
                       loss, pred_s = sess.run([cost, pred_seq_soft], feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})             
                       p_count, c_count, n_count, c_count_node = 0,0,0,0 
                       for sample in range(batch_size):
                           p_count_temp, c_count_temp, n_count_temp, c_count_node_temp = beam(sample,beam_k,pred_s,batch_y_path_temp,batch_y_node_temp,n_classes,node_added)
                           p_count += p_count_temp
                           c_count += c_count_temp
                           n_count += n_count_temp
                           c_count_node += c_count_node_temp                            
                       # calculate path and node accuracy 
                       acc_p = float(c_count/p_count) 
                       node_acc = float(c_count_node/n_count)                   
                       pred_s = None
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(acc_p*100,0*100,0*100,node_acc*100,loss))
                    else:
                       acc = [0,0,0]
                       beam_acc = 0
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(acc_p*100,acc[0]*100,acc[1]*100,node_acc*100,loss))
                repeat+=1
        print(datetime.now())
        # validation
        for batch_idx in range(len(val_filenames)):
            batch_x_val = preprocessing(val_filenames[batch_idx], args.base, args.server)
            batch_y_val_ = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]
            batch_y_path_val, batch_y_val, batch_y_en_val = encode_s2s(pre_path,batch_y_val_,len(batch_x_val),n_steps,n_classes,node_added)
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_filenames)-1))
            for step in range(0,int(len(batch_x_val)/batch_size)):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_en_val[(step*batch_size):(step+1)*batch_size]
                batch_y_node_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                batch_y_path_temp_val = batch_y_path_val[(step*batch_size):(step+1)*batch_size]
                # accuracy at each layer out of n_classes + node_added
                acc = []
                for t in range(n_steps):
                    acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val}))
                # accuracy of 18 paths
                loss, pred_s = sess.run([cost, pred_seq_soft], feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                p_count, c_count, n_count, c_count_node = 0,0,0,0 
                for sample in range(batch_size):
                    p_count_temp, c_count_temp, n_count_temp, c_count_node_temp = beam(sample,beam_k,pred_s,batch_y_path_temp_val,batch_y_node_temp_val,n_classes,node_added)
                    p_count += p_count_temp
                    c_count += c_count_temp
                    n_count += n_count_temp
                    c_count_node += c_count_node_temp                            
                # calculate path and node accuracy 
                acc_p = float(c_count/p_count) 
                node_acc = float(c_count_node/n_count)                   
                pred_s = None
                if args.mode == "infer-tr":
                   save_prediction(train_log_dir,np.argmax(final_label,1),np.argmax(batch_y_path_temp_val,1),epoch,step)
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(acc_p*100,0*100,0*100,node_acc*100,loss))
        # Save the variables to disk.
        out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
        save_path = saver.save(sess, out_file)
        print("Model saved in file: %s" % save_path)
        epoch+=1

        ## inference on test
        if epoch == n_epoch:
           infer_test()

##########################################################################################################
##### resnet-50 or vgg-16
else:
# Launch the graph for resnet-50 or vgg-16
  with tf.Session() as sess:
    if args.model == "resnet-50" and args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
              get_init_fn(args.base))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess)
       init_epoch = 0
    elif args.model == "vgg-16" and args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
              get_init_fn(args.base))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess)
       init_epoch = 0
    else:
       ## if args.restore.split(".")[3].split('_')[-3].split('/')[-1] == args.model: ## dd
       if args.restore.split(".")[0].split('_')[-3].split('/')[-1] == args.model: ## celje
          print("Restore from...",args.restore)
          # initialize sess
          sess.run(init)
          # restore model
          saver.restore(sess,args.restore)
          try:
              ## init_epoch = int(args.restore.split(".")[3].split('_')[-1]) + 1  ## dd
              init_epoch = int(args.restore.split(".")[0].split('_')[-1]) + 1  ## celje, lj
          except ValueError:
              print ('Model loading falied!!')
              sys.exit()
       else:
          print("New training")
          # Define init_fn
          init_fn = slim.assign_from_checkpoint_fn(
                 args.restore,
                 get_init_fn(args.base))
          # initialize sess
          sess.run(init)
          # Call init_fn
          init_fn(sess)
          init_epoch = 0
    # Load label Train and val
    #### oi 
    batch_Y     = np.load(label_dir)     ##### s2s
    batch_Y_val = np.load(label_dir_val) ##### s2s
    batch_Y_te  = np.load(label_dir_te)  ##### s2s                            
    #### ilsvrc65
    #batch_Y = np.eye(57+node_added)[np.load(label_dir)-resnet_label_scale]          ##### s2s
    #if args.model == "vgg-16":
    #   batch_Y = np.r_[batch_Y,batch_Y]
    #batch_Y_val = np.eye(57+node_added)[np.load(label_dir_val)-resnet_label_scale]  ##### s2s
    #batch_Y_te = np.eye(57+node_added)[np.load(label_dir_te)-resnet_label_scale]    ##### s2s
    #### cifar100
    #batch_Y = np.eye(n_classes)[tr_y[:,0]]
    #batch_Y_val = np.eye(n_classes)[val_y[:,0]]
    #batch_Y_te = np.eye(n_classes)[te_y[:,0]]
    ## Allstate
    #NN = 18  ## s2s
    #batch_Y = np.eye(NN)[np.load(label_dir)]
    #batch_Y_val = np.eye(NN)[np.load(label_dir_val)]
    #batch_Y_te = np.eye(NN)[np.load(label_dir_te)]
    epoch = init_epoch
    while (epoch < n_epoch):
        print(datetime.now())
        print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(train_filenames), len(val_filenames)))
        # only for inference
        def infer_test_():
           print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(te_filenames), len(te_filenames)))
           if args.server == 'nu':
              infer_dir = save_dir + "code-tf/hc/tmp/cnn_finetuned/" + args.idx + "-infer-ep-" + str(epoch-1) + "_test/"
           else:
              infer_dir = save_dir + "code/hc-tf/hc/tmp/cnn_finetuned/" + args.idx + "-infer-ep-" + str(epoch-1) + "_test/"
           if not os.path.exists(infer_dir):
              os.makedirs(infer_dir)
           print(datetime.now())
           for batch_idx in range(len(te_filenames)):
               batch_x_te  = preprocessing(te_filenames[batch_idx], args.base, args.server)
               batch_y_te_ = batch_Y_te[batch_idx*len(batch_x_te):(batch_idx+1)*len(batch_x_te)]                               ## s2s
               batch_y_path_te, batch_y_te, batch_y_target_te = encode_s2s(pre_path,batch_y_te_,len(batch_x_te),n_steps,n_classes,node_added)   ## s2s
               batch_y_target_te = None                                                                                        ## s2s
               repeat = 0
               print("....Test...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(te_filenames)-1))
               for step in range(0,int(len(batch_x_te)/batch_size)):
                   batch_x_temp_te = batch_x_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_temp_te = batch_y_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_path_temp_te = batch_y_path_te[(step*batch_size):(step+1)*batch_size]
                   loss, acc, pred_s = sess.run([cost, accuracy, pred_cnn], feed_dict={x: batch_x_temp_te, y: batch_y_temp_te})
                   p_count, c_count = 0, 0
                   for sample in range(batch_size):
                       p_count_temp, c_count_temp = beam_cnn(sample,pred_s,batch_y_path_temp_te,batch_y_temp_te,n_classes)
                       p_count += p_count_temp
                       c_count += c_count_temp
                   acc_p = float(c_count/p_count)
                   #pred_temp = sess.run(pred, feed_dict={x: batch_x_temp_te, y: batch_y_temp_te})
                   print("Test ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                          .format(acc_p*100,0*100,0*100,acc*100,loss))
                   ## save predcitions for each minibatch
                   if step == 0:
                      pred_soft = pred_s
                   else:
                      pred_soft = np.r_[pred_soft,pred_s]
               ## save pred_seq
               save_pred_resnet(infer_dir,pred_soft,batch_y_path_te,batch_y_te,batch_idx)
           print("Inference is done!")
           create_checks(infer_dir)         
           sys.exit()

        # only for inference
        if args.mode == "infer":
           infer_test_()

###########################################################################################

        ## train and validation
        for batch_idx in range(len(train_filenames)):
            batch_x = preprocessing(train_filenames[batch_idx], args.base, args.server)
            batch_y_ = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]                              ## s2s
            batch_y_path, batch_y, batch_y_target = encode_s2s(pre_path,batch_y_,len(batch_x),n_steps,n_classes,node_added)  ## s2s
            # batch_y = y_path node
            batch_y_target = None                                                                              ## s2s
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(train_filenames)-1))
                for step in range(0,int(len(batch_x)/batch_size)):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    batch_y_path_temp = batch_y_path[(step*batch_size):(step+1)*batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x_temp, y: batch_y_temp})
                    loss, acc, pred_s = sess.run([cost, accuracy, pred_cnn], feed_dict={x: batch_x_temp, y: batch_y_temp})
                    p_count, c_count = 0, 0
                    for sample in range(batch_size):
                        p_count_temp, c_count_temp = beam_cnn(sample,pred_s,batch_y_path_temp,batch_y_temp,n_classes)
                        p_count += p_count_temp
                        c_count += c_count_temp
                    acc_p = float(c_count/p_count)
                    print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(acc_p*100,0*100,0*100,acc*100,loss))
                repeat+=1
        # validation
        print(datetime.now())
        for batch_idx in range(len(val_filenames)):
            batch_x_val = preprocessing(val_filenames[batch_idx], args.base, args.server)
            batch_y_val_ = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]                               ## s2s
            batch_y_path_val, batch_y_val, batch_y_target_val = encode_s2s(pre_path,batch_y_val_,len(batch_x_val),n_steps,n_classes,node_added)   ## s2s
            batch_y_target_val = None                                                                                           ## s2s
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_filenames)-1))
            for step in range(0,int(len(batch_x_val)/batch_size)):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                batch_y_path_temp_val = batch_y_path_val[(step*batch_size):(step+1)*batch_size]
                loss, acc, pred_s = sess.run([cost, accuracy, pred_cnn], feed_dict={x: batch_x_temp_val, y: batch_y_temp_val})
                p_count, c_count = 0, 0
                for sample in range(batch_size):
                    p_count_temp, c_count_temp = beam_cnn(sample,pred_s,batch_y_path_temp_val,batch_y_temp_val,n_classes)
                    p_count += p_count_temp
                    c_count += c_count_temp
                acc_p = float(c_count/p_count)
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(acc_p*100,0*100,0*100,acc*100,loss))
        # Save the variables to disk.
        out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
        save_path = saver.save(sess, out_file)
        print("Model saved in file: %s" % save_path)
        epoch+=1

        ## inference on test for cifar100
        if epoch == n_epoch:
           infer_test_()




