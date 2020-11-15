'''
ref 1: bidirectional_rnn.py
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
ref 2: slim resnet
https://github.com/tensorflow/models/tree/master/slim
https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py

update
# Replace RNN to seq2seq
# 1120: train on slim-preprocessing
# 1121: Clean script
# 1124: freeze/unfreeze coded and tested
# 1127: add training parts
'''

from __future__ import print_function

import os, sys, argparse, glob
import hickle as hkl
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import rnn
# Import slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import cvgg as vgg
# imoport cvgg
#cvgg_path = '/home/jkoo/.conda/envs/tf1/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/'
#sys.path.append(cvgg_path)
#import cvgg as vgg

# Define options
parser = argparse.ArgumentParser()
parser.add_argument("-server","-server", dest='server',type=str,
                    help="Select which server", default="allstate")
parser.add_argument("-model","-model_name", dest='model',type=str,
                    help="Select model option", default="resnet-50")
parser.add_argument("-index","-idx", dest='idx',type=str,
                    help="Select idx to save checkpoint", default=None)
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
                    help="Select restoring model", default=None)
parser.add_argument("-print","-prt", dest='prt',type=str,
                    help="Select printing option", default=None)
parser.add_argument("-mode","-mode", dest='mode',type=str,
                    help="Select printing option", default="train")
parser.add_argument("-weight","-weight", dest='weight',type=str,
                    help="Select tree layer weights option", default="1;1;1")
args = parser.parse_args()

# print arg_options
if args.model == "vgg-16":
   print( "Server: {}, Model Name: {}, Index: {}, N_epoch: {}, Learning rate: {}, Batch size: {}, Optimizer: {}".format(args.server, args.model, args.idx, args.epoch, args.lr, args.batch, args.opt))
elif args.model == "hc1":
   print( "Server: {}, Model Name: {}, Index: {}, N_epoch: {}, Learning rate: {}, Batch size: {}, Optimizer: {}, RNN Input Dimension: {}, RNN hidden state: {}, Conversion Option: {}, Layer Averaging Option: {}, Beam k: {}".format(
        args.server, args.model, args.idx, args.epoch, args.lr, args.batch, args.opt, args.input, args.hidden, args.option1, args.option2, args.beam))
else:
  print("Wrong model choice!!")
  sys.exit()

# Load check point
if args.server == "nu":
   home_dir = "/home/lab.analytics.northwestern.edu/jkoo/" ## deepdish
   #home_dir = "/home/jkoo/"                                 ## lj and celje
   data_dir = "/home/public/"                              ## deepdish
   save_dir = "/home/lab.analytics.northwestern.edu/jkoo/" ## deepdish
   #data_dir = "/scratch/jkoo/data/"                         ## lj
   #save_dir = "/scratch/jkoo/"                              ## lj
   #data_dir = "/mnt/jkoo/"                                 ## celje
   #save_dir = "/mnt/jkoo/"                                 ## celje
   checkpoints_dir = home_dir + "code-tf/weights/"
   if args.model == "hc1":
      train_log_dir = save_dir + "code-tf/hc/tmp/hc_finetuned/" + args.idx + "/"
   else:
      train_log_dir = save_dir + "code-tf/hc/tmp/resnet_finetuned/" + args.idx + "/"
   if not os.path.exists(train_log_dir):
       os.makedirs(train_log_dir)
   # imoport cvgg
   cvgg_path = home_dir + '.conda/envs/tf1/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/'
   sys.path.append(cvgg_path)
   import cvggv2 as vgg
   #input_dir = data_dir + "ilsvrc65/arrays/train_hkl_b285_b_285-slim_1/*"
   #label_dir = data_dir + "ilsvrc65/arrays/labels/ilsvrc65.train.name.label.npy"
   #input_dir_val = data_dir + "ilsvrc65/arrays/val_hkl_b285_b_285-slim/*"
   #label_dir_val = data_dir + "ilsvrc65/arrays/labels/ilsvrc65.val.name.label.npy"
   #input_dir_te = data_dir + "ilsvrc65/arrays/test_hkl_b285_b_285/*"
   #label_dir_te = data_dir + "ilsvrc65/arrays/labels/ilsvrc65.test.name.label.npy"
   #input_dir = data_dir + "ilsvrc12-hkl-sub/train_hkl_b256_b_256/*"
   #label_dir = data_dir + "ilsvrc12-hkl-sub/labels/train_labels.npy"
   #input_dir_val = data_dir + "ilsvrc12-hkl-sub/val_hkl_b256_b_256/*"
   #label_dir_val = data_dir + "ilsvrc12-hkl-sub/labels/val_labels.npy"
else:
   home_dir = "/data/jkooa/"
   checkpoints_dir = home_dir + "Git/tf/slim/weights/"
   checkpoints_dir_trained = home_dir +  "code/hc-tf/resnet/tmp/resnet_finetuned/resnet-50_epoch_9.ckpt"
   if args.model == "hc1":
      train_log_dir = home_dir + "code/hc-tf/resnet/tmp/hc_finetuned/" + args.idx + "/"
   else:
      train_log_dir = home_dir + "code/hc-tf/resnet/tmp/resnet_finetuned/" + args.idx + "/"
   if not os.path.exists(train_log_dir):
       os.makedirs(train_log_dir)
   input_dir = home_dir + "preprocessing-hc/labeled_set/train_hkl_b256_b_256/*"
   label_dir = home_dir + "preprocessing-hc/labeled_set/labels/crop_hc_tr_0-skew_label.npy"
   input_dir_val = home_dir + "preprocessing-hc/labeled_set/val_hkl_b256_b_256/*"
   label_dir_val = home_dir + "preprocessing-hc/labeled_set/labels/crop_hc_te_0_label.npy"
   input_dir_te = home_dir + "preprocessing-hc/labeled_set/test_hkl_b256_b_256/*"
   label_dir_te = home_dir + "preprocessing-hc/labeled_set/labels/crop_hc_test_1_label.npy"

def get_init_fn(server):
    """Returns a function run by the chief worker to warm-start the training."""
    if server == "allstate":
       checkpoint_exclude_scopes= ["resnet_v1_50/logits", "predictions"]
       checkpoint_exclude_scopes= ['vgg_16/fc6','vgg_16/fc7','vgg_16/fc8']
    elif server == "nu":
       # checkpoint_exclude_scopes= ["resnet_v1_50/logits", "predictions"]
       checkpoint_exclude_scopes= ['vgg_16/fc6','vgg_16/fc7','vgg_16/fc8']
    else:
       checkpoint_exclude_scopes= [] #["resnet_v1_50/logits", "predictions"]
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
    print("...Loading check point...", 'vgg_16.ckpt')
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

# Parameters
learning_rate = args.lr
batch_size = args.batch
n_epoch = args.epoch
n_repeat = 1

# Network Parameters
n_height = 32                        # Resized image height
n_width = 32                         # Resized image width
n_channels = 3                        # Resized image channel
n_input = n_height*n_width*n_channels # Imagenet data input (img shape: 224*224)
n_steps = len(avg_option)             # timesteps

# how many parent nodes
if args.server == "nu":
   node_added = 20                     # cifar100
   # node_added = 7                     # ilsvrc65
else:
   node_added = 10
# how many leaf nodes
if args.server == "nu":
   n_classes = 100                    # cifar100
   #n_classes = 57                     # ilsvrc65
   #n_classes = 1000                  # ilsvrc10k
   resnet_label_scale = 7             # ilsvrc65
else:
   n_classes = 18

mean_r = 129.30417
mean_g = 124.06996
mean_b = 112.43405

std_r = 68.14695
std_g = 65.37863
std_b = 70.40022

x = tf.placeholder("float", [None, n_height, n_width, n_channels])
y = tf.placeholder("float", [None, n_classes])
y_encoded = tf.placeholder("float", [None, n_steps, n_classes+node_added])

if model_name == "hc1" or "resnet-50":
   # Define dimension of 4 blocks
   bl1_dim = [28,28,256] #[bls['bl1'].get_shape().as_list()[1], bls['bl1'].get_shape().as_list()[2],bls['bl1'].get_shape().as_list()[3]]
   bl2_dim = [14,14,512] #[bls['bl2'].get_shape().as_list()[1], bls['bl2'].get_shape().as_list()[2],bls['bl2'].get_shape().as_list()[3]]
   bl3_dim = [7,7,1024]  #[bls['bl3'].get_shape().as_list()[1], bls['bl3'].get_shape().as_list()[2],bls['bl3'].get_shape().as_list()[3]]
   bl4_dim = [7,7,2048]  #[bls['bl4'].get_shape().as_list()[1], bls['bl4'].get_shape().as_list()[2],bls['bl4'].get_shape().as_list()[3]]
   # Define polling conversion dimension
   l1_scale = 4096 #l1.get_shape().as_list()[1]*l1.get_shape().as_list()[2]*l1.get_shape().as_list()[3] # 4096
   l2_scale = 8192 #l2.get_shape().as_list()[1]*l2.get_shape().as_list()[2]*l2.get_shape().as_list()[3] # 8192
   l3_scale = 4096 #l3.get_shape().as_list()[1]*l3.get_shape().as_list()[2]*l3.get_shape().as_list()[3] # 4096
   l4_scale = 8192 #l4.get_shape().as_list()[1]*l4.get_shape().as_list()[2]*l4.get_shape().as_list()[3] # 8192
elif model_name == "hc1" or "vgg-16":
   # Define dimension of 4 blocks (VGG)
   bl1_dim = [14,14,512] # 'vgg_16/conv5/conv5_1'  #[bls['bl1'].get_shape().as_list()[1], bls['bl1'].get_shape().as_list()[2],bls['bl1'].get_shape().as_list()[3]]
   bl2_dim = [14,14,512] # 'vgg_16/conv5/conv5_2'  #[bls['bl2'].get_shape().as_list()[1], bls['bl2'].get_shape().as_list()[2],bls['bl2'].get_shape().as_list()[3]]
   bl3_dim = [14,14,512] # 'vgg_16/conv5/conv5_3'  #[bls['bl3'].get_shape().as_list()[1], bls['bl3'].get_shape().as_list()[2],bls['bl3'].get_shape().as_list()[3]]
   bl4_dim = [7,7,512]   # 'vgg_16/pool5'          #[bls['bl4'].get_shape().as_list()[1], bls['bl4'].get_shape().as_list()[2],bls['bl4'].get_shape().as_list()[3]]
   # Define polling conversion dimension (VGG)
   l1_scale = 2048 #l1.get_shape().as_list()[1]*l1.get_shape().as_list()[2]*l1.get_shape().as_list()[3] # 4096
   l2_scale = 2048 #l2.get_shape().as_list()[1]*l2.get_shape().as_list()[2]*l2.get_shape().as_list()[3] # 8192
   l3_scale = 2048 #l3.get_shape().as_list()[1]*l3.get_shape().as_list()[2]*l3.get_shape().as_list()[3] # 4096
   l4_scale = 2048 #l4.get_shape().as_list()[1]*l4.get_shape().as_list()[2]*l4.get_shape().as_list()[3] # 8192
else:
   print ('wrong model!')
   sys.exit()

with tf.name_scope('HC_CONV'):
 weights = {
    # 5x5 conv1, 1 input, 32 outputs
    #'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv2, 32 inputs, 64 outputs
    #'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 2x2 conv3, 64 inputs, 128 outputs
    #'wc3': tf.Variable(tf.random_normal([2, 2, 64, 128])),
    # 2x2 conv4, 128 inputs, 256 outputs
    #'wc4': tf.Variable(tf.random_normal([2, 2, 128, 256])),

    # rnn input step 1, from block 1
    'wd1': tf.Variable(tf.random_normal([bl1_dim[0]*bl1_dim[1]*bl1_dim[2], rnn_dim])),
    # rnn input step 2, from block 2
    'wd2': tf.Variable(tf.random_normal([bl2_dim[0]*bl2_dim[1]*bl2_dim[2], rnn_dim])),
    # rnn input step 3, from block 3
    'wd3': tf.Variable(tf.random_normal([bl3_dim[0]*bl3_dim[1]*bl3_dim[2], rnn_dim])),
    # rnn input step 4, from block 4
    'wd4': tf.Variable(tf.random_normal([bl4_dim[0]*bl4_dim[1]*bl4_dim[2], rnn_dim])),

    # For Pooling conversion  option=3.1 & 3.2 for k = filter size = stride = 2
    # block1: 28 * 28 * 256 = 200704 -> 14 * 14 * 256 = 50176 -> 7 * 7 * 256 = 12544 -> 4 * 4 * 256 = 4096
    'wd1p': tf.Variable(tf.random_normal([l1_scale, rnn_dim])),
    # block2: 14 * 14 * 512 = 100352 -> 7 * 7 * 512 = 25088 -> 4 * 4 * 512 = 8192
    'wd2p': tf.Variable(tf.random_normal([l2_scale, rnn_dim])),
    # block3: 7 * 7 * 1024 = 50176 -> 4 * 4 * 1024 = 16384 -> 2 * 2 * 1024 = 4096
    'wd3p': tf.Variable(tf.random_normal([l3_scale, rnn_dim])),
    # block4: 7 * 7 * 2048 = 100352 -> 4 * 4 * 2048 = 32768 -> 2 * 2 * 2048 = 8192
    'wd4p': tf.Variable(tf.random_normal([l4_scale, rnn_dim])),

   # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added])),
    'out0': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added])),
    'out1': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added])),
    'out2': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added])),
    'out3': tf.Variable(tf.random_normal([2*n_hidden[-1], n_classes+node_added])),
    # 14x14 rconv1, 32  input, rnn_dim outputs
    'rwc1': tf.Variable(tf.random_normal([bl1_dim[0], bl1_dim[1], bl1_dim[2], rnn_dim])),
    # 7x7 rconv2, 64 inputs, rnn_dim outputs
    'rwc2': tf.Variable(tf.random_normal([bl2_dim[0], bl2_dim[1], bl2_dim[2], rnn_dim])),
    # 4x4 rconv3, 128 inputs, rnn_dim outputs
    'rwc3': tf.Variable(tf.random_normal([bl3_dim[0], bl3_dim[1], bl3_dim[2], rnn_dim])),
    # 2x2 rconv4, 256 inputs, rnn_dim outputs
    'rwc4': tf.Variable(tf.random_normal([bl4_dim[0], bl4_dim[1], bl4_dim[2], rnn_dim]))
  }
 biases = {
    #'bc1': tf.Variable(tf.random_normal([32])),
    #'bc2': tf.Variable(tf.random_normal([64])),
    #'bc3': tf.Variable(tf.random_normal([128])),
    #'bc4': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([rnn_dim])),
    'bd2': tf.Variable(tf.random_normal([rnn_dim])),
    'bd3': tf.Variable(tf.random_normal([rnn_dim])),
    'bd4': tf.Variable(tf.random_normal([rnn_dim])),

    'out': tf.Variable(tf.random_normal([n_classes])),
    'out0': tf.Variable(tf.random_normal([n_classes+node_added])),
    'out1': tf.Variable(tf.random_normal([n_classes+node_added])),
    'out2': tf.Variable(tf.random_normal([n_classes+node_added])),
    'out3': tf.Variable(tf.random_normal([n_classes+node_added])),
    'rbc1': tf.Variable(tf.random_normal([rnn_dim])),
    'rbc2': tf.Variable(tf.random_normal([rnn_dim])),
    'rbc3': tf.Variable(tf.random_normal([rnn_dim])),
    'rbc4': tf.Variable(tf.random_normal([rnn_dim]))
  }

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

def BiRNN(x, weights, biases, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer):

    # random cropping
    #x = tf.random_crop(x,[batch_size,32,32,3])
    #x = tf.image.resize_image_with_crop_or_pad(x,224,224)
    # return resnet-50
    if model_name == "vgg-16":
       # Load resnet-50 model
       #with slim.arg_scope(resnet_v1.resnet_arg_scope()):
       #     net_, end_points_ = resnet_v1.resnet_v1_50(x, n_classes, is_training=True)
       #return net_[:,0,0,:] ## resnet-50
       # Load vgg-16 model
       with slim.arg_scope(vgg.vgg_arg_scope()):
            net_, end_points_ = vgg.vgg_16(x, n_classes, is_training=True)
            print (tf.shape(net_))
       return net_ ## vgg-16

    # return hc1
    elif model_name == "hc1":
       # Load resnet-50 model (VGG)
       #with slim.arg_scope(resnet_v1.resnet_arg_scope()):
       #     net_, end_points_ = resnet_v1.resnet_v1_50(x, n_classes+node_added, is_training=True)
       with slim.arg_scope(vgg.vgg_arg_scope()):
            net_, end_points_ = vgg.vgg_16(x, n_classes+node_added, is_training=True)
       var_names_cnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')]
       #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1'):
       for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
           var_names_cnn.append(i)
      # Define 4 blocks
       #bl1_name = # 'resnet_v1_50/conv1'
       bl1_name_ = 'vgg_16/conv5/conv5_1'# 'resnet_v1_50/block1'
       bl2_name_ = 'vgg_16/conv5/conv5_2'# 'resnet_v1_50/block2'
       bl3_name_ = 'vgg_16/conv5/conv5_3'# 'resnet_v1_50/block3'
       bl4_name_ = 'vgg_16/pool5'# 'resnet_v1_50/block4'
       blocks = {
             'bl1': end_points_[bl1_name_],
             'bl2': end_points_[bl2_name_],
             'bl3': end_points_[bl3_name_],
             'bl4': end_points_[bl4_name_]
       }
       #for d in ['/gpu:0']:
       #    with tf.device(d):
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
       if con_option == 2:
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
          l1 = tf.reshape(l1, [-1,weights['wd1'].get_shape().as_list()[1]])
          l2 = tf.reshape(l2, [-1,weights['wd2'].get_shape().as_list()[1]])
          l3 = tf.reshape(l3, [-1,weights['wd3'].get_shape().as_list()[1]])
          l4 = tf.reshape(l4, [-1,weights['wd4'].get_shape().as_list()[1]])
          #      l1_0.append(tf.reshape(l1_00, [-1,weights['wd1'].get_shape().as_list()[1]]))
          #      l2_0.append(tf.reshape(l2_00, [-1,weights['wd2'].get_shape().as_list()[1]]))
          #      l3_0.append(tf.reshape(l3_00, [-1,weights['wd3'].get_shape().as_list()[1]]))
          #      l4_0.append(tf.reshape(l4_00, [-1,weights['wd4'].get_shape().as_list()[1]]))
          #with tf.device('/cpu:0'):
          #l1 = tf.add_n(l1_0)
          #with tf.device('/cpu:1'):
          #l2 = tf.add_n(l2_0)
          #with tf.device('/cpu:2'):
          #l3 = tf.add_n(l3_0)
          #with tf.device('/cpu:3'):
          #l4 = tf.add_n(l4_0)
       if con_option == 3.1:
          print('Conversion con_option 3-1: Pooling conversion - pooling and linear')
          # 2-2. conversion avg pooled cnn layer
          def dim_calc(W,D,rnn_dim):
              #return np.max([W*np.sqrt(D/rnn_dim),1])
              return np.max([W*np.sqrt(D/float(rnn_dim)),1])
          ## block1: 28 * 28 * 256 = 200704
          l1 = maxpool2d(conv1, k=2) # 28 * 28 * 256 -> k = 2 -> 14 * 14 * 256 = 50176
          l1 = maxpool2d(l1, k=2)    # 14 * 14 * 256 -> k = 2 -> 7 * 7 * 256 = 12544
          l1 = maxpool2d(l1, k=2)    # 7 * 7 * 256 -> k = 2 -> 4 * 4 * 256 = 4096
          # block2: 14 * 14 * 512 = 100352
          l2 = maxpool2d(conv2, k=2) # 14 * 14 * 512 -> k = 2 -> 7 * 7 * 512 = 25088
          l2 = maxpool2d(l2, k=2)    # 7 * 7 * 512 -> k = 2 -> 4 * 4 * 512 = 8192
          l2 = maxpool2d(l2, k=2) #################################################################################
          # block3: 7 * 7 * 1024 = 50176
          l3 = maxpool2d(conv3, k=2) # 7 * 7 * 1024 -> k = 2 -> 4 * 4 * 1024 = 16384
          l3 = maxpool2d(l3, k=2)    # 4 * 4 * 1024 -> k = 2 -> 2 * 2 * 1024 = 4096
          l3 = maxpool2d(l3, k=2) #################################################################################
          # block4: 7 * 7 * 2048 = 100352
          l4 = maxpool2d(conv4, k=2) # 7 * 7 * 2048 -> k = 2 -> 4 * 4 * 2048 = 32768
          l4 = maxpool2d(l4, k=2)    # 4 * 4 * 2048 -> k = 2 -> 2 * 2 * 2048 = 8192
          #############l1 = tf.reshape(l1, [-1,weights['wd1p'].get_shape().as_list()[0]])
          l1 = tf.contrib.layers.flatten(l1)
          l2 = tf.contrib.layers.flatten(l2)
          l3 = tf.contrib.layers.flatten(l3)
          l4 = tf.contrib.layers.flatten(l4)
          #l1 = tf.reshape(l1, [-1,weights['wd1p'].get_shape().as_list()[0]])
          #l1 = tf.matmul(l1, weights['wd1p'])
          #l2 = tf.reshape(l2, [-1,weights['wd2p'].get_shape().as_list()[0]])
          #l2 = tf.matmul(l2, weights['wd2p'])
          #l3 = tf.reshape(l3, [-1,weights['wd3p'].get_shape().as_list()[0]])
          #l3 = tf.matmul(l3, weights['wd3p'])
          #l4 = tf.reshape(l4, [-1,weights['wd4p'].get_shape().as_list()[0]])
          #l4 = tf.matmul(l4, weights['wd4p'])

       if con_option == 3.2:
          print('Conversion con_option 3-2: Pooling conversion - pooling and p-max')
          # 2-2. conversion avg pooled cnn layer
          def dim_calc(W,D,rnn_dim):
              #return np.max([W*np.sqrt(D/rnn_dim),1])
              return np.max([W*np.sqrt(D/float(rnn_dim)),1])
              #return int(W + 1 - np.sqrt(D/float(rnn_dim)))
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
          l1 = tf.reshape(l1, [-1,l1.get_shape().as_list()[1]*l1.get_shape().as_list()[2]*l1.get_shape().as_list()[3]]) # weights['wd1'].get_shape().as_list()[1]*l1_scale])
          l1, void = tf.nn.top_k(l1,rnn_dim)
          l2 = tf.reshape(l2, [-1,l2.get_shape().as_list()[1]*l2.get_shape().as_list()[2]*l2.get_shape().as_list()[3]]) # weights['wd2'].get_shape().as_list()[1]*l2_scale])
          l2, void = tf.nn.top_k(l2,rnn_dim)
          l3 = tf.reshape(l3, [-1,l3.get_shape().as_list()[1]*l3.get_shape().as_list()[2]*l3.get_shape().as_list()[3]]) # weights['wd3'].get_shape().as_list()[1]*l3_scale])
          l3, void = tf.nn.top_k(l3,rnn_dim)
          l4 = tf.reshape(l4, [-1,l4.get_shape().as_list()[1]*l4.get_shape().as_list()[2]*l4.get_shape().as_list()[3]]) # weights['wd4'].get_shape().as_list()[1]*l4_scale])
          l4, void = tf.nn.top_k(l4,rnn_dim)
       if con_option == 4:
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
       # Averaging converted cnn layers for rnn
       rnn_input = []
       rnn_inputs = []
       rnn_inputs.append(l4)
       rnn_inputs.append(l3)
       rnn_inputs.append(l2)
       rnn_inputs.append(l1)
       for i in range(len(avg_option)):
           rnn_input.append(tf.add_n([rnn_inputs[u-1] for u in avg_option[i]])/len(avg_option[i]))

       # Define lstm cells with tensorflow
       cells_fw = []
       cells_bw = []
       for i in range(rnn_n_layer):
           #with tf.device('/gpu:'+str(i+1)):
                # Forward direction cell
           ## dropout
           #cells_fw.append(rnn.DropoutWrapper(rnn.LSTMCell(n_hidden[i], initializer=tf.orthogonal_initializer(), forget_bias=1.0),input_keep_prob=0.9,output_keep_prob=0.9,
           #                                                                                                                                            state_keep_prob=0.9))
           cells_fw.append(rnn.LSTMCell(n_hidden[i], initializer=tf.orthogonal_initializer(), forget_bias=1.0))
                # Backward direction cell
           ## dropout
           #cells_bw.append(rnn.DropoutWrapper(rnn.LSTMCell(n_hidden[i], initializer=tf.orthogonal_initializer(), forget_bias=1.0),input_keep_prob=0.9,output_keep_prob=0.9,
           #                                                                                                                                            state_keep_prob=0.9))
           cells_bw.append(rnn.LSTMCell(n_hidden[i], initializer=tf.orthogonal_initializer(), forget_bias=1.0))

       # Get lstm cell output: First layer of RNN
       #try:
       outputs, _, _ = rnn.stack_bidirectional_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=rnn_input,
                                                 dtype=tf.float32, scope='HC_RNN')
       var_names_rnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_RNN')]
       for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
           var_names_rnn.append(i)

       #except Exception: # Old TensorFlow version only returns outputs not states
       #    print("RNN part error: This should not be called!!!")
       #    sys.exit()

       # Linear activation, using rnn inner loop last output
       return outputs, var_names_cnn, var_names_rnn

    ## return error message
    else:
       print (".......Wrong model choice!!.......")
       sys.exit()
if model_name =='hc1':
   pred, var_names_cnn, var_names_rnn = BiRNN(x, weights, biases, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer)
else:
   pred = BiRNN(x, weights, biases, model_name, rnn_dim, con_option, avg_option, batch_size, n_classes, node_added, rnn_n_layer)

# encoding function
tree = {'0': [75, 92, 115, 24, 50],
        '1': [21, 93, 52, 87, 111],
       '10': [43, 91, 69, 53, 80],
       '11': [39, 51, 41, 35, 58],
       '12': [84, 95, 86, 54, 83],
       '13': [65, 97, 119, 46, 99],
       '14': [31, 118, 55, 66, 22],
       '15': [49, 98, 47, 113, 64],
       '16': [100, 94, 85, 56, 70],
       '17': [116, 79, 72, 67, 76],
       '18': [110, 28, 68, 33, 78],
       '19': [101, 109, 89, 105, 61],
        '2': [102, 90, 112, 82, 74],
        '3': [48, 29, 81, 30, 36],
        '4': [20, 71, 103, 73, 77],
        '5': [106, 59, 107, 42, 60],
        '6': [104, 40, 114, 45, 25],
        '7': [44, 34, 26, 38, 27],
        '8': [117, 62, 63, 108, 23],
        '9': [37, 96, 88, 32, 57]}

pre_path = {}
for key in tree.keys():
    temp = tree[key]
    for idx in temp:
       pre_path[str(idx)] = [int(key), idx]

def encode(pre_path,batch_y,batch_size,n_steps,n_classes,node_added):
    path = np.zeros((batch_size,n_steps,n_classes+node_added) ,dtype=int)
    for i in range(batch_size):
        label = np.argmax(batch_y[i])
        for j in range(n_steps):
            path[i,j,pre_path[str(label)][j]] = 1
    return path

def preprocessing(batch_x):
    batch_x[:,:,:,0] -= mean_r
    batch_x[:,:,:,1] -= mean_g
    batch_x[:,:,:,2] -= mean_b
    batch_x[:,:,:,0] /= std_r
    batch_x[:,:,:,1] /= std_g
    batch_x[:,:,:,2] /= std_b
    return batch_x

if args.model == "vgg-16":
   # Define loss and optimizer
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
   if args.opt == "adam":
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   elif args.opt == "sgd":
      optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)
   else:
      print("Choose proper Optimizer!!")
      sys.exit()
   # Evaluate model
   correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
   #correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred),1), tf.argmax(y,1))
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

elif args.model == "hc1": # ce
   # Define loss and optimizer
   pred_seq = []       ## output
   pred_seq_soft = []  ## softmax of output
   for i in range(n_steps):
       pred_seq.append(tf.matmul(pred[i], weights['out'+str(i)])+biases['out'+str(i)])
       if i == 0:
          cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:])* tf.ones([1,1], tf.float32)*w[i]
          pred_seq_soft.append(tf.nn.softmax(pred_seq[i]))
       else:
          cost += tf.nn.softmax_cross_entropy_with_logits(logits=pred_seq[i], labels=y_encoded[:,i,:]) * tf.ones([1,1], tf.float32)*w[i]
          pred_seq_soft.append(tf.nn.softmax(pred_seq[i]))
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
       #correct_preds.append(tf.equal(tf.argmax(pred_seq[i],1), tf.argmax(y_encoded[:,i,:],1)))
       correct_preds.append(tf.equal(tf.argmax(tf.nn.softmax(pred_seq[i]),1), tf.argmax(y_encoded[:,i,:],1)))
       accuracys.append(tf.reduce_mean(tf.cast(correct_preds[i], tf.float32)))
   #### Inference##

##### beam search ################################################################################################
   def beam(k,pred_seq,batch_size,n_classes,node_added):
       tree = {'0': [75, 92, 115, 24, 50],
               '1': [21, 93, 52, 87, 111],
              '10': [43, 91, 69, 53, 80],
              '11': [39, 51, 41, 35, 58],
              '12': [84, 95, 86, 54, 83],
              '13': [65, 97, 119, 46, 99],
              '14': [31, 118, 55, 66, 22],
              '15': [49, 98, 47, 113, 64],
              '16': [100, 94, 85, 56, 70],
              '17': [116, 79, 72, 67, 76],
              '18': [110, 28, 68, 33, 78],
              '19': [101, 109, 89, 105, 61],
               '2': [102, 90, 112, 82, 74],
               '3': [48, 29, 81, 30, 36],
               '4': [20, 71, 103, 73, 77],
               '5': [106, 59, 107, 42, 60],
               '6': [104, 40, 114, 45, 25],
               '7': [44, 34, 26, 38, 27],
               '8': [117, 62, 63, 108, 23],
               '9': [37, 96, 88, 32, 57]}
       pre_path = {}
       for key in tree.keys():
           temp = tree[key]
           for idx in temp:
              pre_path[str(idx)] = [int(key), idx]

       # define final label matrix
       pred_label = np.zeros((batch_size,n_classes+node_added),dtype=np.float32) ##########################3 for ilsvrc65
       # define index at each label ############################### for cifar100
       l1_n = 20                                    #################
       l2_n = 120        # 2 + 5                    #################
       # define joint prob at each level
       pred_l1 = np.zeros((batch_size,n_classes+node_added),dtype=np.float32)
       pred_l2 = np.zeros((batch_size,n_classes+node_added),dtype=np.float32)
       # pred_l3 = np.zeros((batch_size,n_classes+node_added),dtype=np.float32)
       # create joint prob at each level
       ## level 1
       for i in range(batch_size):
           for j in range(l1_n):
               pred_l1[i,j] = pred_seq[0][i,j]
       ## level 2
       for i in range(batch_size):
           for j in range(l1_n):
               idx = tree[str(j)]
               for q in idx:
                   pred_l2[i,q] = pred_l1[i,j]*pred_seq[1][i,q]
       # search best label
       for i in range(batch_size):
           # Search level 1
           index_1 = pred_l1[i,:l1_n].argsort()     ##
           if k > 19:                                ## this can be changed for 19
              index_1 = index_1[(l1_n-19):]
           else:
              index_1 = index_1[(l1_n-k):]
           # Search level 2
           ## create a set of level 2 from k best at level 1.
           index_2 = tree[str(index_1[0])]           ##### it is to create index_2 by adding one of items in index_1
           for q in range(1,k):
               index_2.extend(tree[str(index_1[q])]) ##
           ## if not in idx_1_set, then set prob = 0
           for j in range(l1_n,l2_n):
               if j not in index_2:
                  pred_l2[i,j] = 0
           index_2 = pred_l2[i,:].argsort() ##
           index_2 = index_2[len(index_2)-1] ### find highest prob idx
           # set final label
           label_index = pre_path[str(index_2)][1]  #################################################### this is changed for ilsvr65
           pred_label[i,label_index] = 1
       return pred_label
else:
   # kl of Bernoulli distribution
   # to be removed
   print("No more use!!")
   sys.exit()
   pred_seq = []
   for i in range(n_steps):
       pred_seq.append(tf.contrib.distributions.Bernoulli(logits=(tf.matmul(pred[i], weights['out'+str(i)])+biases['out'+str(i)])))
       if i == 0:
          cost = tf.contrib.distributions.kl_divergence(distribution_a=pred_seq[i], distribution_b=tf.contrib.distributions.Bernoulli(probs=y_encoded[:,i,:]))
       else:
          cost += tf.contrib.distributions.kl_divergence(distribution_a=pred_seq[i], distribution_b=tf.contrib.distributions.Bernoulli(probs=y_encoded[:,i,:]))
   cost = tf.reduce_mean(cost)
   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   # Evaluate model
   correct_preds = []
   accuracys = []
   for i in range(n_steps):
       correct_preds.append(tf.equal(tf.argmax(tf.contrib.distributions.Bernoulli.mean(pred_seq[i]),1), tf.argmax(y_encoded[:,i,:],1)))
       accuracys.append(tf.reduce_mean(tf.cast(correct_preds[i], tf.float32)))

# save final label for heatmap
#def save_prediction(train_log_dir,y_prd,y_true,epoch,step):
def save_prediction(infer_dir,inference,epoch):
    outfile_name = infer_dir + "infer_" + str(epoch) + ".npy"
    np.save(outfile_name,inference)
    return

# save predicted_seq and true path
def save_pred_seq(infer_dir,seq_soft,batch_y_val,batch_idx):
    infer_node_dir = infer_dir + 'node/'
    if not os.path.exists(infer_node_dir):
           os.makedirs(infer_node_dir)
    outfile_name_prd = infer_node_dir + "prd_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_prd,seq_soft)
    outfile_name_act = infer_node_dir + "act_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act,batch_y_val)
    return

# save prediction and true label
def save_pred_resnet(infer_dir,pred_soft,batch_y_val,batch_idx):
    infer_node_dir = infer_dir + 'node/'
    if not os.path.exists(infer_node_dir):
           os.makedirs(infer_node_dir)
    outfile_name_prd = infer_node_dir + "prd_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_prd,pred_soft)
    outfile_name_act = infer_node_dir + "act_" + '%05d' % (batch_idx) + ".npy"
    np.save(outfile_name_act,batch_y_val)
    return

# Initializing the variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
#saver = tf.train.Saver(max_to_keep=None)
saver = tf.train.Saver(max_to_keep=1)

# Load data
#train_filenames = sorted(glob.glob(input_dir))
#val_filenames = sorted(glob.glob(input_dir_val))
#train_filenames = sorted(glob.glob(input_dir_val))
#test_filenames = sorted(glob.glob(input_dir_te))

datasets_f = tf.contrib.keras.datasets.cifar100.load_data('fine')
N = int(len(datasets_f[0][1]) * 0.95)
tr_x, tr_y = datasets_f[0][0][:N], datasets_f[0][1][:N] # tr_x = [batch,h,w,ch]
val_x, val_y = datasets_f[0][0][N:], datasets_f[0][1][N:]
te_x, te_y = datasets_f[1][0], datasets_f[1][1]

###############################################################################
####### hc1

# Configuration of the session
#session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.95

if args.model == "hc1":
# Launch the graph for hc1
  with tf.Session() as sess:
    if args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'vgg_16.ckpt'),# 'resnet_v1_50.ckpt'),
              get_init_fn(args.server))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess) ## random initialization
       init_epoch = 0
    else:
       print("Restore from...",args.restore)
       # initialize sess
       sess.run(init)
       # restore model
       saver.restore(sess,args.restore)
       try:
           init_epoch = int(args.restore.split(".")[3].split('_')[-1]) + 1  ## dd
           ## init_epoch = int(args.restore.split(".")[0].split('_')[-1]) + 1  ## celje, lj
       except ValueError:
           print ('Model loading falied!!')
           sys.exit()
    '''
    else:
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              args.restore,
              get_init_fn(args.server))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess)
       init_epoch = 0
    '''
       # Load label Train and val
    batch_Y = np.eye(n_classes+node_added)[tr_y[:,0]+node_added]           ##### this is changed for cifar100
    batch_Y_val = np.eye(n_classes+node_added)[val_y[:,0]+node_added]   ##### this is changed for cifar100
    batch_Y_te = np.eye(n_classes+node_added)[te_y[:,0]+node_added]   ##### this is changed for cifar100
    #batch_Y = np.eye(n_classes+node_added)[np.load(label_dir)]   ##### this is changed for ilsvrc65
    #batch_Y = np.r_[batch_Y,batch_Y]                             ##### this is changed for ilsvrc65 added
    #batch_Y_val = np.eye(n_classes+node_added)[np.load(label_dir_val)] ##### this is changed for ilsvrc65
    epoch = init_epoch
    tr_now = "RNN"
    # start training and validation
    while (epoch < n_epoch):
        # Print out time and current learnining rate
        print(datetime.now())
        #print("Current running rate {}".format(sess.run(optimzer._lr)))
        # only for inference
        # if args.mode == "infer": #############################################
        def infer_test():
           print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(tr_y), len(val_y)))
           infer_dir = home_dir + "code-tf/hc/tmp/hc_finetuned/" + args.idx + "-infer-ep-" + str(epoch-1) + "/"
           if not os.path.exists(infer_dir):
              os.makedirs(infer_dir)
           # validation
           for batch_idx in range(int(len(te_y)/batch_size)):
               batch_x_te = te_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
               #batch_x_te = batch_x_te.transpose((3,1,2,0)).astype(np.float32)
               #batch_x_te[:,:,:,0] -= mean_r
               #batch_x_te[:,:,:,1] -= mean_g
               #batch_x_te[:,:,:,2] -= mean_b
               batch_x_te = preprocessing(batch_x_te)
               batch_y_te = batch_Y_te[batch_idx*len(batch_x_te):(batch_idx+1)*len(batch_x_te)]
               batch_y_en_te = encode(pre_path,batch_y_te,len(batch_x_te),n_steps,n_classes,node_added)
               repeat = 0
               print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(te_y)-1))
               for step in range(0,len(batch_x_te)/batch_size):
                   batch_x_temp_te = batch_x_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_temp_te = batch_y_en_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_path_temp_te = batch_y_te[(step*batch_size):(step+1)*batch_size]
                   # accuracy at each layer out of n_classes + node_added
                   acc = []
                   for t in range(n_steps):
                       acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_te, y_encoded: batch_y_temp_te}))
                   # accuracy of 18 paths
                   pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp_te, y_encoded: batch_y_temp_te})
                   final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                   ## save beam acc
                   temp = np.c_[np.argmax(final_label,1),np.argmax(batch_y_path_temp_te,1)]
                   if batch_idx == 0:
                      inference = temp
                   else:
                      inference = np.r_[inference,temp]
                   correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp_te,1))
                   acc_path = np.mean(correct_path)
                   loss = sess.run(cost, feed_dict={x: batch_x_temp_te, y_encoded: batch_y_temp_te})
                   print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                          .format(acc_path*100,acc[0]*100,acc[1]*100,0*100,loss))
                   ## save pred_s for each minibatch
                   if step == 0:
                      seq_soft_0 = np.expand_dims(pred_s[0],axis=1) ## (19,1,64)
                      seq_soft_1 = np.expand_dims(pred_s[1],axis=1) ## (19,1,64)
                      #seq_soft_2 = np.expand_dims(pred_s[2],axis=1) ## (19,1,64)
                      seq_soft = np.r_['1',seq_soft_0,seq_soft_1]
                      #seq_soft = np.r_['1',seq_soft,seq_soft_2]
                   else:
                      seq_soft_0 = np.expand_dims(pred_s[0],axis=1) ## (19,1,64)
                      seq_soft_1 = np.expand_dims(pred_s[1],axis=1) ## (19,1,64)
                      #seq_soft_2 = np.expand_dims(pred_s[2],axis=1) ## (19,1,64)
                      seq_soft_temp = np.r_['1',seq_soft_0,seq_soft_1]
                      #seq_soft_temp = np.r_['1',seq_soft_temp,seq_soft_2]
                      seq_soft = np.r_[seq_soft,seq_soft_temp]
               ## save pred_seq
               save_pred_seq(infer_dir,seq_soft,batch_y_en_te,batch_idx)
           save_prediction(infer_dir,inference,epoch-1)
           print("Inference is done!")
           sys.exit()

########################################################################################################################
        # Alternatin: Train and validation
        print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(tr_y), len(val_y)))
        # decide switch
        if epoch > -1:
           tr_now = "entire"
        elif (epoch % 30 == 0) and (epoch != 0):
           if tr_now == "RNN":
              tr_now = "CNN"
           else:
              tr_now = "RNN"
        #if epoch == 1:
        #   infer_test()

        # Training RNN
        if tr_now == "RNN":
         print("Training ..epoch {} parts.. {}".format(epoch, tr_now))
         # RNN train for one epoch
         for batch_idx in range(int(len(tr_y)/batch_size)):
            batch_x = tr_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
            #batch_x[:,:,:,0] -= mean_r
            #batch_x[:,:,:,1] -= mean_g
            #batch_x[:,:,:,2] -= mean_b
            batch_x = preprocessing(batch_x)
            batch_y = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]
            batch_y_en = encode(pre_path,batch_y,len(batch_x),n_steps,n_classes,node_added)
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(tr_y)-1))
                for step in range(0,len(batch_x)/batch_size):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y_en[(step*batch_size):(step+1)*batch_size]
                    batch_y_path_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    #aa, a = sess.run([bb,b], feed_dict={x: batch_x_temp})
                    #print (len(a))
                    #print (aa.shape, a.shape) #, print (a[1].shape)
                    #sys.exit()
                    sess.run(optimizer_rnn, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    if args.prt == "full":
                       # accuracy at each layer out of n_classes + node_added
                       acc = []
                       for t in range(n_steps):
                           acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp, y_encoded: batch_y_temp}))
                       # accuracy of 18 paths
                       pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                       correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp,1))
                       beam_acc = np.mean(correct_path)
                       #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                       #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,0*100,loss))
                    else:
                       acc = [0,0,0]
                       beam_acc = 0
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,acc[2]*100,loss))
                repeat+=1
         print(datetime.now())
         # validation
         for batch_idx in range(int(len(val_y)/batch_size)):
            batch_x_val = val_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x_val = batch_x_val.transpose((3,1,2,0)).astype(np.float32)
            #batch_x_val[:,:,:,0] -= mean_r
            #batch_x_val[:,:,:,1] -= mean_g
            #batch_x_val[:,:,:,2] -= mean_b
            batch_x_val = preprocessing(batch_x_val)
            batch_y_val = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]
            batch_y_en_val = encode(pre_path,batch_y_val,len(batch_x_val),n_steps,n_classes,node_added)
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_y)-1))
            for step in range(0,len(batch_x_val)/batch_size):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_en_val[(step*batch_size):(step+1)*batch_size]
                batch_y_path_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                # accuracy at each layer out of n_classes + node_added
                acc = []
                for t in range(n_steps):
                    acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val}))
                # accuracy of 18 paths
                pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                if args.mode == "infer-tr":
                   save_prediction(train_log_dir,np.argmax(final_label,1),np.argmax(batch_y_path_temp_val,1),epoch,step)
                correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp_val,1))
                acc_path = np.mean(correct_path)
                #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                loss = sess.run(cost, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(acc_path*100,acc[0]*100,acc[1]*100,0*100,loss))
         # Save the variables to disk.
         out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
         save_path = saver.save(sess, out_file)
         print("Model saved in file: %s" % save_path)
         epoch+=1

        # Training CNN
        elif tr_now == "CNN":
         print("Training ..epoch {} parts.. {}".format(epoch, tr_now))
         # CNN train for one epoch
         for batch_idx in range(int(len(tr_y)/batch_size)):
            batch_x = tr_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
            #batch_x[:,:,:,0] -= mean_r
            #batch_x[:,:,:,1] -= mean_g
            #batch_x[:,:,:,2] -= mean_b
            batch_x = preprocessing(batch_x)
            batch_y = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]
            batch_y_en = encode(pre_path,batch_y,len(batch_x),n_steps,n_classes,node_added)
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(tr_y)-1))
                for step in range(0,len(batch_x)/batch_size):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y_en[(step*batch_size):(step+1)*batch_size]
                    batch_y_path_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    sess.run(optimizer_cnn, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    if args.prt == "full":
                       # accuracy at each layer out of n_classes + node_added
                       acc = []
                       for t in range(n_steps):
                           acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp, y_encoded: batch_y_temp}))
                       # accuracy of 18 paths
                       pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                       correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp,1))
                       beam_acc = np.mean(correct_path)
                       #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                       #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,0*100,loss))
                    else:
                       acc = [0,0,0]
                       beam_acc = 0
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,acc[2]*100,loss))
                repeat+=1
         print(datetime.now())
         # validation
         for batch_idx in range(int(len(val_y)/batch_size)):
            batch_x_val = val_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x_val = batch_x_val.transpose((3,1,2,0)).astype(np.float32)
            #batch_x_val[:,:,:,0] -= mean_r
            #batch_x_val[:,:,:,1] -= mean_g
            #batch_x_val[:,:,:,2] -= mean_b
            batch_x_val = preprocessing(batch_x_val)
            batch_y_val = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]
            batch_y_en_val = encode(pre_path,batch_y_val,len(batch_x_val),n_steps,n_classes,node_added)
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_y)-1))
            for step in range(0,len(batch_x_val)/batch_size):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_en_val[(step*batch_size):(step+1)*batch_size]
                batch_y_path_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                # accuracy at each layer out of n_classes + node_added
                acc = []
                for t in range(n_steps):
                    acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val}))
                # accuracy of 18 paths
                pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                if args.mode == "infer-tr":
                   save_prediction(train_log_dir,np.argmax(final_label,1),np.argmax(batch_y_path_temp_val,1),epoch,step)
                correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp_val,1))
                acc_path = np.mean(correct_path)
                #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                loss = sess.run(cost, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(acc_path*100,acc[0]*100,acc[1]*100,0*100,loss))
         # Save the variables to disk.
         out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
         save_path = saver.save(sess, out_file)
         print("Model saved in file: %s" % save_path)
         epoch+=1

        # Training entire network
        else: ## tr_now == "entire":
         print("Training ..epoch {} parts.. {}".format(epoch, tr_now))
         # train entire for one epoch
         for batch_idx in range(int(len(tr_y)/batch_size)):
            batch_x = tr_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
            #batch_x[:,:,:,0] -= mean_r
            #batch_x[:,:,:,1] -= mean_g
            #batch_x[:,:,:,2] -= mean_b
            batch_x = preprocessing(batch_x)
            batch_y = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]
            batch_y_en = encode(pre_path,batch_y,len(batch_x),n_steps,n_classes,node_added)
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(tr_y)-1))
                for step in range(0,len(batch_x)/batch_size):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y_en[(step*batch_size):(step+1)*batch_size]
                    batch_y_path_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                    if args.prt == "full":
                       # accuracy at each layer out of n_classes + node_added
                       acc = []
                       for t in range(n_steps):
                           acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp, y_encoded: batch_y_temp}))
                       # accuracy of 18 paths
                       pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                       correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp,1))
                       beam_acc = np.mean(correct_path)
                       #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                       #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,0*100,loss))
                    else:
                       acc = [0,0,0]
                       beam_acc = 0
                       loss = sess.run(cost, feed_dict={x: batch_x_temp, y_encoded: batch_y_temp})
                       print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(beam_acc*100,acc[0]*100,acc[1]*100,acc[2]*100,loss))
                repeat+=1
         print(datetime.now())
         # validation
         for batch_idx in range(int(len(val_y)/batch_size)):
            batch_x_val = val_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x_val = batch_x_val.transpose((3,1,2,0)).astype(np.float32)
            #batch_x_val[:,:,:,0] -= mean_r
            #batch_x_val[:,:,:,1] -= mean_g
            #batch_x_val[:,:,:,2] -= mean_b
            batch_x_val = preprocessing(batch_x_val)
            batch_y_val = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]
            batch_y_en_val = encode(pre_path,batch_y_val,len(batch_x_val),n_steps,n_classes,node_added)
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_y)-1))
            for step in range(0,len(batch_x_val)/batch_size):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_en_val[(step*batch_size):(step+1)*batch_size]
                batch_y_path_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                # accuracy at each layer out of n_classes + node_added
                acc = []
                for t in range(n_steps):
                    acc.append(sess.run(accuracys[t], feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val}))
                # accuracy of 18 paths
                pred_s = sess.run(pred_seq_soft, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                final_label = beam(beam_k,pred_s,batch_size,n_classes,node_added)
                if args.mode == "infer-tr":
                   save_prediction(train_log_dir,np.argmax(final_label,1),np.argmax(batch_y_path_temp_val,1),epoch,step)
                correct_path = np.equal(np.argmax(final_label,1), np.argmax(batch_y_path_temp_val,1))
                acc_path = np.mean(correct_path)
                #print ("Accuracy: {:06.2f}% ".format(acc_path*100))
                #acc_path = sess.run(accuracy_path, feed_dict={x: batch_x_temp_val, y: batch_y_path_temp_val})
                loss = sess.run(cost, feed_dict={x: batch_x_temp_val, y_encoded: batch_y_temp_val})
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(acc_path*100,acc[0]*100,acc[1]*100,0*100,loss))
         # Save the variables to disk.
         out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
         save_path = saver.save(sess, out_file)
         print("Model saved in file: %s" % save_path)
         epoch+=1

        ## inference on test
        if epoch == n_epoch:
           infer_test()

##########################################################################################################
##### vgg-16

if args.model == "vgg-16":
# Launch the graph for vgg-16
  with tf.Session() as sess:
    if args.restore == "":
       print("New training")
       # Define init_fn
       init_fn = slim.assign_from_checkpoint_fn(
              os.path.join(checkpoints_dir, 'vgg_16.ckpt'),# 'resnet_v1_50.ckpt'),
              get_init_fn(args.server))
       # initialize sess
       sess.run(init)
       # Call init_fn
       init_fn(sess) ## random initialization
       init_epoch = 0
    else:
       print("Restore from...",args.restore)
       # initialize sess
       sess.run(init)
       # restore model
       saver.restore(sess,args.restore)
       try:
           init_epoch = int(args.restore.split(".")[3].split('_')[-1]) + 1  ## dd
           ## init_epoch = int(args.restore.split(".")[0].split('_')[-1]) + 1  ## celje, lj
       except ValueError:
           print ('Model loading falied!!')
           sys.exit()
    # Load label Train and val
    batch_Y = np.eye(n_classes)[tr_y[:,0]]      ##### this is changed for cifar100
    batch_Y_val = np.eye(n_classes)[val_y[:,0]] ##### this is changed for cifar100
    batch_Y_te = np.eye(n_classes)[te_y[:,0]]   ##### this is changed for cifar100
    #batch_Y = np.eye(n_classes)[np.load(label_dir)-resnet_label_scale]           ##### this is changed for ilsvrc65
    # batch_Y = np.r_[batch_Y,batch_Y]
    #batch_Y_val = np.eye(n_classes)[np.load(label_dir_val)-resnet_label_scale]   ##### this is changed for ilsvrc65
    #batch_Y = np.eye(n_classes)[np.load(label_dir)]           ##### this is changed for ilsvrc12-10k
    #batch_Y_val = np.eye(n_classes)[np.load(label_dir_val)]   ##### this is changed for ilsvrc12-10k
    epoch = init_epoch
    while (epoch < n_epoch):
        print(datetime.now())
        print("Start ..epoch {} ..train {} ..validation {}".format(epoch, len(tr_y), len(val_y)))
        ## validation
        if args.mode == "infer":
           infer_dir = home_dir + "code-tf/hc/tmp/resnet_finetuned/" + args.idx + "-infer/"
           if not os.path.exists(infer_dir):
              os.makedirs(infer_dir)
           print(datetime.now())
           for batch_idx in range(int(len(te_y)/batch_size)):
               batch_x_te = te_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
               #batch_x_te = batch_x_te.transpose((3,1,2,0)).astype(np.float32)
               #batch_x_te[:,:,:,0] -= mean_r
               #batch_x_te[:,:,:,1] -= mean_g
               #batch_x_te[:,:,:,2] -= mean_b
               batch_x_te = preprocessing(batch_x_te)
               batch_y_te = batch_Y_te[batch_idx*len(batch_x_te):(batch_idx+1)*len(batch_x_te)]
               repeat = 0
               print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(te_y)-1))
               for step in range(0,len(batch_x_te)/batch_size):
                   batch_x_temp_te = batch_x_te[(step*batch_size):(step+1)*batch_size]
                   batch_y_temp_te = batch_y_te[(step*batch_size):(step+1)*batch_size]
                   acc = sess.run(accuracy, feed_dict={x: batch_x_temp_te, y: batch_y_temp_te})
                   pred_temp = sess.run(pred, feed_dict={x: batch_x_temp_te, y: batch_y_temp_te})
                   loss = sess.run(cost, feed_dict={x: batch_x_temp_te, y: batch_y_temp_te})
                   print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                          .format(0*100,0*100,0*100,acc*100,loss))
                   ## save predcitions for each minibatch
                   if step == 0:
                      pred_soft = pred_temp
                   else:
                      pred_soft = np.r_[pred_soft,pred_temp]
               ## save pred_seq
               save_pred_resnet(infer_dir,pred_soft,batch_y_te,batch_idx)
           print("Inference is done!")
           sys.exit()

        ## train and validation
        for batch_idx in range(int(len(tr_y)/batch_size)):
            batch_x = tr_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
            #batch_x[:,:,:,0] -= mean_r
            #batch_x[:,:,:,1] -= mean_g
            #batch_x[:,:,:,2] -= mean_b
            batch_x = preprocessing(batch_x)
            batch_y = batch_Y[batch_idx*len(batch_x):(batch_idx+1)*len(batch_x)]
            repeat = 0
            while (repeat < n_repeat):
                print("..Train..epoch {} -- Repeat {} -- batch: {} / {},".format(epoch,repeat,batch_idx,len(tr_y)-1))
                for step in range(0,len(batch_x)/batch_size):
                    batch_x_temp = batch_x[(step*batch_size):(step+1)*batch_size]
                    batch_y_temp = batch_y[(step*batch_size):(step+1)*batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x_temp, y: batch_y_temp})
                    acc = sess.run(accuracy, feed_dict={x: batch_x_temp, y: batch_y_temp})
                    loss = sess.run(cost, feed_dict={x: batch_x_temp, y: batch_y_temp})
                    print("Train ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                              .format(0*100,0*100,0*100,acc*100,loss))
                repeat+=1
        # validation
        print(datetime.now())
        for batch_idx in range(int(len(val_y)/batch_size)):
            batch_x_val = val_x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
            #batch_x_val = batch_x_val.transpose((3,1,2,0)).astype(np.float32)
            #batch_x_val[:,:,:,0] -= mean_r
            #batch_x_val[:,:,:,1] -= mean_g
            #batch_x_val[:,:,:,2] -= mean_b
            batch_x_val = preprocessing(batch_x_val)
            batch_y_val = batch_Y_val[batch_idx*len(batch_x_val):(batch_idx+1)*len(batch_x_val)]
            repeat = 0
            print("....Validation...epoch {} -- batch: {} / {},".format(epoch,batch_idx,len(val_y)-1))
            for step in range(0,len(batch_x_val)/batch_size):
                batch_x_temp_val = batch_x_val[(step*batch_size):(step+1)*batch_size]
                batch_y_temp_val = batch_y_val[(step*batch_size):(step+1)*batch_size]
                acc = sess.run(accuracy, feed_dict={x: batch_x_temp_val, y: batch_y_temp_val})
                loss = sess.run(cost, feed_dict={x: batch_x_temp_val, y: batch_y_temp_val})
                print("Validation ....Beam: {:06.2f}% Acc1: {:06.2f}% Acc2: {:06.2f}% Acc3: {:06.2f}% Loss: {:08.5f}"
                       .format(0*100,0*100,0*100,acc*100,loss))
        # Save the variables to disk.
        out_file = os.path.join(train_log_dir,model_name+"_epoch_"+str(epoch)+".ckpt")
        save_path = saver.save(sess, out_file)
        print("Model saved in file: %s" % save_path)
        epoch+=1

