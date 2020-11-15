################################################

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import keras

def model_resnet(x,n_class):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
         net_, end_points_ = resnet_v1.resnet_v1_50(x, n_class, is_training=True)
    #var_names_cnn = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1')]

    #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HC_CONV'):
    #     var_names_cnn.append(i)
        
    # Define 4 blocks
    bl1_name_ = 'resnet_v1_50/block1'
    bl2_name_ = 'resnet_v1_50/block2'
    bl3_name_ = 'resnet_v1_50/block3'
    bl4_name_ = 'resnet_v1_50/block4'

    blocks = {
        'bl1': end_points_[bl1_name_],
        'bl2': end_points_[bl2_name_],
        'bl3': end_points_[bl3_name_],
        'bl4': end_points_[bl4_name_]}

    conv1 = blocks['bl1']
    conv2 = blocks['bl2']
    conv3 = blocks['bl3']
    conv4 = blocks['bl4']
    return conv4

################################################
from keras import layers, models, optimizers
from keras.layers import Input, Conv2D, Dense
from keras.layers import Reshape, Layer, Lambda
from keras.models import Model
from keras.utils import to_categorical
from keras import initializers
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

import numpy as np
import tensorflow as tf

# First, let’s define the Squash function:
def squash(output_vector, axis=-1):
    norm = tf.reduce_sum(tf.square(output_vector), axis, keep_dims=True)
    return output_vector * norm / ((1 + norm) * tf.sqrt(norm + 1.0e-10))

# After defining the Squash function, we can define the masking layer:
class MaskingLayer(Layer):
    def call(self, inputs, **kwargs):
        input, mask = inputs
        return K.batch_dot(input, mask, 1)

    def compute_output_shape(self, input_shape):
        *_, output_shape = input_shape[0]
        return (None, output_shape)
    
# Now, let’s define the primary Capsule function:
def PrimaryCapsule(n_vector, n_channel, n_kernel_size, n_stride, padding='valid'):
    def builder(inputs):
        output = Conv2D(filters=n_vector * n_channel, kernel_size=n_kernel_size, strides=n_stride, padding=padding)(inputs)
        output = Reshape( target_shape=[-1, n_vector], name='primary_capsule_reshape')(output)
        return Lambda(squash, name='primary_capsule_squash')(output)
    return builder

# After that, let’s write the capsule layer class:
class CapsuleLayer(Layer):
    def __init__(self, n_capsule, n_vec, n_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.n_capsule = n_capsule
        self.n_vector = n_vec
        self.n_routing = n_routing
        self.kernel_initializer = initializers.get('he_normal')
        self.bias_initializer = initializers.get('zeros')

    def build(self, input_shape): # input_shape is a 4D tensor
        _, self.input_n_capsule, self.input_n_vector, *_ = input_shape
        self.W = self.add_weight(shape=[self.input_n_capsule, self.n_capsule, self.input_n_vector, self.n_vector], initializer=self.kernel_initializer, name='W')
        self.bias = self.add_weight(shape=[1, self.input_n_capsule, self.n_capsule, 1, 1], initializer=self.bias_initializer, name='bias', trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
        input_tiled = tf.tile(input_expand, [1, 1, self.n_capsule, 1, 1])
        input_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]), 
                            elems=input_tiled, initializer=K.zeros( [self.input_n_capsule, self.n_capsule, 1, self.n_vector]))
        for i in range(self.n_routing): # routing
            c = tf.nn.softmax(self.bias, dim=2)
            outputs = squash(tf.reduce_sum( c * input_hat, axis=1, keep_dims=True))
            if i != self.n_routing - 1:
                self.bias += tf.reduce_sum(input_hat * outputs, axis=-1, keep_dims=True)
        return tf.reshape(outputs, [-1, self.n_capsule, self.n_vector])

    def compute_output_shape(self, input_shape):
        # output current layer capsules
        return (None, self.n_capsule, self.n_vector)

# The class below will compute the length of the capsule
class LengthLayer(Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=False))

    def compute_output_shape(self, input_shape):
        *output_shape, _ = input_shape
        return tuple(output_shape)

# The function below will compute the margin loss:    
def margin_loss(y_ground_truth, y_prediction):
    _m_plus = 0.9
    _m_minus = 0.1
    _lambda = 0.5
    L = y_ground_truth * tf.square(tf.maximum(0., _m_plus - y_prediction)) + _lambda * ( 1 - y_ground_truth) * tf.square(tf.maximum(0., y_prediction - _m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# After defining the different necessary building blocks of the network we can now preprocess the MNIST dataset input for the network:
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#y_train = to_categorical(y_train.astype('float32'))
#y_test = to_categorical(y_test.astype('float32'))
#X = np.concatenate((x_train, x_test), axis=0)
#Y = np.concatenate((y_train, y_test), axis=0)

# Below are some variables that will represent the shape of the input, number of output classes, and number of routings:
input_shape = [224, 224, 3] #[28, 28, 1]
n_class     = 30#10
n_routing   = 3

# Now, let’s create the encoder part of the network:
x = Input(shape=input_shape)
conv1 = Conv2D(filters=256, kernel_size=9*8*3, strides=1, padding='valid', activation='relu', name='conv1')(x)
primary_capsule = PrimaryCapsule(n_vector=8, n_channel=32, n_kernel_size=9, n_stride=2)(conv1)
#conv0 = model_resnet(x,n_class)
#conv1 = Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(conv0)
#primary_capsule = PrimaryCapsule(n_vector=8, n_channel=32, n_kernel_size=7, n_stride=2)(conv0)
digit_capsule = CapsuleLayer( n_capsule=n_class, n_vec=16, n_routing=n_routing, name='digit_capsule')(primary_capsule)
output_capsule = LengthLayer(name='output_capsule')(digit_capsule)

#print (conv1)
print (primary_capsule)
print (digit_capsule)
print (output_capsule)
#print (y_test.shape)

# Then let’s create the decoder part of the network:
#mask_input = Input(shape=(n_class, ))
#mask = MaskingLayer()([digit_capsule, mask_input])  # two inputs
#dec = Dense(512, activation='relu')(mask)
#dec = Dense(1024, activation='relu')(dec)
#dec = Dense(784, activation='sigmoid')(dec)
#dec = Dense(224*224*3, activation='sigmoid')(dec)
#dec = Reshape(input_shape)(dec)

# Now let’s create the entire model and compile it:
#model = Model([x, mask_input], [output_capsule, dec])
#model.compile(optimizer='adam', loss=[ margin_loss, 'mae' ], metrics=[ margin_loss, 'mae', 'accuracy'])

model = Model(x,output_capsule)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss=margin_loss, metrics=[ margin_loss, 'accuracy'])

model.summary()

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

mean_r = 123.68 # ilsvrc1k-123.68  | cifar100-129.30417
mean_g = 116.779 # ilsvrc1k-116.779 | cifar100-124.06996
mean_b = 103.935 # ilsvrc1k-103.935 | cifar100-112.43405
std_r = 1 # | cifar100-68.14695
std_g = 1 # | cifar100-65.37863
std_b = 1 # | cifar100-70.40022

def preprocessing(batch_x_name): #, cnn_base, server):
    #if cnn_base == "res" or model_name == "resnet-50":
    #   if server == 'nu':
    #      #batch_x = hkl.load(batch_x_name) ## ilsvrc65
    #      batch_x = np.load(batch_x_name)  ## oi 
    #   else: ## allstate
    #      batch_x = np.load(batch_x_name)
    #   # batch_x = batch_x.transpose((3,1,2,0)).astype(np.float32)
    #else: ## cnn_base == "vgg"
    #   batch_x = np.load(batch_x_name)
    batch_x = np.load(batch_x_name)
    batch_x[:,:,:,0] -= mean_r
    batch_x[:,:,:,1] -= mean_g
    batch_x[:,:,:,2] -= mean_b
    batch_x[:,:,:,0] /= std_r
    batch_x[:,:,:,1] /= std_g
    batch_x[:,:,:,2] /= std_b
    return batch_x

data_dir = "/scratch/jkoo/data/open-images/"                                 ## lj and celje
input_dir = data_dir + "arrays/tr_npy_b256_slim/*.npy"
label_dir = data_dir + "arrays/labels/tr_list_label_path.npy"
input_dir_val = data_dir + "arrays/val_npy_b256_slim/*.npy"
label_dir_val = data_dir + "arrays/labels/val_list_label_path.npy"

train_filenames = sorted(glob.glob(input_dir))
batch_Y         = np.load(label_dir)     ##### s2s

val_filenames = sorted(glob.glob(input_dir_val))
batch_Y_val   = np.load(label_dir_val)     ##### s2s

Epoch = 10
minibatch_n = 256
epoch = 0 
save_dir = '/scratch/jkoo/code-tf/hc/tmp/cnn_finetuned/idx-cap-0/'
### training with larger filter size 

while (epoch < Epoch):
    # training 
    for batch_idx in range(len(train_filenames)):
        batch_x = preprocessing(train_filenames[batch_idx])#, args.base, args.server)
        batch_y = batch_Y[batch_idx*minibatch_n:(batch_idx+1)*minibatch_n]
        model.fit(batch_x, batch_y, batch_size=32, epochs=1)
        
    # validation
    acc_val = 0
    for batch_idx in range(len(val_filenames)):
        batch_x = preprocessing(val_filenames[batch_idx])#, args.base, args.server)
        batch_y = batch_Y_val[batch_idx*minibatch_n:(batch_idx+1)*minibatch_n]
        #results = model.evaluate(batch_x, batch_y, batch_size=32)
        ynew = model.predict(batch_x)
        #fscore(batch_y,ynew)
        save_prd_name = save_dir + 'val/epoch' + str(epoch) + '/prd_' + '%05d' % (batch_idx) + '.npy'
        np.save(save_prd_name,ynew)
        #acc_val += results[2]
        #print ('val acc',batch_idx, results[2])
    #print ('Mean val acc', float(acc_val / len(val_filenames)) )
    save_weight_name = save_dir + 'model/capsule_trained_' + str(epoch) + '.h5'
    model.save_weights(save_weight_name)
    epoch +=1

















