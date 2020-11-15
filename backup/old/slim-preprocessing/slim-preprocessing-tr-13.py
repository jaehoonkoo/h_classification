
import numpy as np
import os, sys, base64, glob
import tensorflow as tf
import hickle as hkl
from scipy import misc

from random import randint

def random_(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

slim_path = "/home/jkoo/code-tf/slim/models/research/slim/"
sys.path.append(slim_path)

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing
from tensorflow.contrib import slim

image_size = vgg.vgg_16.default_image_size

home_dir = '/scratch/jkoo/data/open-images/'
save_dir = '/scratch/jkoo/data/open-images/arrays/tr_npy_b256_slim_13/'
file_names = np.load(home_dir + 'tr_img_path.npy')

batch_size = 256
scale = -1 # batch index subset0 = (0,19),  subset1 = (20,39),...
start = int(sys.argv[1])
end = start + batch_size

corrupted_imgs = []
empty_img = np.zeros((1,image_size,image_size,3),dtype=np.float32)
count = 0

for i in range(start,end):
    print ("processing....", i)
    url = file_names[i]
    image = misc.imread(url)
    assert image.dtype == 'uint8', image
    if len(image.shape) == 2:
       image = np.asarray([image, image, image])
       image = image.transpose((1,2,0))
    if i % batch_size == 0:
       seed_w = random_(4)
       seed_h = random_(4)
       processed_image = vgg_preprocessing.preprocess_image(tf.constant(image), image_size, image_size, is_training=True, seed_h=seed_h, seed_w=seed_w)
       processed_images  = tf.expand_dims(processed_image, 0)
       sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
       with sess.as_default():
            try:
                temp = processed_images.eval()
                save_batch = np.r_[temp]
            except:
                print (i,url)
                temp = empty_img
                save_batch = np.r_[temp]
                corrupted_imgs.append([i,count,url])
                count +=1
    elif (i+1) % batch_size == 0:
       seed_w = random_(4)
       seed_h = random_(4)
       processed_image = vgg_preprocessing.preprocess_image(tf.constant(image), image_size, image_size, is_training=True, seed_h=seed_h, seed_w=seed_w)
       processed_images  = tf.expand_dims(processed_image, 0)
       sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
       with sess.as_default():
            try:
                temp = processed_images.eval()
                save_batch = np.r_[save_batch,temp]
            except:
                print (i,url)
                temp = empty_img
                save_batch = np.r_[save_batch,temp]
                corrupted_imgs.append([i,count,url])
                count +=1
            file_name = save_dir + '%05d' % ((i+1) / batch_size + scale) + '.npy'
            np.save(file_name,save_batch)
            corrupted_name = save_dir + '%05d' % ((i+1) / batch_size + scale) + '_corrupted.npy'
            np.save(corrupted_name,corrupted_imgs)
    else:
       seed_w = random_(4)
       seed_h = random_(4)
       processed_image = vgg_preprocessing.preprocess_image(tf.constant(image), image_size, image_size, is_training=True, seed_h=seed_h, seed_w=seed_w)
       processed_images  = tf.expand_dims(processed_image, 0)
       sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
       with sess.as_default():
            try:
                temp = processed_images.eval()
                save_batch = np.r_[save_batch,temp]
            except:
                print (i,url)
                temp = empty_img
                save_batch = np.r_[save_batch,temp]
                corrupted_imgs.append([i,count,url])
                count +=1


