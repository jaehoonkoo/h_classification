# h_classification
Combined Convolutional and Recurrent Neural Networks for Hierarchical Classification of Images

---------------------------------------------------------------
Author: [Jaehoon Koo](https://www.linkedin.com/in/jaehoon-koo-bb384aa1/)
---------------------------------------------------------------
This directory contains sample scripts of DHNN paper (https://ieeexplore.ieee.org/abstract/document/9378237) pusblished in [IEEE BigData 2020](http://bigdataieee.org/BigData2020/AcceptedPapers.html).

Scripts are written in Python3 with Tensorflow based on TF-Slim library (https://github.com/google-research/tf-slim), SENet (https://github.com/kobiso/SENet-tensorflow-slim), CBAM (https://github.com/kobiso/CBAM-keras), and bidirectional RNN (https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py).

---------------------------------------------------------------
Dataset:

https://northwestern.app.box.com/s/i23bm7eee5irjt164sn1zqzxvyyyrc37/folder/70446923518

---------------------------------------------------------------
Files:

[code/hc-s2s-oi-v1-2-pr-run-val-se.py](https://github.com/jke513/h_classification/blob/master/code/hc-s2s-oi-v1-2-pr-run-val-se.py): General tree model with SE net

[code/hc-s2s-oi-v1-2-pr-run-val-att.py](https://github.com/jke513/h_classification/blob/master/code/hc-s2s-oi-v1-2-pr-run-val-att.py): General tree model with CBAM net

[code/PR-node-bulk-val-te.py](https://github.com/jke513/h_classification/blob/master/code/PR-node-bulk-val-te.py): Compute node accuracy of general tree model

[code/PR-path-bulk-ori-val-te.py](https://github.com/jke513/h_classification/blob/master/code/PR-path-bulk-ori-val-te.py): Compute path accuracy of general tree model

---------------------------------------------------------------
To run:

python hc-s2s-oi-v1-2-pr-run-val-se.py -server="nu" -model="resnet-50" -idx="idx-0" -batch=32 -lr=0.0001 -restore="" -prt="full" -opt="adam" -epoch=15 -base="res" -seed=1234

python hc-s2s-oi-v1-2-pr-run-val-se.py -server="nu" -model="hc" -idx="idx-0" -batch=32 -lr=0.0001 -input=2048 -hidden=1024 -conversion=3.1 -average="1;2;3 4" -restore="" -prt="full" -k=2 -opt="adam" -epoch=20 -alt1=-1 -alt2=20 -base="res" -seed=1234

python hc-s2s-oi-v1-2-pr-run-val-att.py -server="nu" -model="resnet-50" -idx="idx-0" -batch=32 -lr=0.0001 -restore="" -prt="full" -opt="adam" -epoch=15 -base="res" -seed=1234 -attention=cbam_block

python hc-s2s-oi-v1-2-pr-run-val-att.py -server="nu" -model="hc" -idx="idx-0" -batch=32 -lr=0.0001 -input=256 -hidden=512 -conversion=3.1 -average="1;2;3 4" -restore="" -prt="full" -k=2 -opt="adam" -epoch=20 -alt1=-1 -alt2=20 -base="res" -seed=1234 -attention=cbam_block
 
---------------------------------------------------------------
