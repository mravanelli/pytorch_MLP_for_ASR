## Introduction:
This codes implements a basic MLP for HMM-DNN speech recognition. The MLP is trained with *pytorch*, while feature extraction, alignments, and decoding are performed with *Kaldi*. The current implementation supports dropout and batch normalization. An example for phoneme recognition using the standard TIMIT dataset is provided.
 
## Prerequisites:
 - Make sure that python is installed (the code is tested with python 2.7). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

- If not already done, install pytorch (http://pytorch.org/) and make sure the installation is actually working. As a first test, type “python” and, once entered into the console, type “import torch”. Make sure everything is fine. 

- If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of Kaldi binaries into the .bashrc file. As a first test to check the installation, open  a bash shell, type “copy-feats” and make sure no errors appear.

- Install kaldi-io package from the kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python). It provides a simple interface between kaldi and python. To install it:

1) run git clone https://github.com/vesis84/kaldi-io-for-python.git <kaldi-io-dir> 
2) add PYTHONPATH=${PYTHONPATH}:<kaldi-io-dir> to $HOME/.bashrc 
3) now type import kaldi_io from the python console and make sure the package is correctly imported.  You can find more info (including some reading and writing tests) on  https://github.com/vesis84/kaldi-io-for-python

 
The code has been tested with:
- Python  2.7 
- Ubuntu 17.04
- Pytorch 0.3
 
## How to run a TIMIT experiment:

#### 1. Run the Kaldi s5 baseline of TIMIT.  
This step is necessary to  derive features and labels later used to train the MLP.  In particular: 
- go to *$KALDI_ROOT/egs/timit/s5*.
- run the script *run.sh*. Make sure everything works fine. Please, also run the Karel’s DNN baseline using  “$KALDI_ROOT/egs/timit/s5/local/nnet/run_dnn.sh”.
- Compute the alignments for test and dev data with the following commands:
If you wanna use tri3 alignments type:
``` 
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
```
If you wanna use dnn alignments (as suggested) type:
``` 
steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
``` 
            
 
#### 2. Split the feature lists in  chunks. 
Go to the "pytorch_MLP_for_ASR" folder.
The "create_chunks.sh" script first shuffles or sorts (based on the sentence length) a kaldi feature list and then split it into a certain number of chunks. Shuffling a list could be good for feed-forward DNNs, while a sorted list can be useful for RNNs (not used here). The code also computes per-speaker and per-sentence CMVN.  
 
 For mfcc features run:
 ``` 
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/train mfcc_lists 5 train 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/dev mfcc_lists 1 dev 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/test mfcc_lists 1 test 0
 ``` 
 For fMLLR features run:
 ``` 
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/data-fmllr-tri3/train fmllr_lists 5 train 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/data-fmllr-tri3/dev fmllr_lists 1 dev 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/data-fmllr-tri3/test fmllr_lists 1 test 0
 ``` 

#### 3. Setup the Config file. 
- Open the file *TIMIT_MLP_mfcc.cfg*,*TIMIT_MLP_fmllr.cfg* and modify them according to your paths.
1) *tr_fea_scp* contains a list of the scp files created with *create_chunks.sh*. 
2) *tr_fea_opts* allows users to easily add normalizations, derivatives and other types of feature processing  (see for instance *TIMIT_MLP_mfcc.cfg*). 
3) *tr_lab_folder* is the kaldi folder containing the alignments (labels)
4) *tr_lab_opts* allows users to derive context-dependent phone targets (when set to *ali-to-pdf*) or monophone targets (when set to *ali-to-phones --per-frame*)
5) Modify the paths for dev and test data
6) Feel free to modify the DNN architecture and the other optimization parameters according to your needs. 
7) The required *count_file* in the config file (used to normalize the DNN posteriors before feeding the decoder) corresponds to the following file: *$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts* (that is automatically created by Kaldi when running s5 recipe)
8) Use the option *use_cuda=1* for running the code on a GPU (strongly suggested).
9) Use the option *save_gpumem=0* to save gpu memory. The code will be a little bit slower (about 10-15%), but it saves gpu memory. Use *save_gpumem=1* only if your GPU has more that 2GB of memory. 

 
#### 4. Train the DNN. 
- To run DNN training type:
```
python MLP_ASR --cfg TIMIT_mfcc.cfg
``` 
or 
```
python MLP_ASR --cfg TIMIT_fmllr.cfg
``` 

If everything is working fine, your output (for fMLLR features) should look like this:
``` 
epoch 1 training_cost=3.185877, training_error=0.689695, dev_error=0.551222, test_error=0.546513, learning_rate=0.080000, execution_time(s)=122.376381
epoch 2 training_cost=1.942640, training_error=0.531165, dev_error=0.499130, test_error=0.496296, learning_rate=0.080000, execution_time(s)=98.377268
epoch 3 training_cost=1.726837, training_error=0.486872, dev_error=0.479094, test_error=0.480323, learning_rate=0.080000, execution_time(s)=80.488474
epoch 4 training_cost=1.605198, training_error=0.460752, dev_error=0.466708, test_error=0.467406, learning_rate=0.080000, execution_time(s)=80.552025
epoch 5 training_cost=1.517380, training_error=0.441869, dev_error=0.457906, test_error=0.460688, learning_rate=0.080000, execution_time(s)=80.770670
epoch 6 training_cost=1.446593, training_error=0.424615, dev_error=0.453285, test_error=0.453660, learning_rate=0.080000, execution_time(s)=80.582975
epoch 7 training_cost=1.390792, training_error=0.412072, dev_error=0.448672, test_error=0.449844, learning_rate=0.080000, execution_time(s)=81.266987
epoch 8 training_cost=1.341816, training_error=0.400947, dev_error=0.441144, test_error=0.441814, learning_rate=0.080000, execution_time(s)=80.780559
epoch 9 training_cost=1.301799, training_error=0.391474, dev_error=0.438507, test_error=0.440588, learning_rate=0.080000, execution_time(s)=80.589483
epoch 10 training_cost=1.265721, training_error=0.382258, dev_error=0.435233, test_error=0.438567, learning_rate=0.080000, execution_time(s)=80.757482
epoch 11 training_cost=1.234330, training_error=0.375200, dev_error=0.434359, test_error=0.436927, learning_rate=0.080000, execution_time(s)=81.102308
epoch 12 training_cost=1.204328, training_error=0.367166, dev_error=0.432048, test_error=0.433611, learning_rate=0.080000, execution_time(s)=80.633779
epoch 13 training_cost=1.179567, training_error=0.361736, dev_error=0.430162, test_error=0.434388, learning_rate=0.080000, execution_time(s)=80.789723
epoch 14 training_cost=1.154040, training_error=0.355597, dev_error=0.430750, test_error=0.433180, learning_rate=0.080000, execution_time(s)=80.880159
epoch 15 training_cost=1.087669, training_error=0.337308, dev_error=0.415972, test_error=0.422594, learning_rate=0.040000, execution_time(s)=80.836763
epoch 16 training_cost=1.054655, training_error=0.328579, dev_error=0.417336, test_error=0.423976, learning_rate=0.040000, execution_time(s)=79.909844
epoch 17 training_cost=1.022706, training_error=0.320052, dev_error=0.412804, test_error=0.418415, learning_rate=0.020000, execution_time(s)=80.027892
epoch 18 training_cost=1.004981, training_error=0.315107, dev_error=0.411277, test_error=0.418363, learning_rate=0.020000, execution_time(s)=80.182567
epoch 19 training_cost=0.993245, training_error=0.311901, dev_error=0.410142, test_error=0.419330, learning_rate=0.020000, execution_time(s)=80.251891
epoch 20 training_cost=0.984327, training_error=0.310019, dev_error=0.410796, test_error=0.416654, learning_rate=0.020000, execution_time(s)=81.954019
epoch 21 training_cost=0.968733, training_error=0.305669, dev_error=0.408787, test_error=0.416550, learning_rate=0.010000, execution_time(s)=80.997200
epoch 22 training_cost=0.960621, training_error=0.302933, dev_error=0.408011, test_error=0.415566, learning_rate=0.010000, execution_time(s)=80.916291
epoch 23 training_cost=0.953147, training_error=0.300761, dev_error=0.408330, test_error=0.417154, learning_rate=0.010000, execution_time(s)=80.901808
epoch 24 training_cost=0.949771, training_error=0.300328, dev_error=0.406893, test_error=0.415255, learning_rate=0.005000, execution_time(s)=81.138160
``` 
#### 4. Kaldi Decoding.
The training script creates (during the last epoch) a file *pout_test.ark* containing a set of likelihoods (i.e., normalized posterior probabilities) computed on the test sentences. These likelihoods can be used to feed the Kaldi decoder in this way:
``` 
cd kaldi_decoding_scripts

 ./decode_dnn_TIMIT.sh $KALDI_ROOT/egs/timit/s5/exp/tri3/graph \
 $KALDI_ROOT/egs/timit/s5/data/test/ \
 $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \
 ../TIMIT_MLP_mfcc/decoding_test \
 "cat ../TIMIT_MLP_mfcc/pout_test.ark"
``` 

#### 5. Check the results.
- After that training and decoding phases are finished, you can go into the *kaldi_decoding_scripts* folder and run *./RESULTS* to check the system performance.  
 
- If everything if fine, you should obtain  Phone Error Rates (PER%) similar to the following ones:
mfcc features: PER=18.7%
fMLLR features: PER=16.7%

Note that, despite the simplicity of the MLP implemented here, the performance obtained with this implementation are slightly better than that achieved with the kaldi baselines (even without pre-training or sMBR).  See for instance the file $KALDI_ROOT/egs/timit/s5/RESULTS$


## Reference:
Please, cite my PhD thesis in you use the code:

*[1] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017*

https://arxiv.org/abs/1712.06086
