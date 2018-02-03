## Introduction:
This code implements a basic *MLP* for *HMM-DNN* speech recognition. The MLP is trained with *pytorch*, while feature extraction, alignments, and decoding are performed with *Kaldi*. The current implementation supports dropout and batch normalization. An example for phoneme recognition using the standard TIMIT dataset is provided.
 
## Prerequisites:
 - Make sure that python is installed (the code is tested with python 2.7). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

- If not already done, install pytorch (http://pytorch.org/) and make sure that the installation works. As a first test, type “python” and, once entered into the console, type “import torch”. Make sure everything is fine. 

- If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into *$HOME/.bashrc*. As a first test to check the installation, open  a bash shell, type “copy-feats” and make sure no errors appear.

- Install *kaldi-io* package from the *kaldi-io-for-python* project (https://github.com/vesis84/kaldi-io-for-python). It provides a simple interface between kaldi and python. To install it:

1) run git clone https://github.com/vesis84/kaldi-io-for-python.git <kaldi-io-dir> 
2) add *PYTHONPATH=${PYTHONPATH}:<kaldi-io-dir>* to *$HOME/.bashrc* 
3) now type *import kaldi_io* from the python console and make sure the package is correctly imported.  You can find more info (including some reading and writing tests) on  https://github.com/vesis84/kaldi-io-for-python

 
The code has been tested with:
- Python  2.7 
- Ubuntu 17.04
- Pytorch 0.3
- Cuda 9.1
 
## How to run a TIMIT experiment:

#### 1. Run the Kaldi s5 baseline of TIMIT.  
This step is necessary to  compute features and labels later used to train the pytorch MLP.  In particular: 
- go to *$KALDI_ROOT/egs/timit/s5*.
- run the script *run.sh*. Make sure everything works fine. Please, also run the Karel’s DNN baseline using  *local/nnet/run_dnn.sh*.
- Compute the alignments for test and dev data with the following commands.

If you wanna use *tri3* alignments, type:
``` 
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
```

If you wanna use *dnn* alignments (as suggested), type:
``` 
steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
``` 
            
 
#### 2. Split the feature lists into chunks. 
Go to the *pytorch_MLP_for_ASR* folder.
The *create_chunks.sh* script first shuffles or sorts (based on the sentence length) a kaldi feature list and then split it into a certain number of chunks. Shuffling a list could be good for feed-forward DNNs, while a sorted list can be useful for RNNs (not used here). The code also computes per-speaker and per-sentence CMVN.  
 
 For mfcc features run:
 ``` 
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/train mfcc_lists 5 train 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/dev mfcc_lists 1 dev 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/test mfcc_lists 1 test 0
 ``` 
 For fMLLR features run:
 ``` 
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data-fmllr-tri3/train fmllr_lists 5 train 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data-fmllr-tri3/dev fmllr_lists 1 dev 0
 ./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data-fmllr-tri3/test fmllr_lists 1 test 0
 ``` 

#### 3. Setup the Config file. 
- Open the files *TIMIT_MLP_mfcc.cfg*,*TIMIT_MLP_fmllr.cfg* and modify them according to your paths.
1) *tr_fea_scp* contains the list of features created with *create_chunks.sh*. 
2) *tr_fea_opts* allows users to easily add normalizations, derivatives and other types of feature processing  (see for instance *TIMIT_MLP_mfcc.cfg*). 
3) *tr_lab_folder* is the kaldi folder containing the alignments (labels).
4) *tr_lab_opts* allows users to derive context-dependent phone targets (when set to *ali-to-pdf*) or monophone targets (when set to *ali-to-phones --per-frame*).
5) Modify the paths for dev and test data.
6) Feel free to modify the DNN architecture and the other optimization parameters according to your needs. 
7) The required *count_file* (used to normalize the DNN posteriors before feeding the decoder and automaticallt created by kadldi when running s5 recipe) can be found here: *$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts*.
8) Use the option *use_cuda=1* for running the code on a GPU (strongly suggested).
9) Use the option *save_gpumem=0* to save gpu memory. The code will be a little bit slower (about 10-15%), but it saves gpu memory. Use *save_gpumem=1* only if your GPU has more that 2GB of memory. 

 
#### 4. Train the DNN. 
- Type the following command to run DNN training :
```
python MLP_ASR.py --cfg TIMIT_MLP_mfcc.cfg 2> log.log
``` 
or 
```
python MLP_ASR.py --cfg TIMIT_MLP_fmllr.cfg 2> log.log
``` 

If everything is working fine, your output (for fMLLR features) should look like this:
``` 
epoch 1 training_cost=3.185270, training_error=0.690495, dev_error=0.549124, test_error=0.549172, learning_rate=0.080000, execution_time(s)=85.436095
epoch 2 training_cost=1.950891, training_error=0.533513, dev_error=0.498461, test_error=0.499940, learning_rate=0.080000, execution_time(s)=78.238582
epoch 3 training_cost=1.737371, training_error=0.489724, dev_error=0.474726, test_error=0.479321, learning_rate=0.080000, execution_time(s)=78.390865
epoch 4 training_cost=1.610313, training_error=0.461962, dev_error=0.464242, test_error=0.465437, learning_rate=0.080000, execution_time(s)=78.282750
epoch 5 training_cost=1.521487, training_error=0.442533, dev_error=0.455979, test_error=0.457632, learning_rate=0.080000, execution_time(s)=77.166886
epoch 6 training_cost=1.452035, training_error=0.426761, dev_error=0.451179, test_error=0.453436, learning_rate=0.080000, execution_time(s)=77.064029
epoch 7 training_cost=1.394820, training_error=0.413627, dev_error=0.443357, test_error=0.445354, learning_rate=0.080000, execution_time(s)=78.169549
epoch 8 training_cost=1.347145, training_error=0.402646, dev_error=0.441773, test_error=0.444214, learning_rate=0.080000, execution_time(s)=77.795720
epoch 9 training_cost=1.305390, training_error=0.392546, dev_error=0.437266, test_error=0.443972, learning_rate=0.080000, execution_time(s)=77.853706
epoch 10 training_cost=1.269022, training_error=0.383710, dev_error=0.434375, test_error=0.442246, learning_rate=0.080000, execution_time(s)=78.647257
epoch 11 training_cost=1.235830, training_error=0.375467, dev_error=0.431452, test_error=0.433421, learning_rate=0.080000, execution_time(s)=77.407901
epoch 12 training_cost=1.205622, training_error=0.368032, dev_error=0.432220, test_error=0.434976, learning_rate=0.080000, execution_time(s)=78.074666
epoch 13 training_cost=1.132139, training_error=0.348587, dev_error=0.419605, test_error=0.423768, learning_rate=0.040000, execution_time(s)=77.901673
epoch 14 training_cost=1.098085, training_error=0.339950, dev_error=0.418413, test_error=0.423302, learning_rate=0.040000, execution_time(s)=78.078263
epoch 15 training_cost=1.079947, training_error=0.335365, dev_error=0.418103, test_error=0.424390, learning_rate=0.040000, execution_time(s)=77.930880
epoch 16 training_cost=1.042311, training_error=0.325323, dev_error=0.412053, test_error=0.417673, learning_rate=0.020000, execution_time(s)=78.131381
epoch 17 training_cost=1.025281, training_error=0.320353, dev_error=0.413433, test_error=0.418553, learning_rate=0.020000, execution_time(s)=78.066443
epoch 18 training_cost=1.004788, training_error=0.314823, dev_error=0.408852, test_error=0.415479, learning_rate=0.010000, execution_time(s)=79.046657
epoch 19 training_cost=0.995931, training_error=0.312059, dev_error=0.409269, test_error=0.414081, learning_rate=0.010000, execution_time(s)=77.593635
epoch 20 training_cost=0.985299, training_error=0.309571, dev_error=0.407089, test_error=0.412613, learning_rate=0.005000, execution_time(s)=78.187198
epoch 21 training_cost=0.980231, training_error=0.308210, dev_error=0.406648, test_error=0.412423, learning_rate=0.005000, execution_time(s)=78.114028
epoch 22 training_cost=0.977153, training_error=0.307378, dev_error=0.406378, test_error=0.413494, learning_rate=0.005000, execution_time(s)=78.196592
epoch 23 training_cost=0.970285, training_error=0.305204, dev_error=0.405064, test_error=0.412578, learning_rate=0.002500, execution_time(s)=80.574895
epoch 24 training_cost=0.968486, training_error=0.304418, dev_error=0.406182, test_error=0.412734, learning_rate=0.002500, execution_time(s)=78.166218
``` 
#### 4. Kaldi Decoding.
During the last epoch, the training script creates  a file *pout_test.ark* containing a set of likelihoods (i.e., normalized posterior probabilities) computed on the test sentences. These likelihoods can be used to feed the Kaldi decoder in this way:
``` 
cd kaldi_decoding_scripts
``` 
For mfcc features:

``` 
 ./decode_dnn_TIMIT.sh $KALDI_ROOT/egs/timit/s5/exp/tri3/graph \
 $KALDI_ROOT/egs/timit/s5/data/test/ \
 $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \
 ../TIMIT_MLP_mfcc/decoding_test \
 "cat ../TIMIT_MLP_mfcc/pout_test.ark"
``` 

For fMLLR features
``` 
 ./decode_dnn_TIMIT.sh $KALDI_ROOT/egs/timit/s5/exp/tri3/graph \
 $KALDI_ROOT/egs/timit/s5/data/test/ \
 $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \
 ../TIMIT_MLP_fmllr/decoding_test \
 "cat ../TIMIT_MLP_fmllr/pout_test.ark"
``` 

#### 5. Check the results.
- After that training and decoding phases are finished, you can go into the *pytorch_MLP_for_ASR* folder and run *./RESULTS* to check the system performance.  
 
If everything if fine, you should obtain  *Phone Error Rates (PER%)* similar to the following ones:

- mfcc features: PER=18.0%
- fMLLR features: **PER=16.8%**

For reference purposes, you can take a look to our results here: *TIMIT_MLP_fmllr_reference* or *TIMIT_MLP_mfcc_reference*.

Note that, despite its simplicity, the performance obtained with this implementation is slightly better than that achieved with the kaldi baselines (even without pre-training or sMBR).  For comparison purposes, see for instance the file *$KALDI_ROOT/egs/timit/s5/RESULTS*. 

Note also that small variations of PER with respect to these reference values (e.g, +/- 0.5 %) are normal (since due to a different DNN initialization). 


## Reference:
Please, cite my PhD thesis if you use this code:

*[1] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017*

https://arxiv.org/abs/1712.06086
