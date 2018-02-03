## Introduction:
This codes implements a basic *MLP* for *HMM-DNN* speech recognition. The MLP is trained with *pytorch*, while feature extraction, alignments, and decoding are performed with *Kaldi*. The current implementation supports dropout and batch normalization. An example for phoneme recognition using the standard TIMIT dataset is provided.
 
## Prerequisites:
 - Make sure that python is installed (the code is tested with python 2.7). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

- If not already done, install pytorch (http://pytorch.org/) and make sure that the installation works. As a first test, type “python” and, once entered into the console, type “import torch”. Make sure everything is fine. 

- If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into the *.bashrc file*. As a first test to check the installation, open  a bash shell, type “copy-feats” and make sure no errors appear.

- Install kaldi-io package from the kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python). It provides a simple interface between kaldi and python. To install it:

1) run git clone https://github.com/vesis84/kaldi-io-for-python.git <kaldi-io-dir> 
2) add PYTHONPATH=${PYTHONPATH}:<kaldi-io-dir> to $HOME/.bashrc 
3) now type *import kaldi_io* from the python console and make sure the package is correctly imported.  You can find more info (including some reading and writing tests) on  https://github.com/vesis84/kaldi-io-for-python

 
The code has been tested with:
- Python  2.7 
- Ubuntu 17.04
- Pytorch 0.3
 
## How to run a TIMIT experiment:

#### 1. Run the Kaldi s5 baseline of TIMIT.  
This step is necessary to  derive features and labels later used to train the MLP.  In particular: 
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
1) *tr_fea_scp* contains a list of the list files created with *create_chunks.sh*. 
2) *tr_fea_opts* allows users to easily add normalizations, derivatives and other types of feature processing  (see for instance *TIMIT_MLP_mfcc.cfg*). 
3) *tr_lab_folder* is the kaldi folder containing the alignments (labels).
4) *tr_lab_opts* allows users to derive context-dependent phone targets (when set to *ali-to-pdf*) or monophone targets (when set to *ali-to-phones --per-frame*).
5) Modify the paths for dev and test data.
6) Feel free to modify the DNN architecture and the other optimization parameters according to your needs. 
7) The required *count_file* in the config file (used to normalize the DNN posteriors before feeding the decoder) corresponds to the following file: *$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts* (that is automatically created by Kaldi when running the TIMIT s5 recipe).
8) Use the option *use_cuda=1* for running the code on a GPU (strongly suggested).
9) Use the option *save_gpumem=0* to save gpu memory. The code will be a little bit slower (about 10-15%), but it saves gpu memory. Use *save_gpumem=1* only if your GPU has more that 2GB of memory. 

 
#### 4. Train the DNN. 
- To run DNN training type:
```
python MLP_ASR.py --cfg TIMIT_MLP_mfcc.cfg 2> log.log
``` 
or 
```
python MLP_ASR.py --cfg TIMIT_MLP_fmllr.cfg 2> log.log
``` 

Note that training process might take from 30 minutes to 1 hours to finish. 
If everything is working fine, your output (for fMLLR features) should look like this:
``` 
epoch 1 training_cost=3.384317, training_error=0.721703, dev_error=0.593744, test_error=0.590824, learning_rate=0.080000, execution_time(s)=84.179209
epoch 2 training_cost=2.142425, training_error=0.572163, dev_error=0.543180, test_error=0.542955, learning_rate=0.080000, execution_time(s)=77.895830
epoch 3 training_cost=1.898246, training_error=0.524813, dev_error=0.519763, test_error=0.525082, learning_rate=0.080000, execution_time(s)=78.218724
epoch 4 training_cost=1.752852, training_error=0.495105, dev_error=0.505801, test_error=0.512097, learning_rate=0.080000, execution_time(s)=77.818596
epoch 5 training_cost=1.646851, training_error=0.472181, dev_error=0.503417, test_error=0.505293, learning_rate=0.080000, execution_time(s)=77.857225
epoch 6 training_cost=1.561680, training_error=0.453188, dev_error=0.493636, test_error=0.498075, learning_rate=0.080000, execution_time(s)=78.049902
epoch 7 training_cost=1.489096, training_error=0.436349, dev_error=0.487659, test_error=0.493153, learning_rate=0.080000, execution_time(s)=78.819512
epoch 8 training_cost=1.428991, training_error=0.422880, dev_error=0.484532, test_error=0.492272, learning_rate=0.080000, execution_time(s)=78.206443
epoch 9 training_cost=1.374744, training_error=0.409588, dev_error=0.482670, test_error=0.491426, learning_rate=0.080000, execution_time(s)=78.222811
epoch 10 training_cost=1.327537, training_error=0.398480, dev_error=0.477404, test_error=0.484519, learning_rate=0.080000, execution_time(s)=78.109824
epoch 11 training_cost=1.285530, training_error=0.388687, dev_error=0.479894, test_error=0.486867, learning_rate=0.080000, execution_time(s)=78.121776
epoch 12 training_cost=1.185874, training_error=0.362461, dev_error=0.463589, test_error=0.471257, learning_rate=0.040000, execution_time(s)=78.223371
epoch 13 training_cost=1.141434, training_error=0.351222, dev_error=0.463875, test_error=0.473830, learning_rate=0.040000, execution_time(s)=78.133419
epoch 14 training_cost=1.090926, training_error=0.336912, dev_error=0.456567, test_error=0.463417, learning_rate=0.020000, execution_time(s)=79.801834
epoch 15 training_cost=1.062440, training_error=0.329381, dev_error=0.456584, test_error=0.464297, learning_rate=0.020000, execution_time(s)=77.320703
epoch 16 training_cost=1.036693, training_error=0.322880, dev_error=0.453334, test_error=0.461120, learning_rate=0.010000, execution_time(s)=77.257489
epoch 17 training_cost=1.019714, training_error=0.317592, dev_error=0.453555, test_error=0.460395, learning_rate=0.010000, execution_time(s)=77.273143
epoch 18 training_cost=1.008937, training_error=0.315300, dev_error=0.451064, test_error=0.459048, learning_rate=0.005000, execution_time(s)=78.162366
epoch 19 training_cost=0.998883, training_error=0.311941, dev_error=0.450746, test_error=0.459963, learning_rate=0.005000, execution_time(s)=78.229541
epoch 20 training_cost=0.991998, training_error=0.310614, dev_error=0.450027, test_error=0.458771, learning_rate=0.002500, execution_time(s)=78.084323
epoch 21 training_cost=0.987904, training_error=0.308987, dev_error=0.449358, test_error=0.457304, learning_rate=0.002500, execution_time(s)=78.315039
epoch 22 training_cost=0.985269, training_error=0.308505, dev_error=0.448999, test_error=0.458305, learning_rate=0.002500, execution_time(s)=77.315663
epoch 23 training_cost=0.982597, training_error=0.307779, dev_error=0.448672, test_error=0.457943, learning_rate=0.001250, execution_time(s)=77.309540
epoch 24 training_cost=0.978345, training_error=0.306505, dev_error=0.447864, test_error=0.457114, learning_rate=0.000625, execution_time(s)=78.290923
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
- After that training and decoding phases are finished, you can go into the *kaldi_decoding_scripts* folder and run *./RESULTS* to check the system performance.  
 
- If everything if fine, you should obtain  Phone Error Rates (PER%) similar to the following ones:
mfcc features: PER=18.7%
fMLLR features: PER=16.7%

Note that, despite its simplicity, the performance obtained with this implementation is slightly better than that achieved with the kaldi baselines (even without pre-training or sMBR).  For comparison purposes, see for instance the file *$KALDI_ROOT/egs/timit/s5/RESULTS*.


## Reference:
Please, cite my PhD thesis if you use this code:

*[1] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017*

https://arxiv.org/abs/1712.06086
