# pytorch_speechMLP 
# Mirco Ravanelli 
# Montreal Institute for Learning Algoritms (MILA)
# University of Montreal 

# January 2018

# Description: 
# This code implements with pytorch a basic MLP  for speech recognition. 
# It exploits an interface to  kaldi for feature computation and decoding. 
# How to run it:
# python MLP_speech.py --cfg TIMIT_MLP.cfg

import kaldi_io
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import timeit
import torch.optim as optim
import os
from shutil import copyfile
from data_io import load_chunk,load_counts,read_opts


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, N_hid, drop_rate,use_batchnorm,num_classes):
        super(MLP, self).__init__()
        
        # list initialization
        self.hidden  = nn.ModuleList([])
        self.droplay  = nn.ModuleList([])
        self.bnlay  = nn.ModuleList([])
        self.criterion = nn.CrossEntropyLoss()
        
        curr_in_dim=input_dim
        for i in range(N_hid):
         fc = nn.Linear(input_dim, hidden_dim) 
         fc.weight = torch.nn.Parameter(torch.Tensor(hidden_dim,curr_in_dim).uniform_(-np.sqrt(0.01/(curr_in_dim+hidden_dim)),np.sqrt(0.01/(curr_in_dim+hidden_dim))))
         fc.bias = torch.nn.Parameter(torch.zeros(hidden_dim)) 
         curr_in_dim=hidden_dim
         self.hidden.append(fc)
         self.droplay.append(nn.Dropout(p=drop_rate))
         if use_batchnorm:
           self.bnlay.append(nn.BatchNorm1d(hidden_dim,momentum=0.05))

        
        self.fco = nn.Linear(curr_in_dim, num_classes)
        self.fco.weight = torch.nn.Parameter(torch.zeros(num_classes,curr_in_dim))
        self.fco.bias = torch.nn.Parameter(torch.zeros(num_classes))
        
        
    
    def forward(self, x,lab):
        out=x      
        for i in range(N_hid):
         fc=self.hidden[i]
         drop=self.droplay[i]
         
         if use_batchnorm:
          batchnorm=self.bnlay[i]
          out = drop(F.relu(batchnorm(fc(out))))
         else:
          out = drop(F.relu(fc(out)))
        
        out = self.fco(out)
        pout=F.log_softmax(out,dim=1)
        pred=pout.max(dim=1)[1]
        err =  sum((pred!=lab.long()).float())
        loss = self.criterion(out, lab.long()) # note that softmax is included in nn.CrossEntropyLoss()
        return [loss,err,pout,pred]

    

# Reading options in cfg file
options=read_opts()
 
# Reading training data options
tr_fea_scp=options.tr_fea_scp.split(',')
tr_fea_opts=options.tr_fea_opts
tr_lab_folder=options.tr_lab_folder
tr_lab_opts=options.tr_lab_opts

# Reading dev data options
dev_fea_scp=options.dev_fea_scp
dev_fea_opts=options.dev_fea_opts
dev_lab_folder=options.dev_lab_folder
dev_lab_opts=options.dev_lab_opts

# Reading test data options
te_fea_scp=options.te_fea_scp
te_fea_opts=options.te_fea_opts
te_lab_folder=options.te_lab_folder
te_lab_opts=options.te_lab_opts

# Reading count file from kaldi
count_file=options.count_file

# reading architectural options
left=int(options.cw_left)
right=int(options.cw_right)
hidden_dim=int(options.hidden_dim)
N_hid=int(options.N_hid)
drop_rate=float(options.drop_rate)
use_batchnorm=bool(int(options.use_batchnorm))
seed=int(options.seed)
use_cuda=bool(int(options.use_cuda))

# reading optimization options
N_ep=int(options.N_ep)
batch_size=int(options.batch_size)
lr=float(options.lr)
halving_factor=float(options.halving_factor)
improvement_threshold=float(options.improvement_threshold)
save_gpumem=int(options.save_gpumem)

# Create output folder
if not os.path.exists(options.out_folder):
    os.makedirs(options.out_folder)

# copy cfg file into the output folder
copyfile(options.cfg, options.out_folder+'/conf.cfg')    

# Setting torch seed
torch.manual_seed(seed)

# Creating the res file 
res_file = open(options.out_folder+'/results.res',"w")

for ep in range(1,N_ep+1):
 
 # ---TRAINING LOOP---#
 err_sum=0.0
 loss_sum=0.0
 n_train_batches_tot=0
 N_ex_tr_tot=0
 start_epoch=timeit.default_timer()
 
 # Processing training chunks
 for chunk in range(len(tr_fea_scp)):   
   
  seed=seed+100
  
  # Reading training chunk
  [train_name,tr_set,tr_end_index_tr]=load_chunk(tr_fea_scp[chunk],tr_fea_opts,tr_lab_folder,tr_lab_opts,left,right,seed)

  if not(save_gpumem):
   tr_set=torch.from_numpy(tr_set).float().cuda()
  else:
    tr_set=torch.from_numpy(tr_set).float()  
  
  # Computing training examples and batches
  N_ex_tr=tr_set.shape[0]
  N_ex_tr_tot=N_ex_tr_tot+N_ex_tr
  
  n_train_batches=N_ex_tr // batch_size
  n_train_batches_tot=n_train_batches_tot+n_train_batches
  
  beg_batch=0
  end_batch=batch_size
 
  if ep==1 and chunk==0:
    # Initialization of the MLP
    N_fea=tr_set.shape[1]-1
    N_out=int(tr_set[:,N_fea].max()-tr_set[:,N_fea].min()+1) 
    net = MLP(N_fea, hidden_dim, N_hid,drop_rate,use_batchnorm,N_out)
    
    # Loading model into the cuda device
    if use_cuda:
     net.cuda() 
       
    # Optimizer initialization
    optimizer = optim.SGD(net.parameters(), lr=lr) 
   
    # Loading Dev data
    [dev_name,dev_set,dev_end_index]=load_chunk(dev_fea_scp,dev_fea_opts,dev_lab_folder,dev_lab_opts,left,right,-1)
    
    if not(save_gpumem):
     dev_set=torch.from_numpy(dev_set).float().cuda()
    else:
     dev_set=torch.from_numpy(dev_set).float()
     
    # Loading Test data
    [te_name,te_set,te_end_index]=load_chunk(te_fea_scp,te_fea_opts,te_lab_folder,te_lab_opts,left,right,-1)

    if not(save_gpumem):
     te_set=torch.from_numpy(te_set).float().cuda()
    else:
      te_set=torch.from_numpy(te_set).float()
      
  net.train()    
  # Processing trainibg batches
  for i in range(n_train_batches):
   
   # features and labels for batch i
   inp= Variable(tr_set[beg_batch:end_batch,0:N_fea])
   lab= Variable(tr_set[beg_batch:end_batch,N_fea])
   
   if save_gpumem and use_cuda:
       inp=inp.cuda()
       lab=lab.cuda()
       
   # free the gradient buffer
   optimizer.zero_grad()  
   
   # Forward phase
   [loss,err,pout,pred] = net(inp,lab)
   
   # Gradient computation
   loss.backward()
  
   # updating parameters
   optimizer.step()
   
   # Loss accumulation 
   loss_sum=loss_sum+loss.data
   err_sum=err_sum+err.data
   
   # update it to the next batch 
   beg_batch=end_batch
   end_batch=beg_batch+batch_size
    
  del train_name,tr_set,tr_end_index_tr

 # Average Loss
 loss_tr=loss_sum/n_train_batches_tot
 err_tr=err_sum/N_ex_tr_tot
 
 end_epoch=timeit.default_timer() 
 
 
 
 # ---EVALUATION OF DEV---#
 beg_snt=0
 err_sum=0.0
 loss_sum=0.0
 n_dev_snt=len(dev_name)
 net.eval()  
 # Reading dev-set sentence by sentence
 for i in range(n_dev_snt):
  
  end_snt=dev_end_index[i]
  inp= Variable(dev_set[beg_snt:end_snt,0:N_fea],volatile=True)
  lab= Variable(dev_set[beg_snt:end_snt,N_fea],volatile=True)
  
  if save_gpumem and use_cuda:
     inp=inp.cuda()
     lab=lab.cuda()
  
  [loss,err,pout,pred] = net(inp,lab)
  loss_sum=loss_sum+loss.data
  err_sum=err_sum+err.data
 
  beg_snt=dev_end_index[i]
  
 loss_dev=loss_sum/n_dev_snt
 err_dev=(err_sum/dev_set.shape[0]).cpu().numpy()
 
 # Learning rate annealing (if improvement on dev-set is small)
 lr_ep=lr
 if ep==1:
  err_dev_prev=err_dev
 else:
   err_dev_new=err_dev
   if ((err_dev_prev-err_dev_new)/err_dev_new)< improvement_threshold:
     lr_ep=lr
     lr=lr*halving_factor
   err_dev_prev=err_dev_new
   
 # set the next epoch learning rate 
 for param_group in optimizer.param_groups:
    param_group['lr'] = lr
   



 # ---EVALUATION OF TEST---#
 beg_snt=0
 err_sum=0.0
 loss_sum=0.0
 n_te_snt=len(te_name)
 net.eval()  
 
 if ep==N_ep:
  # set folder for posteriors ark    
  post_file=kaldi_io.open_or_fd(options.out_folder+'/pout_test.ark','wb')
  counts = load_counts(count_file)

 for i in range(n_te_snt):
  
  end_snt=te_end_index[i]
  inp= Variable(te_set[beg_snt:end_snt,0:N_fea],volatile=True)
  lab= Variable(te_set[beg_snt:end_snt,N_fea],volatile=True)
  
  if save_gpumem and use_cuda :
     inp=inp.cuda()
     lab=lab.cuda()
  
  [loss,err,pout,pred] = net(inp,lab)
  
  if ep==N_ep:
   # writing the ark containing the normalized posterior probabilities (needed for kaldi decoding)
   kaldi_io.write_mat(post_file, pout.data.cpu().numpy()-np.log(counts/np.sum(counts)), te_name[i])
  
  loss_sum=loss_sum+loss.data
  err_sum=err_sum+err.data
 
  beg_snt=te_end_index[i]
  
 loss_te=loss_sum/n_te_snt
 err_te=err_sum/te_set.shape[0]
 

 
 print('epoch %i training_cost=%f, training_error=%f, dev_error=%f, test_error=%f, learning_rate=%f, execution_time(s)=%f' %(ep,loss_tr,err_tr,err_dev,err_te,lr_ep,end_epoch-start_epoch))
 res_file.write('epoch %i training_cost=%f, training_error=%f, dev_error=%f, test_error=%f, learning_rate=%f, execution_time(s)=%f\n' %(ep,loss_tr,err_tr,err_dev,err_te,lr_ep,end_epoch-start_epoch))
 
post_file.close()
res_file .close()
# Model Saving
torch.save(net.state_dict(), options.out_folder+'/model.pkl')


# If everything went fine, you can run the kaldi phone-loop decoder:
# cd kaldi_decoding_scripts
#./decode_dnn_TIMIT.sh /home/mirco/kaldi-trunk/egs/timit/s5/exp/tri3/graph /home/mirco/kaldi-trunk/egs/timit/s5/data/test/ /home/mirco/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali /home/mirco/pytorch_exp/TIMIT_MLP_fmllr/decoding_test cat /home/mirco/pytorch_exp/TIMIT_MLP_fmllr/pout_test.ark


  
  
  
  
  
 
