"""
Created on Wed Oct 31 12:14:44 2018

@author: liuwenzhong
"""
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl 
from torch.autograd import Variable 
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
 
 
torch.nn.Module.dump_patches = True    
  
class SICDm6APredict(object): 
    def __init__(self, modelpklfile, threshold, thresholdlevel,fullseqlen,findseqlen, Basestr, featlen):
        
        self.seq_dict_test=dict()
        self.seq_label_test=dict()
        self.testlist_dict=dict()
        self.thresholdlevel= thresholdlevel
        self.threekmercode=dict()
        self.modelpklfile=modelpklfile
        self.threshold=threshold
        self.fullseqlen=fullseqlen
        self.findseqlen=findseqlen
        self.Basestr=Basestr
        self.featlen=featlen
      
     
     
    def ROC( self,y_test,y_predicted):
        y = y_test#np.array(y_test)
        pred =y_predicted# np.array(y_predicted)
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        #print(fpr)
        #print(tpr)
        #print(thresholds)
        v_auc=auc(fpr, tpr)
        print(v_auc)
        #x = [_v[0] for _v in xy_arr]
        #y = [_v[1] for _v in xy_arr]
        pl.title("ROC curve of %s (AUC = %.4f)" % ('M6A' , v_auc))
        pl.xlabel("False Positive Rate")
        pl.ylabel("True Positive Rate")
        pl.plot( fpr ,tpr)
        pl.show()
        return v_auc
    
    

    
    

    def buildthreekmercode(self):
 
        encode=dict()
   
        len1=len(self.Basestr )

        for i in range(len1):
            s1=self.Basestr[i:i+1]
            for j in range(len1):            
                s2=self.Basestr[j:j+1]
                for k in range(len1):
                    s3=self.Basestr[k:k+1]
                    ss=s1+s2+s3
                    kk=int(str(i)+str(j)+str(k))
                    encode[ss]=kk

        return encode
     
    
     



    def predict(self,seq_dict_test , seq_label_test  ):
        #print(seq_dict_test)
 
        self.threekmercode=self.buildthreekmercode()
         
        img_test=SeqDataset(seq_dict_test , seq_label_test,  self.fullseqlen, self.findseqlen, self.threekmercode )
     
        test_loader= DataLoader(dataset= img_test, batch_size=110, shuffle=False)
        
        model =SICNet(self.featlen,self.findseqlen)  #model.load_state_dict(torch.load(args.weight,map_location=map_location)) 
        #print(model. w_2_2)
        #model_dict = model.state_dict()
        #model  =torch.load(self.modelpklfile )
        if torch.cuda.is_available():
            model   =torch.load(self.modelpklfile,map_location=lambda storage, loc: storage.cuda(0))
        else:
            model =torch.load(self.modelpklfile,map_location='cpu')
        #model. load_state_dict(model_pre  )
        
        if torch.cuda.is_available():
            model. cuda()
        #print(model. w_2_2)
        #model.load_state_dict(torch.load(self.modelpklfile,map_location='cpu'))
        #model=torch.load(self.modelpklfile,map_location='cpu')
        
        
        #pretrained_dict = {k: v for k, v in model_pretrain.items() if k in model_dict}
        #取出预训练模型中与新模型的dict中重合的部分
        
        #model_dict.update(pretrained_dict)#用预训练模型参数更新new_model中的部分参数
        #model.load_state_dict(model_dict) #将更新后的model_dict加载进new model中
        # evaluation--------------------------------
       

        model .eval()
        model.zero_grad()
     
        y_predicted_com =[]
     
        for batch_x, batch_y in test_loader:
            len1=len( batch_y)      
            if torch.cuda.is_available():
                batch_x= batch_x.cuda()  
                out  = model (batch_x )            
                y_predicted= out.cpu() .detach().numpy() [:,1]
            else:
                out  = model (batch_x )            
                y_predicted= out. detach().numpy() [:,1]     
                
            y_predicted=y_predicted.reshape((len1,1))
            #print(y_predicted )
            if len(y_predicted_com)<=0:
                y_predicted_com=y_predicted
            else:
                y_predicted_com=np.vstack((y_predicted_com,y_predicted)  ) 
        
        #print(y_predicted_com)
        score_predict=np.asarray(y_predicted_com)
        #print( score_predict)
        #print(type(self.threshold))
        label_predict=np.int64(1*(score_predict>=self.threshold))  
         
        #build prediction string
        predictionstring ="Result\n\n** Threshhold =  "+str(self.threshold)+" , while specificity is "+self.thresholdlevel+".\n** Start position is 1. A site is 6mA when the predicted_label is 1, otherwise it is a non-6mA site．\n\n"
        predictionstring+="name_position\tshort_sequence\tscore\tpredicted_label\n"
        kk=0
        for key in seq_dict_test:
            seq=seq_dict_test[key]
            predictionstring +=key+"\t"+seq+"\t"+str(score_predict[kk][0])+"\t"+str(label_predict[kk][0])+"\n"
            kk+=1
            
        return predictionstring
        
      
       

    
class SeqDataset(Dataset):
 
    def __init__(self, seq_dict , seq_label ,seqlen,findseqlen, threekmercode ): 
        self.seqlen=seqlen
        self.findseqlen=findseqlen
        self.threekmercode= threekmercode
        self.imgs = self.buildseqcode( seq_dict , seq_label   ) 

    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    def buildseqcode( self,seq_dict , seq_label   ):     
        imgs = []
       
        for key in  seq_dict.keys():
            seq= seq_dict[key]  .replace("N","-").replace("n","-")     
            mat= self.buildseqqtbin( seq)         
            label=seq_label[key]
            imgs.append((mat,label))
 
        return imgs
     
    def buildseqqtbin(self, seq):
        seq=seq.strip()
        lefthalf=int((self.seqlen-self.findseqlen)/2)
        righthalf=self.seqlen-lefthalf 
        seq=seq[lefthalf:righthalf]  
        len1=len(seq)
        mat = np.zeros((self.findseqlen ), dtype=np.longlong) 
        i=0      
        while i <len1 :#3-mer  
       
            basestr=""
            if i<len1-2:
                basestr=seq[i:i+3 ]
            elif i==len1-2:
                basestr=seq[i:i+2 ]+"-"
            else:
                basestr=seq[i:i+1 ]+"--"
            bin_k=self.threekmercode[basestr]    
            mat[i]=bin_k       
            i +=1

        return mat  
    
  
 
class SICNet(nn.Module):

   def __init__(self,featlen,findseqlen):   
        super(SICNet, self).__init__()
        self.featlen=featlen
        self.findseqlen=findseqlen
        self.embed_3_1=nn.Embedding(1024,self.featlen)
 
        self.LSTM3_1 =nn.GRU(input_size=self.featlen,hidden_size=self.featlen ,bidirectional=True,
                     num_layers=2,  dropout=0.5, batch_first=True   )  
        
        self.LSTM3_2 =nn.GRU(input_size=self.findseqlen,hidden_size=128 , 
                             num_layers=2,  dropout=0.5, batch_first=True    )  
  
          
        self.fc_1= nn.Sequential( nn.Linear( 128   ,16) , 
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.BatchNorm1d(16, momentum=0.5),
                                 nn.Dropout(0.25),

                                   )
        self.fc_2= nn.Sequential( nn.Linear(16,2) ) 
  
    
        self.w_1_1 = Parameter(Tensor(1, self.findseqlen).float().cuda())
        
        self.w_1_2 = Parameter(Tensor(1, self.findseqlen).float().cuda())
        
        self.w_2_1 = Parameter(Tensor(1,self.featlen*2).float().cuda())
        self.w_2_2  = Parameter(Tensor(1,self.featlen*2).float().cuda())
       
        self.reset_weigths()

   def reset_weigths(self):
        stdv1 = 1.0 / math.sqrt(self.findseqlen)
        stdv2 = 1.0 / math.sqrt(self.featlen*2 )    
         
        init.uniform_(self.w_1_1, -stdv1, stdv1)
        init.uniform_(self.w_1_2, -stdv1, stdv1)
        
        init.uniform_(self.w_2_1, -stdv2, stdv2)
        init.uniform_(self.w_2_2, -stdv2, stdv2)
        
 
       
   def forward(self, x):
       btnum1=x.shape[0]
       
       x_3_1=self.embed_3_1 (x)

       x_3_1,_=self.LSTM3_1( x_3_1) 
       x_3_1= x_3_1.contiguous() 
  
       x_3_2=  x_3_1.permute(0,2,1)
       x_3_2,_ =self.LSTM3_2( x_3_2 )
     
       x_3_2=x_3_2.contiguous() 
       
  
       dd2=torch.zeros (btnum1, 128 ).float() 
       if torch.cuda.is_available():
           dd2=dd2.cuda()
       for i in range(btnum1):
           xb2=x_3_2[i,:,:]  
       
           sigma2=F.sigmoid(torch. mm(self.w_2_1, xb2)     )          
           mem2= F. tanh(torch. mm(self.w_2_2 ,  xb2)  ) 
           ms2=torch.mul(sigma2,mem2)  
           #if torch.cuda.is_available():
           #    ms2.cuda()
          
           dd2[i,:]= ms2.squeeze(0).float() 
           

           
       x_5=self.fc_1(dd2     )
       x_5=self.fc_2(x_5)
      
        

        
       out=F.log_softmax(x_5,dim=1)
      
       return out    
