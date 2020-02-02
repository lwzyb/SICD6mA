import wx
import wx.xrc
import os 
from SICDm6A_Predict import SICDm6APredict,SICNet
from SICDm6A_Result import SICResultFrame 
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl
import re 
import os 

Human_modelpklfile=""
Rice_modelpklfile=""
Mouse_modelpklfile=""
 
"""
Genome	95%	90%	80%
 
"""
Human_threshold_high=-0.86118
Human_threshold_midum=-1.76532
Human_threshold_low=-2.92709

Rice_threshold_high=-0.64277
Rice_threshold_midum=-2.26020
Rice_threshold_low=-3.95734

 

# 
motif="[aA]"
p_motif=re.compile(motif)       
    
def readconfig():
    global Human_modelpklfile
    global Rice_modelpklfile
    global Mouse_modelpklfile 
    
    global Human_threshold_high
    global Human_threshold_medum
    global Human_threshold_low
    
    global Rice_threshold_high
    global Rice_threshold_medum
    global Rice_threshold_low

    global Mouse_threshold_high
    global Mouse_threshold_medum
    global Mouse_threshold_low

 
    
    
    
    fh=open(os.getcwd()+"/config.txt")
     
    for line in fh:
        if line.startswith('Human_modelpklfile='):
            Human_modelpklfile=os.getcwd()+"/"+line.replace('Human_modelpklfile=',"").replace('\n',"")
        elif line.startswith('Rice_modelpklfile='):
            Rice_modelpklfile=os.getcwd()+"/"+line.replace('Rice_modelpklfile=',"").replace('\n',"")
 
        elif line.startswith('Human_threshold_high='):
           Human_threshold_high= float(line.replace('Human_threshold_high=',"").replace('\n',"")    )   
        elif line.startswith('Human_threshold_medum='):
           Human_threshold_medum= float(line.replace('Human_threshold_medum=',"").replace('\n',""))
        elif line.startswith('Human_threshold_low='):
           Human_threshold_low=float(line.replace('Human_threshold_low=',"").replace('\n',"") )
        elif line.startswith('Rice_threshold_high='):
           Rice_threshold_high=float(line.replace('Rice_threshold_high=',"").replace('\n',"") )
        elif line.startswith('Rice_threshold_medum='):
           Rice_threshold_medum=float(line.replace('Rice_threshold_medum=',"").replace('\n',"") )    
        elif line.startswith('Rice_threshold_low='):
           Rice_threshold_low=float(line.replace('Rice_threshold_low=',"").replace('\n',"") )
     
             
    fh.close()
    

class SICFrame ( wx.Frame ):
    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"SICD6mA", pos = wx.DefaultPosition, size = wx.Size( 817,790 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        ######init##############
        self.threshold=Human_threshold_high
        self.thresholdlevel='95%'
        self.fullseqlen=41
        self.findseqlen=41
        self.featlen=125
        self.Basestr="-AGCT"
        self.modefile=""
        ##############	
        self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
        fgSizer1 = wx.FlexGridSizer( 5, 1, 0, 0 )
        fgSizer1.SetFlexibleDirection( wx.BOTH ) 
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        fgSizer1.SetMinSize( wx.Size( 800,600 ) ) 
        self.m_panel16 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        fgSizer12 = wx.FlexGridSizer( 2, 1, 0, 0 )
        fgSizer12.SetFlexibleDirection( wx.BOTH )
        fgSizer12.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_panel1 = wx.Panel( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel1.SetMinSize( wx.Size( 800,50 ) )
        
        gSizer1 = wx.GridSizer( 1, 1, 0, 0 )
        
        self.m_staticText2 = wx.StaticText( self.m_panel1, wx.ID_ANY, u"SICD6mA for Predicting 6mA Sites ", wx.DefaultPosition, wx.Size( 800,45 ), wx.ALIGN_CENTRE )
        self.m_staticText2.Wrap( -1 )
        self.m_staticText2.SetFont( wx.Font( 16, 70, 90, 92, False, "Times New Roman" ) )
        
        gSizer1.Add( self.m_staticText2, 0, wx.ALL, 5 )
        
        self.m_panel1.SetSizer( gSizer1 )
        
        self.m_panel1.Layout()
        gSizer1.Fit( self.m_panel1 )
        fgSizer12.Add( self.m_panel1, 1, wx.EXPAND, 5 )
        
        self.m_panel2 = wx.Panel( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,30 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        fgSizer2 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer2.SetFlexibleDirection( wx.BOTH )
        fgSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        fgSizer2.SetMinSize( wx.Size( 800,20 ) ) 
        self.m_staticText4 = wx.StaticText( self.m_panel2, wx.ID_ANY, u"Input your genomic sequences with FASTA format.", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4.Wrap( -1 )
        self.m_staticText4.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer2.Add( self.m_staticText4, 0, wx.ALL, 5 )
        
        self.m_panel2.SetSizer( fgSizer2 )
        self.m_panel2.Layout()
        fgSizer12.Add( self.m_panel2, 1, wx.EXPAND, 5 )
        
        self.m_panel16.SetSizer( fgSizer12 )
        self.m_panel16.Layout()
        fgSizer12.Fit( self.m_panel16 )
        fgSizer1.Add( self.m_panel16, 1, wx.EXPAND, 5 )
        
        self.m_panel3 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,500 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel3.SetFont( wx.Font( 10, 70, 94, 92, False, "Times New Roman" ) )
        self.m_panel3.SetMinSize( wx.Size( 800,500 ) )
        
        fgSizer4 = wx.FlexGridSizer( 0, 3, 0, 0 )
        fgSizer4.SetFlexibleDirection( wx.BOTH )
        fgSizer4.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_txtseq = wx.TextCtrl( self.m_panel3, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 800,500 ), wx.TE_LEFT|wx.TE_MULTILINE )
        fgSizer4.Add( self.m_txtseq, 0, wx.ALL, 5 )
        
        self.m_panel3.SetSizer( fgSizer4 )
        self.m_panel3.Layout()
        fgSizer1.Add( self.m_panel3, 1, wx.EXPAND, 5 )
        
        self.m_panel4 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel4.SetMinSize( wx.Size( 800,50 ) )
        
        fgSizer7 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer7.SetFlexibleDirection( wx.BOTH )
        fgSizer7.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_staticText21 = wx.StaticText( self.m_panel4, wx.ID_ANY, u"Or upload a file:", wx.DefaultPosition, wx.Size( 150,30 ), wx.ALIGN_LEFT )
        self.m_staticText21.Wrap( -1 )
        self.m_staticText21.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer7.Add( self.m_staticText21, 0, wx.ALL, 5 )#wx.EmptyString
        
        self.m_loadfile = wx.FilePickerCtrl( self.m_panel4, wx.ID_ANY,wx.EmptyString , u"Select a file", u"*.*", wx.DefaultPosition, wx.Size( 550,-1 ), wx.FLP_DEFAULT_STYLE|wx.FLP_FILE_MUST_EXIST|wx.FLP_OPEN )
        
        self.m_loadfile.SetFont( wx.Font( 10, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer7.Add( self.m_loadfile, 0, wx.ALL, 5 )
        
        self.m_panel4.SetSizer( fgSizer7 )
        self.m_panel4.Layout()
        fgSizer7.Fit( self.m_panel4 )
        fgSizer1.Add( self.m_panel4, 1, wx.EXPAND, 5 )
        
        self.m_panel5 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,80 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel5.SetFont( wx.Font( 14, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41 = wx.FlexGridSizer( 1, 2, 0, 0 )
        fgSizer41.SetFlexibleDirection( wx.BOTH )
        fgSizer41.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        m_rb_modeChoices = [ u"Human", u"Rice_Nip"  ]
        self.m_rb_mode = wx.RadioBox( self.m_panel5, wx.ID_ANY, u"Mode", wx.DefaultPosition, wx.Size( 400,80 ), m_rb_modeChoices, 1, wx.RA_SPECIFY_ROWS )
        self.m_rb_mode.SetSelection( 0 )
        self.m_rb_mode.SetFont( wx.Font( 10, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41.Add( self.m_rb_mode, 0, wx.ALL|wx.EXPAND, 5 )
        
        m_rb_threshholdChoices = [ u"High(95%)", u"Medum(90%)", u"Low(80%)" ]
        self.m_rb_threshhold = wx.RadioBox( self.m_panel5, wx.ID_ANY, u"Threshhold(Specificity)", wx.DefaultPosition, wx.Size( 400,80 ), m_rb_threshholdChoices, 1, wx.RA_SPECIFY_ROWS )
        self.m_rb_threshhold.SetSelection( 0 )
        self.m_rb_threshhold.SetFont( wx.Font( 10, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41.Add( self.m_rb_threshhold, 0, wx.ALL|wx.EXPAND, 5 )
        
        self.m_panel5.SetSizer( fgSizer41 )
        self.m_panel5.Layout()
        fgSizer1.Add( self.m_panel5, 1, wx.EXPAND, 5 )
        
        self.m_panel111 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,40 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        fgSizer71 = wx.FlexGridSizer( 1, 5, 0, 0 )
        fgSizer71.SetFlexibleDirection( wx.BOTH )
        fgSizer71.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_staticText5 = wx.StaticText( self.m_panel111, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 150,-1 ), 0 )
        self.m_staticText5.Wrap( -1 )
        fgSizer71.Add( self.m_staticText5, 0, wx.ALL, 5 )
        
        self.m_bt_submit = wx.Button( self.m_panel111, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_submit.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_submit, 0, wx.ALL, 5 )
        
        self.m_bt_reset = wx.Button( self.m_panel111, wx.ID_ANY, u"Reset", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_reset.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_reset, 0, wx.ALL, 5 )
        
        self.m_bt_exit = wx.Button( self.m_panel111, wx.ID_ANY, u"Exit", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_exit.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_exit, 0, wx.ALL, 5 )
        
        self.m_example = wx.Button( self.m_panel111, wx.ID_ANY, u"Example", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_example.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_example, 0, wx.ALL, 5 )
        
        self.m_panel111.SetSizer( fgSizer71 )
        self.m_panel111.Layout()
        fgSizer1.Add( self.m_panel111, 1, wx.EXPAND, 5 )
        
        self.SetSizer( fgSizer1 )
        self.Layout()
        
        self.Centre( wx.BOTH )
        
        # Connect Events
        self.m_example.Bind( wx.EVT_BUTTON, self.onbtexample )
        
        self.m_rb_threshhold.Bind( wx.EVT_KEY_DOWN, self.onrbthreshholdclicked )
        self.m_rb_mode.Bind( wx.EVT_KEY_DOWN, self.onrbmodeclicked )
        self.m_bt_submit.Bind( wx.EVT_BUTTON, self.onbtsubmit )
        self.m_bt_reset.Bind( wx.EVT_BUTTON, self.onbtreset )
        self.m_bt_exit.Bind( wx.EVT_BUTTON, self.onbtexit )
        
        icon = wx.Icon("puzzle.ico", wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        
        readconfig()







    
    def __del__( self ):
        pass
    
    # Virtual event handlers, overide them in your derived class
    def onbtexample( self, event ):
        """
        dlg = wx.MessageDialog(None, u"消息对话框测试", u"标题信息", wx.YES_NO | wx.ICON_QUESTION) 
        if dlg.ShowModal() == wx.ID_YES: 
            self.Close(True) 
            dlg.Destroy() 
        """
        strexample=">test\nATTGTCAAAATGTCTGGAACAGAATACTCAAATTGACTAGTTCTGGCCTTTTCCCTTAAATGGGCACGAGTAGGAGCAACAGACTACATCATCACTATCTCTAGAGAAATAGATCTTGCGAGAGAAAAAAACGTTGGTTGGTCTGCTTTTGGCTCTTTTGTCAATTAAATCCCCGGATGTACCTCAAAAAGACTGTAAAAGACTGGCTGGTGGACTAACGATGGCTTTCCTCAGCAGAAAGGAGGGAGAAAAAAAATTCAACTGGAACATCCAAAAGCGTTGAAATTCTTTGTGGGCAATATACATGAGATGGCTCCTAAGGAATATAAGGAGGTTGAAAAGAAGTTTCATCTGTACCAGTGTTTCTGCAATCCAGTTGGAGAGAACAGATTATGACTTATCTAGTGGTGAGATCGCACAGCCCAACTATGCGACCTCCGAAGCACATTTTCCCAGAATCTTTCCCTCAAGCACCTGGAACCTGGAACCCCCGAAGAAGAATGCTTGTCACGATGCGGCAAGGGAACACCTTTGGAAGGAGTAGATCTATTTTTTTATTTTTGAATTTTTGGGACTGTTGACCTTGCCTGCCTGAGAGCAAAAGAGAGACAACGACTGAGCAAGCACTACCACCAGACACTGTTACTGGCGAATTAGAAGACCTGATGTTTCGTTGGTCCTAGACCCTCAGTGCAAACCATCGAGGATGACTCCATATCCCACAACCGTGAACTTATGTCTCTGTGCCTCTCTGAATTTGCCGTGTAGTGGCTTCAGCCTGGACACCTTGGCTAGACTGCATACCCTGTCCCTTGAGTGCTATTCCTATCAGACACTATCAGAACAGTGCATTCCTCCAGAATTGGGATAGTAGCCAGGACAGAACAATTTCCTAGTCGCGACCCTACACGGCTGAACACGGACTTGACCCGTTTACTAAGCAAACAACCGAGATGGGCTGCACTTACTGGTACATTGAAGCTACATCGACTGTTGTCGAAATGTTCTTGACCACTCCGCAATTGACACTACCTGAATAGATATCAGCAACCGCGTGTGAGGTTTGTGGAAGGACGTCTGGTGCTCCTCATTGAGGGCCAATATTAGTAGCTAAGCGTTCGCGTAGCCATCTGATCGAGCCTCTTGTTATCAGTTCCAATGGTAACTCTTACTTCGTGTTTGCAAGAGACTAACTAATGAAAGACCCAAAATGTCTGCCAAATACTGGGCCAAGCATTGTTCCTGATAAGGGGACTCGAGCATATCTATCATTGCGGTCTTCATTTTCTCACATACACTTACGACATCGAACTAGGGTATGCAGCTACAACACGCCCAGTTAGCAGTGTATATATTGCGACGACCGTTGCGTTATTTACCTTTGCAGCCTTTAACTTACATGTATTGCGCAAAGATAAATGCAGCTAATAAGTCGTGTCTTAAAAAAGAAAAAAAAAAAAAAAGTGCTGGGAATCTCTTCTTCTTCGTTCCGATACCACCCTTGGCTATCATCTGCGCTATTACTGCCCAGATATGTATTATGAGCTAATTCATTTAGCTCAGCTTTCTTAATCAGTTTGAAACTATCCCCGATAGCGACAATCTTTGAACACCCCTCCTTCATCATTGTGCAGTTTAAAAATGTATAACCATTTGGAAAAAGTTTTGGTCTTGAGCTCAGCCAGTAACTAGATGTTTTTTTCTCTTTGAATTTGCTCTGCCCCTTGTTGGCCGCAGAGTTATGTTCTATTTTAAACGATGAATCTTT\n"
        self.m_txtseq.SetValue(strexample)

    
    def onrbfromatclicked( self, event ):
        event.Skip()
    
    def onrbformatbox( self, event ):
        event.Skip()
    
    def onrbthreshholdclicked( self, event ):
        event.Skip()
    
    def onrbmodeclicked( self, event ):
        event.Skip()
    
    def onbtsubmit( self, event ):
      
 
        
        if self.m_rb_mode.GetSelection() ==0:
            self.fullseqlen=41
            self.findseqlen=41
            self.Basestr="-AGCT"        
            self.modefile= Human_modelpklfile


            
        elif  self.m_rb_mode.GetSelection() ==1:           
            self.fullseqlen=41
            self.findseqlen=41
            self.Basestr="-AGCT"  
            self.modefile= Rice_modelpklfile

    
  

            
        
        
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Human_threshold_high
            self.thresholdlevel='95%'
   

        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Human_threshold_medum
            self.thresholdlevel='90%'
     
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Human_threshold_low
            
            self.thresholdlevel='80%'
   
            
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==1:
            self.threshold=Rice_threshold_high
            self.thresholdlevel='95%'
   

        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==1:
            self.threshold=Rice_threshold_medum
            self.thresholdlevel='90%'
   
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==1:
            self.threshold=Rice_threshold_low
            self.thresholdlevel='80%'
            
 
       
        #print(self.threshold)
      
        
        seqdictfull,labeldictfull=self.buildshortseq()   
        SICPredict=SICDm6APredict(self.modefile,self.threshold,self.thresholdlevel,  self.fullseqlen,self.findseqlen,self.Basestr,self.featlen)
        ss=SICPredict.predict(seqdictfull,labeldictfull)
        #print(222,ss)
        frame = SICResultFrame(None,ss)
        frame.Show(True)

        
    def onbtreset( self, event ):
        self.resettxt()
    
    def onbtexit( self, event ):
        self.Close(True)
        #dlg = wx.MessageDialog(None, u"消息对话框测试", u"标题信息", wx.YES_NO | wx.ICON_QUESTION) 
        #if dlg.ShowModal() == wx.ID_YES:
        #    self.Close(True) 
        #    dlg.Destroy() 
        #self.Close(True) 
        
        
        #self.Destroy()
        #wx.Exit()
    
    def resettxt(self):
        self.m_txtseq.Clear()
        #self.m_loadfile.
        
    def buildshortseq(self):
        
        v1=self.m_txtseq.GetValue() 
        v2=self.m_loadfile.GetPath()
        if len(v1)==0 and len(v2)==0:
            dlg2 = wx.MessageDialog(None, u"Need to input sequence with fasta format!",u"Information",  wx.YES_NO  )
            if dlg2.ShowModal() == wx.ID_YES:
                #self.Close(True)
                pass
            dlg2.Destroy()
            return
        #print(v2)
        if len(v1)>0: 
            seqdict=self.readfastatxt(v1)
        elif len(v2)>0: 
            seqdict=self.readfastafile(v2)
        #print(len(seqdict))
        seqdictfull=dict()
        labeldictfull=dict()
          
        #print(self.m_rb_mode.Selection )
     
         
        for key in seqdict :
            seq=seqdict[key]
            
            if seq=="":
                break
            #print(key,seq)
            seqdictfull,labeldictfull=self.findm6ashortseq(seqdictfull,labeldictfull, key,seq )
        #print(0000,seqdictfull)
        return seqdictfull,labeldictfull 
            
    def checkfastafilesize(self,fastafile):
        size = os.path.getsize(fastafile)
        size_in_Mb     = size/(1024*1024)
        if size_in_Mb >1:
            print(size_in_Mb,"Error!File size cannot exceed 2M!")
            dlg = wx.MessageDialog(None, u"File size cannot exceed 1M!", u"Error", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES: 
                dlg.Destroy()
                return -1
        return 1
        
                
    def checkfastatxtize(self,fastatxt):
        len1 =len(fastatxt)
        
        if len1 >100000:
            print(len1,"Error! The number of characters cannot exceed 100,000!")
            dlg = wx.MessageDialog(None, u"The number of characters cannot exceed 100,000!", u"Error", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES: 
                dlg.Destroy()
                return -1
        return 1
                    
            
    def readfastafile(self,fastafile):
        if self.checkfastafilesize( fastafile)==-1:
            return dict()
            
        fh=open(fastafile)
        seq=dict()
        for line in fh:
            if line.startswith('>'):
                
                name=line[1:] 
                seq[name]=''
            else:
                seq[name]+=line.replace('\n',"")
        fh.close()
        return seq
    
    def readfastatxt(self,fastatxt):
        if self.checkfastatxtize( fastatxt)        ==-1:
            return dict()
        seqdict=dict()
        ss=fastatxt.split("\n")     
     
        for i in range(len(ss)):
           
            line=ss[i]
            #print(line)
            if line=="":
                break
            if line.startswith('>'):
                name=line [1:len(line)]
                #print(name)
                seqdict[name]=''
            else:
                seqdict[name]+=line 
         
        return seqdict
       
        

 
    def findm6ashortseq(self,seqdictfull,labeldictfull,key,seq ):
        #print(seq)
 
        seqlen=len(seq)
        seqdict=seqdictfull
        labeldict=labeldictfull
        i=0
        for m in p_motif.finditer(seq):
            
            start=m.start()-20
            end=m.end()+20
            seqkey=key+"_"+str(m.start() + 1)
            subseq=""
            if start <0:
                subseq=seq[0:end ]  
                subseq=subseq.rjust(41,"-")
                
            elif end>seqlen:
                subseq=seq[start:seqlen ]  
                subseq=subseq.ljust(41,"-") 
            else:
                subseq=seq[start:end] 
            subseq=subseq.replace("a","A")
            subseq=subseq.replace("t","T")
            subseq=subseq.replace("c","C")
            subseq=subseq.replace("g","G")
            subseq=subseq.replace("n","N")
            seqdict[seqkey]=subseq 
            labeldict[seqkey]=0
            i+=1
        
 
        return seqdict,labeldict
                
                
    
    
    
        
if __name__ == u'__main__': 
    app = None
    app = wx.App()
    frame = SICFrame(None)
    frame.Show(True)
    app.MainLoop()
    #

