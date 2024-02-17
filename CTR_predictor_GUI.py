
"""
GUI
"""
import time
from threading import Thread
import wx
#from wx.lib.pubsub import setuparg1
#from wx.lib.pubsub import pub as Publisher
import pyperclip
#from pubsub import publisher

#import pubsub as Publisher
#from flask import Flask
#from multiprocessing import Process

app=wx.App()

class TestThread(Thread):
    """Test Worker Thread Class."""
 
    #----------------------------------------------------------------------
    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.start()    # start the thread
 
    #----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(60):
            time.sleep(1)
            wx.CallAfter(self.postTime, i)
        #time.sleep(1)
        #wx.CallAfter(Publisher.sendMessage, "update", "Prediction done!")
        #wx.CallAfter(Publisher.sendMessage, "update", "Prediction done!")
 
    #----------------------------------------------------------------------
    #def postTime(self, amt):
        #"""
        #Send time to GUI
        #"""
        #amtOfTime = (amt + 1) * 10
        #Publisher.sendMessage("update", amtOfTime)
 

class MyFrame(wx.Frame):    
    
    def __init__(self):
        
        
        super().__init__(parent=None, title='CTR predictor')
        
        self.panel = wx.Panel(self)    
        self.instruction1=wx.StaticText(self.panel,label='First choose the correct promotion type, product and lifecycle of the email ',pos=(0,40))
        self.instruction2=wx.StaticText(self.panel,label='Now please enter the content of the email and press the "Predict" button to predict its click through rate',pos=(775,40))
        self.result = wx.StaticText(self.panel, label="",pos=(910,415))
        self.prom_choice_label=wx.StaticText(self.panel,label='Promotion type',pos=(8,80))
        self.prom_label=wx.StaticText(self.panel,label="",pos=(0,110))
        self.prod_choice_label=wx.StaticText(self.panel,label='Product',pos=(130,80))
        self.prod_label=wx.StaticText(self.panel,label="",pos=(130,110))
        self.life_choice_label=wx.StaticText(self.panel,label='Lifecycle',pos=(230,80))
        self.life_label=wx.StaticText(self.panel,label="",pos=(230,110))
        self.mod_label=wx.StaticText(self.panel,label="Prediction model",pos=(0,130))
        
        self.displayLbl = wx.StaticText(self.panel, label="",pos=(780,350))
        self.processLbl=wx.StaticText(self.panel,label="",pos=(920,400))
        
        my_sizer = wx.BoxSizer(wx.VERTICAL)        
        self.text_ctrl = wx.TextCtrl(self.panel,size=(1000,200),style=wx.TE_MULTILINE|wx.VSCROLL,pos=(500,100))
        
        #my_sizer.Add(self.text_ctrl, 1, wx.ALL | wx.EXPAND, 15)        
        my_btn = wx.Button(self.panel, label='Predict',pos=(750,100))
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 300) 
        
        self.prom_choice = wx.Choice(self.panel,choices = distinct_promotions,name='Promotion type',pos=(0,95))
        #self.prom_choice.SetStringSelection(string='Promotion') 
        self.prom_choice.Bind(wx.EVT_CHOICE, self.OnChoice_prom)
        self.prod_choice=wx.Choice(self.panel,choices=distinct_products,name='Product',pos=(130,95))
        self.prod_choice.Bind(wx.EVT_CHOICE, self.OnChoice_prod)
        self.life_choice=wx.Choice(self.panel,choices=distinct_lifecycles,id=-1,name='Lifecycle',pos=(230,95))
        self.life_choice.Bind(wx.EVT_CHOICE, self.OnChoice_life)
        #self.png = wx.Image(r'\\sbirstafil001\users\CFarrugia\CTR_hist.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        #wx.StaticBitmap(self.panel, -1, self.png, (650, 500), (self.png.GetWidth(), self.png.GetHeight()))
        models=['Model 1', 'Model 2','Model 3']
        self.mod_choice=wx.Choice(self.panel,choices =models,name='Model',pos=(0,150))
        self.mod_choice.Bind(wx.EVT_CHOICE,self.OnChoice_mod)
        
        self.index=0
        
        self.email_ctr_dict={}
        self.list_ctrl = wx.ListCtrl(
        self.panel, size=(800,450),pos=(600, 500), 
        style=wx.LC_REPORT | wx.BORDER_SUNKEN
        )
        self.list_ctrl.InsertColumn(0, 'Content', width=700)
        self.list_ctrl.InsertColumn(1, 'CTR', width=100)
        self.list_ctrl.Bind(wx.EVT_RIGHT_UP, self.ShowPopup)

        #Publisher.subscribe(self.updateDisplay, "update")
        
        self.SetBackgroundColour((100, 179, 179))
        self.panel.SetSizer(my_sizer)        
        self.Show()
        #wx.MessageBox('You need to insert promotion type, product and lifecycle of the email', 'Warning',wx.OK | wx.ICON_WARNING)
        #prm_ind=self.prom_choice.GetSelection()

    
    def OnChoice_prom(self,event): 
      #prm_ind=self.prom_choice.GetSelection()
      self.prm=self.prom_choice.GetString(self.prom_choice.GetSelection())
      self.mapped_prm=integerMapping[self.prm]
      #self.prom_label.SetLabel("You selected "+ prm +" from Choice")  

    def OnChoice_prod(self,event): 
      
      self.prd=self.prod_choice.GetString(self.prod_choice.GetSelection())  
      self.mapped_prd=int_mapping_prod[self.prd]
      #self.prod_label.SetLabel("You selected "+ prd +" from Choice")
      
    def OnChoice_life(self,event):
        
      self.lcy=self.life_choice.GetString(self.life_choice.GetSelection())
      self.mapped_life=int_mapping_life[self.lcy]
      #self.life_label.SetLabel("You selected "+lcy+" from Choice")      
     
    def OnChoice_mod(self,event):
        self.modl=self.mod_choice.GetString(self.mod_choice.GetSelection())
            
    def ShowPopup(self, event):
        menu = wx.Menu()
        menu.Append(1, "Copy selected items")
        menu.Bind(wx.EVT_MENU, self.CopyItems, id=1)
        self.PopupMenu(menu)

    

    def CopyItems(self, event):
    
        listSelectedLines =[]
        index = self.list_ctrl.GetFirstSelected()  
    
        while index is not -1:
            listSelectedLines.append(self.list_ctrl.GetItem(index, 0).GetText())
            index = self.list_ctrl.GetNextSelected(index)             
    
        pyperclip.copy(''.join(listSelectedLines))
        
        
    def on_press(self, event):
        
        
        self.value = self.text_ctrl.GetValue()
        #self.list_ctrl.InsertStringItem(self.index, self.text_ctrl.GetValue() )
        #self.list_ctrl.SetStringItem(self.index, 1, "2010")
       
        #self.index += 1
        
        if not self.value:
            wx.MessageBox("You didn't input any text", 'Warning',wx.OK | wx.ICON_WARNING)
            #self.result.SetLabel("You didn't enter anything")
            #self.quote = wx.StaticText(self.panel, label="Prediction:",pos=(910,400))
            
        elif self.prom_choice.GetSelection()==-1 or self.prod_choice.GetSelection()==-1 or self.life_choice.GetSelection()==-1 or self.mod_choice.GetSelection()==-1: #or prd or prm:
            wx.MessageBox('You need to insert promotion type, product and lifecycle of the email, as well as which prediction model you want to use', 'Warning',wx.OK | wx.ICON_WARNING)
         
        elif self.modl =='Model 1':
            
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor(self.value,self.mapped_prd,self.mapped_prm,self.mapped_life)
            
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
            #wx.MessageBox('Click rate:' + self.prediction[0], 'Warning',wx.OK | wx.ICON_WARNING)
         
            #self.index += 1
            #self.result.SetLabel('The predicted click rate for the email you entered is')
        
        elif self.modl == 'Model 2':
            
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor_xgb(self.value,self.mapped_prd,self.mapped_prm,self.mapped_life)
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
        
        elif self.modl == 'Model 3':
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor_randf_w2v(self.value)
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
        
        #self.my_btn.Disable()
        
"""   
    def updateDisplay(self, msg):
            
            #Receives data from thread and updates the display
            
            t = msg.data
            if isinstance(t, int):
                self.displayLbl.SetLabel("")
                #self.displayLbl.SetLabel("Prediction done in %s seconds" % t)
            else:
                self.displayLbl.SetLabel("")
                #self.displayLbl.SetLabel("%s" % t)
                #my_btn.Enable()
"""
                
if __name__ == '__main__':
    #app = wx.App()
    frame = MyFrame()
    app.MainLoop()

del app    

import webapp2
routes = [('/', MyFrame)]

my_app = webapp2.WSGIApplication(routes, debug=True)

"""
def run_webserver():
    
    app = Flask(__name__)
    app.run(debug=True)
    app.route("/")
if __name__ == '__main__':
    p = Process(target=MyFrame)
    p.start()

    p2 = Process(target=run_webserver)
    p2.start()