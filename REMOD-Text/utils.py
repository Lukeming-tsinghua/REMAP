import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def accuracy(label,score):
    pred = torch.argmax(score,1)
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return accuracy_score(label,pred)

class Summary:
    def __init__(self,path,taskname,setname):
        self.path = path
        self.taskname = taskname
        self.setname = setname
        self.defaultname = "result.pth"
        self.summary = {}
    
    def update(self,epoch,score,pred,label,loss):
        cm = confusion_matrix(label,pred)
        acc = accuracy_score(label,pred)
        report = classification_report(label,pred)
        s = (score,pred,label,cm,acc,report,loss)
        self.summary[epoch] = s
        dirpath = os.path.join(self.path,self.taskname,str(epoch),self.setname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath,self.defaultname)
        torch.save(s,filepath)
        print(self.taskname,self.setname,str(epoch),loss,acc,cm,report)
    
    def save(self):
        dirpath = os.path.join(self.path,self.taskname,self.__class__.__name__,self.setname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath,self.defaultname)
        torch.save(self.summary,filepath)


class ModelCheckpoint:
    def __init__(self,path,taskname):
        self.path = path
        self.defaultname = "checkpoint.pth"
        self.taskname = taskname
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def save(self,epoch,model,optimizer,scheduler,loss):
        dirpath = os.path.join(self.path,self.taskname,repr(model),str(epoch))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath,self.defaultname)
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': loss
            }
        torch.save(state,filepath)
    
    def load(self,epoch,model,optimizer,scheduler):
        dirpath = os.path.join(self.path,self.taskname,repr(model),str(epoch))
        if not os.path.exists(dirpath):
            raise ValueError(f"file path {filepath} does not exist")
        filepath = os.path.join(dirpath,self.defaultname)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch,model,optimizer,scheduler,loss

class AlphaTest:
    def __init__(self, alpha, pred, label):
        self.alpha = np.array(alpha)
        self.pred = np.array(pred)
        self.label = np.array(label)

        self.correct_alpha = self.alpha[self.pred == self.label]
        self.wrong_alpha = self.alpha[self.pred != self.label]

        self.positive_alpha = self.alpha[self.label != 0]
        self.negative_alpha = self.alpha[self.label == 0]
        
    def levene_test(self):
        print('test whether mean(correct_alpha) == mean(wrong_alpha)')
        print('mean of correct_alpha = ', np.mean(self.correct_alpha), ' mean of wrong_alpha =', np.mean(self.wrong_alpha))
        if stats.levene(self.correct_alpha, self.wrong_alpha).pvalue < 0.05:
            print(stats.ttest_ind(self.correct_alpha, self.wrong_alpha,equal_var= False))
        else:
            print(stats.ttest_ind(self.correct_alpha, self.wrong_alpha))

    def t_test(self):
        # test var(positive_alpha) == var(negative_alpha) by t-test
        print('test whether mean(positive_alpha) == mean(negative_alpha)')
        print('mean of positive_alpha = ', np.mean(self.positive_alpha), ' mean of negative_alpha =', np.mean(self.negative_alpha))
        if stats.levene(self.positive_alpha, self.negative_alpha).pvalue < 0.05:
            print(stats.ttest_ind(self.positive_alpha, self.negative_alpha, equal_var= False))
        else:
            print(stats.ttest_ind(self.positive_alpha, self.negative_alpha))
