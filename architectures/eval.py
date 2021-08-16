import numpy as np
from sklearn.metrics import confusion_matrix

#File used to store function to measure performances and other features from results. 
class Performance:
    def precision(self, ypred, ytrue):#real pos/estim pos : 1-->everytime we estimate pos, it's pos
        return np.sum(ytrue*ypred)/np.sum(ypred)
    def recall(self, ypred, ytrue):#part of pos really detected : 1--> everytime it's pos, we detect it
        return np.sum(ytrue*ypred)/np.sum(ytrue)
    def F_score(self, ypred, ytrue, beta=1):
        p = self.precision(ytrue, ypred)
        r = self.recall(ytrue, ypred)
        return (1+beta**2)*p*r/(beta**2*p+r)
    def accuracy(self, ypred, ytrue):#
        return np.sum(ypred==ytrue)/len(ypred)
    def confusion_matrix(self, ypred, ytrue):
        return confusion_matrix(ytrue, ypred)
    

def show_top(classifier, list_col):
    feature_names = np.array(list_col)#liste de str
    top = np.argsort(classifier.coef_[0])
    return feature_names[top]
    #print("%s: %s" % ('RELEVANT', " ".join(feature_names[top10])))

def show_top10(classifier, list_col):
    feature_names = np.array(list_col)
    top10 = np.argsort(classifier.coef_[0])[-10:]
    #print(classifier.coef_[top10])
    #print(top10)
    return "%s: %s" % ('RELEVANT', " ".join(feature_names[top10]))