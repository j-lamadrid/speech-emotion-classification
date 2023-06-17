from libsvm.svmutil import *
import warnings

class SVMModel:
    
    def __init__(self):
        
        warnings.filterwarnings("ignore")
        
        self.svm_model = None
    
    
    def train(self, X_train, y_train):
        """
        Use Summary Statistics of MFCC
        """
        
        self.svm_model = svm_train(y_train, [x.numpy() for x in X_train], '-t 2 -c 10')
                     
        
    def get_accuracies(self,  X_train, y_train, X_test, y_test):
        
        svm_train = [x.flatten().cpu().real.numpy() for x in X_train]
        svm_test = [x.flatten().cpu().real.numpy() for x in X_test]

        p_label_train, p_acc_train, p_val_train = svm_predict(y_train, svm_train, self.svm_model)
        p_label_test, p_acc_test, p_val_test = svm_predict(y_test, svm_test, self.svm_model)
        
        return p_label_test, p_acc_test, p_val_test