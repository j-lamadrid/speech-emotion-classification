import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio
from torchvision import models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import utils

import matplotlib.pyplot as plt

import warnings



class ViTModel:
    
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.filterwarnings("ignore")
        
        self.identity = 'ViT'
        model = models.vit_b_16(pretrained=True)
        model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        model.heads.head = nn.Linear(in_features=768, out_features=6, bias=True)
      
        self.model = model.to(self.device)
        self.epochs = 6
        self.train_accs = []
        self.test_accs = []
    
    
    def train(self, X_train, y_train, X_test, y_test, lr=0.00001, epochs=6, display=True):
        """
        Use MFCC in Execution
        """
        r = T.Resize((224, 224))
        vit_train = [r(s) for s in X_train]
        vit_test = [r(s) for s in X_test]
        
        self.epochs = epochs
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        label_ohe = F.one_hot(torch.tensor(y_train), 6)
        
        for e in range(self.epochs):
            
            for i in range(len(vit_train)):
                
                self.model.train()
                
                x = vit_train[i].type(torch.FloatTensor).to(self.device)
                y = label_ohe[i].type(torch.FloatTensor).to(self.device)

                scores = self.model(x).to(self.device)
                loss = F.cross_entropy(scores[0], y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            self.train_accs.append(loader.get_accuracies(self.model, vit_train, y_train)[0])
            self.test_accs.append(loader.get_accuracies(self.model, vit_test, y_test)[0])
            if display:
                print('Epoch %d, loss = %.4f' % (e + 1, loss.item()))
                

    def plot_accuracies(self):
        x_axis = np.arange(1, self.epochs + 1)

        plt.plot(x_axis, train_accs, label='Train')
        plt.plot(x_axis, test_accs, label='Test')

        plt.legend()

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        plt.title(self.identity)

        plt.show()
        
        
    def confusion_matrix(self,  X_train, y_train, X_test, y_test):
        r = T.Resize((224, 224))
        vit_train = [r(s) for s in X_train]
        vit_test = [r(s) for s in X_test]
        
        train_a, train_p = loader.get_accuracies(self.model, vit_train, y_train)
        test_a, test_p = loader.get_accuracies(self.model, vit_test, y_test)
        
        cm = confusion_matrix(y_test, test_p)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()