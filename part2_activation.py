from turtle import forward
import numpy as np

class activation_ReLu:
    def forward(self,input):
        self.output = np.maximum(0,input)




# ReLu = max(0,n)


