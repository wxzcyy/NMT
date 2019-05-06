#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    
    def __init__(self, features):
        super(Highway, self).__init__()        
        self.W_proj = nn.Linear(features, features)
        self.W_gate = nn.Linear(features, features)
    
    def forward(self, conv_out):
        """ 
            @param conv_out : (b, e)卷积网络的输出
            @returns X_highway : (b, e)Highway输出
        """
        
        X_proj = F.relu(self.W_proj(conv_out)) # (b, e)
        X_gate = torch.sigmoid(self.W_gate(X_proj)) # (b, e)
        
        X_highway = X_gate*X_proj + (1-X_gate)*conv_out
        
        return X_highway
    
