#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, max_word_length=21):
        super(CNN, self).__init__()
        
        self.max_word_length = max_word_length
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size=self.max_word_length - kernel_size + 1)
        
    def forward(self, input):
        """对input进行卷积
            @param input: (b, channels(char_embedding_length), char_count)
            @returns X_conv: (b, word_embedding_length)
        """
        X_conv = self.conv(input)
        X_max_pool = self.max_pool(F.relu(X_conv))
        return X_max_pool
### END YOUR CODE

