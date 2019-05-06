#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        
        # 字符翻译器
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        
        # 输入为c_{t-1}, h_{t-1}, x_{t}
        self.char_output_projection = nn.Linear(hidden_size , len(target_vocab.char2id), bias=True) # 获取输出的分数
        
        #获取一个新的字符嵌入层
        padding_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx) 
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input = self.decoderCharEmb(input) # batch表示单词的数目，length表示最大的单词长度
        outputs, dec_hidden = self.charDecoder(input, dec_hidden) # dec_hidden是上一隐藏层的输出状态
#        print(outputs.shape)
        scores = self.char_output_projection(outputs) # 求出每一个输出层的状态
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # 输入 (length - 1, batch)
        source_sequence = char_sequence[:-1]
        # 输出 (length - 1, batch)
        target_sequence = char_sequence[1:]
        
        
        scores, dec_hidden = self.forward(source_sequence, dec_hidden) # scores: (length, b, vocab_size)
#        print(score)
        scores = -nn.functional.log_softmax(scores, dim=-1)
        # 获取目标单词串的掩码
        dec_mask = (source_sequence != self.target_vocab.char2id['<pad>']).float()
#        print(dec_mask)
#        print(dec_mask.shape)
        gather = torch.gather(scores, dim=-1, index=torch.unsqueeze(target_sequence, dim=-1))
#        print(gather.shape)
#        print(torch.squeeze(gather, dim=-1).shape)
#        print(dec_mask*torch.squeeze(gather, dim=-1))
        
        logits = torch.sum(dec_mask*torch.squeeze(gather, dim=-1), dim=0)
        cross_entropy = torch.sum(logits)
        return cross_entropy
        
        

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        batch_size = initialStates[0].size(1)
        
        output_ids = None
#        print(output_words)
        current_char = torch.tensor([self.target_vocab.char2id['{']] * batch_size, device=device)
        current_char = torch.unsqueeze(self.decoderCharEmb(current_char), dim=0)
        h_t, c_t = initialStates[0], initialStates[1]
#        print(current_char.shape)
        
        for t in range(max_length):
            outputs, (h_t, c_t) = self.charDecoder(current_char, (h_t, c_t))
#            print(f'output:{outputs}')
#            print(f'hidden:{h_t}')
            s_t = self.char_output_projection(h_t)
            p_t = nn.functional.softmax(s_t, dim=-1)
#            print(p_t.shape)
            current_char = torch.argmax(p_t, dim=-1)
            if t == 0:
                output_ids = current_char
            else:
                output_ids = torch.cat([output_ids, current_char], dim=0)
            current_char = self.decoderCharEmb(current_char)
        
        #获取字符
        
        output_ids = output_ids.permute(1,0).tolist()
#        print(output_ids)
        output_words = []
        for output_id in output_ids:
            word = []
            for char_id in output_id:
#                print(self.target_vocab.id2char[char_id])
                if char_id != self.target_vocab.end_of_word:
                    word = word + [self.target_vocab.id2char[char_id]]
                else:
                    break
            output_words.append(''.join(word))
        
#        print(output_words)
        
        return output_words
        
        ### END YOUR CODE

