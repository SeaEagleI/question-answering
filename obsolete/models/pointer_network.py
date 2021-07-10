# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Output layer: Ptr-Net
class PointerNetwork(nn.Module):
    def __init__(self, pass_hidden_dim, question_hidden_dim, attn_size=75, dropout=0.2):
        """ Pointer Network
                Args:
                    pass_hidden_dim(int): size of input
                Input:
                    - **H_passage** of shape `(passage_legth, batch, pass_hidden_dim)`: a float tensor in which we determine
                    the importance of information in the passage regarding a question
                    - **U_question** of shape `(question_length, batch, question_hidden_dim)`: a float tensor containing question
                    representation
                Output:
                    - start(torch.tensor of shape (batch_size, passage_length, 1)): start position of the answer
                    - end(torch.tensor of shape (batch_size, passage_length, 1)): end position of the answer
        """
        super(PointerNetwork, self).__init__()
        # for c, ha*
        self.Whp = nn.Linear(pass_hidden_dim, attn_size, bias=False)
        self.Wha = nn.Linear(question_hidden_dim, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)
        self.cell = nn.GRUCell(pass_hidden_dim, question_hidden_dim, False)
        # for rQ
        self.Wuq = nn.Linear(question_hidden_dim, attn_size, bias=False)
        self.v1 = nn.Linear(attn_size, 1, bias=True)

    def get_initial_state(self, u_question):
        # Attention Pooling 0: u_question => rQ
        s = self.v1(torch.tanh(self.Wuq(u_question)))
        a = F.softmax(s, 0)
        rQ = (a * u_question).sum(0)
        return rQ

    def forward(self, h_passage, u_question, batch_first=False):
        # Reshape
        if batch_first:
            h_passage = h_passage.transpose(0, 1)
            u_question = u_question.transpose(0, 1)

        # ha0 = rQ: [1, batch_size, question_hidden_dim]
        ha0 = self.get_initial_state(u_question)
        # Attention Pooling 1: ha0, h_passage => start_logits, c
        Wh = self.Whp(h_passage)
        s1 = self.v(torch.tanh(Wh + self.Wha(ha0)))
        start_logits = s1.transpose(0, 1)  # shape[pass_len,batch_size,1] => shape[batch_size,pass_len,1]
        a1 = F.softmax(s1, 0)
        c = (a1 * h_passage).sum(0)
        # RNN (GRU): c, ha0 => ha1
        ha1 = self.cell(c, ha0)
        # Attention Pooling 2: ha1, h_passage => end_logits (No need to compute ha2)
        s2 = self.v(torch.tanh(Wh + self.Wha(ha1)))
        end_logits = s2.transpose(0, 1)  # shape[pass_len,batch_size,1] => shape[batch_size,pass_len,1]
        # Return start & end logits
        return start_logits, end_logits
