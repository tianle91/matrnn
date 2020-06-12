from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tssparse import get_input_t


class LSTMTSSparseToTTE(nn.Module):
    
    def __init__(self, num_evtypes, max_t):
        super(LSTMTSSparseToTTE, self).__init__()
        self.num_evtypes = num_evtypes
        self.max_t = max_t
        self.lstm = nn.LSTM(
            input_size=num_evtypes*2, 
            hidden_size=num_evtypes, 
            num_layers=1, 
            bidirectional=False
        )
        
    def forward(self, arrivaltimes: List[List[int]]):
        batch_size = len(arrivaltimes)
        hidden = (
            # (num_layers * num_directions, batch, hidden_size)
            torch.randn(1, batch_size, self.num_evtypes), 
            torch.randn(1, batch_size, self.num_evtypes),
        )
        output_l = []
        for t in range(self.max_t):
            # (seq_len, batch, input_size)
            input_t = torch.Tensor(
                get_input_t(t, arrivaltimes, training=True)
            ).view(1, batch_size, -1)
            output_t, hidden = self.lstm(input_t, hidden)
            # output_t: (seq_len, batch, num_directions * hidden_size)
            output_l.append(output_t)
        
        return F.softplus(torch.cat(output_l, dim=0))
