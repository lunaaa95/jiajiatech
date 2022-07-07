import torch.nn as nn


# GRU 模型
class GRU(nn.Module):

    def __init__(self, input_size, nlayers, nhid, dropout):
        super(GRU, self).__init__()

        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.Linear(nhid, input_size)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_sz, requires_grad=True):
        weight = next(self.parameters())
        return weight.new_zeros((self.nlayers, batch_sz, self.nhid), requires_grad=requires_grad)

    def forward(self, input, hidden):
        # input: bsz, seq_len, input_size
        # hidden: 1, bsz, nhid
        output, hidden = self.rnn(input, hidden)
        # output: bsz, seq_len, nhid
        # hidden: 1, bsz, nhid
        decoded = self.decoder(hidden.reshape(hidden.size(0) * hidden.size(1), hidden.size(2)))
        # decoded: bsz, input_size
        return decoded
