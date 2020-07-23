import torch
import torch.nn as nn

class Bilstm_Attention_Model(nn.Module):
    def __init__(self, ntoken, nlabel, ninp, nhid, nlayers, dropout = 0.5):
        super(Bilstm_Attention_Model, self).__init__()

        self.nlayers = nlayers
        self.nhid = nhid
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True)
        #self.attention = nn.Linear(2*nhid, 1)
        self.attention_weight = nn.Parameter(torch.randn(nhid))
        self.out = nn.Linear(nhid, nlabel)

        self.init_weight()

    def init_hidden(self, bsz, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid).to(device),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid).to(device))
    def init_weight(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)
    def forward(self, x, hidden, maxlen):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.bilstm(emb, hidden)
        att = self.attention(out.view(maxlen, -1, 2, self.nhid))
        pre = self.out(att)
        return torch.softmax(pre, dim=1)
    def attention(self, out):
        #out shape:max_len x batch_size x 2 x nhid
        h = out[:, :, 0, :]+out[:, :, 1, :]
        m = self.tanh(h.squeeze(2))
        w = torch.einsum('k,ijk->ij', [self.attention_weight, m])
        w = torch.softmax(w, 0)
        r = torch.einsum('ij,ijk->jk', [w, m])
        return r
''' 
if __name__=='__main__':
    device = torch.device('cpu')
    model = Bilstm_Attention_Model(10,7,128,128,2,device,2)
    input = torch.LongTensor([[1,3],[2,4],[3,5]])
    o = model.forward(input,3)
    print(o)
    print(o.shape)
'''







