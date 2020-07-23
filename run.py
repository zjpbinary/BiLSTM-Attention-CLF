from bilstm import *
from loaddata import *
from config import *
def train_model(pairs, model, bsz, maxlen, criterion, optimizer, device, epoch):
    model.train()
    hidden = model.init_hidden(bsz, device)
    for i in range(epoch):
        for j,batch in enumerate(pairs):
            target = torch.LongTensor(batch[0]).to(device)
            x = torch.LongTensor(batch[1]).transpose(0, 1).to(device)
            out = model.forward(x, hidden, maxlen)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            print('第%d次迭代' % i, '第%d个batch' % j, 'loss为：', loss)
def predict_model(tuple, model, maxlen,  device):
    model.eval()
    hidden = model.init_hidden(1, device)
    right = 0.
    for i in range(len(tuple[0])):
        x = torch.LongTensor(tuple[1][i]).unsqueeze(0).transpose(0,1).to(device)
        out = model.forward(x, hidden, maxlen)
        pre = torch.argmax(out)
        if pre.item()==tuple[0][i]:
            right+=1
    precision = right/len(tuple[0])
    print('测试集上的精度为：', precision)

if __name__ == '__main__':
    args = TextCNNConfig()
    max_len, vocab_size, label_size, trainpairs, testtuple = load_data(args.trainfile, args.testfile, args.bsz, args.max_len)
    device = torch.device(args.device)
    model = Bilstm_Attention_Model(vocab_size, label_size, args.ninp, args.nhid, args.nlayers, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train_model(trainpairs, model, args.bsz, max_len, criterion, optimizer, device, args.epoch)
    predict_model(testtuple, model, max_len, device)