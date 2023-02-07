import torch
import math
import torch.nn as nn
import torch.optim as optim
import args
from numpy import *
from util_for_BQ import *
from tqdm import tqdm_notebook,tqdm
from torch.nn import functional as F
from sklearn import metrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def length_to_mask(lengths):
    a = torch.zeros(lengths.shape,dtype=torch.int64)
    mask = a == lengths
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len = args.max_len):
        super(PositionalEncoding, self).__init__()
        pe=torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = torch.transpose(x,0,1)
        x = x + self.pe[:x.size(0),:]
        return x




class M1(nn.Module):
    def __init__(self,vocab_size, hidden_dim,num_class):
        super(M1, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.embeds = nn.Embedding(vocab_size,self.embedding_dim)

        self.bn_embeds = nn.BatchNorm1d(self.embedding_dim)
        # self.embeddings = nn.Embedding(vocab_size,embedding_dim)

        self.position_embedding = PositionalEncoding(self.embedding_dim)
        encoder_layer1 = nn.TransformerEncoderLayer(self.embedding_dim, 4,dim_feedforward=512, dropout=0.1,activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer1,8).to(device)
        self.vc = nn.Linear(args.embedding_dim, 1, bias=False).to(device)
        self.vd = nn.Linear(args.embedding_dim, 1, bias=False).to(device)
        self.vm = nn.Linear(args.embedding_dim, 1, bias=False).to(device)

        self.alpha = nn.Parameter(torch.FloatTensor([0.1])).to(device)
        self.att_fc= nn.Linear(4*max_len,max_len)


        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True).to(device)
        self.lr = nn.Linear(200,100)
        self.fc = nn.Sequential(
            nn.Linear(4548,1000),
            nn.Linear(1000,self.num_class)
        )

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def soft_attention_align(self, x1, x2,mat):
        attention = torch.matmul(x1, x2.transpose(1, 2)).to(device) + self.alpha * mat.to(device)
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2).to(device)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1).to(device)
        return x1_align, x2_align


    def forward(self,sent1,s1_mask,sent2,s2_mask,mat,label,is_train = True):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        # x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        # x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        x1 = self.bn_embeds(self.embeds(sent1.to(device)).transpose(1, 2).contiguous()).transpose(1, 2).to(device)
        x2 = self.bn_embeds(self.embeds(sent2.to(device)).transpose(1, 2).contiguous()).transpose(1, 2).to(device)

        x1_ = self.position_embedding(x1)
        tf_1 = self.transformer(x1_, src_key_padding_mask=s1_mask.to(device))
        x1_tf = tf_1.transpose(0, 1)

        x2_ = self.position_embedding(x2)
        tf_2 = self.transformer(x2_, src_key_padding_mask=s2_mask.to(device))
        x2_tf = tf_2.transpose(0, 1)


        x_1,x_2 = self.soft_attention_align(x1_tf,x2_tf,mat)

        row = torch.sum(mat,dim=1).to(device)
        line = torch.sum(mat,dim=2).to(device)

        o1, _ = self.lstm(x_1)
        o2, _ = self.lstm(x_2)

        row_ = row.unsqueeze(-1).to(device)
        line_ = line.unsqueeze(-1).to(device)

        output1 = torch.cat([x1_tf, o1], dim=2)
        output2 = torch.cat([x2_tf, o2], dim=2)

        q1_rep = self.apply_multiple(output1)
        q2_rep = self.apply_multiple(output2)

        m = self.lr(torch.cat([line,row],dim=-1).float())

        x = torch.cat([q1_rep, q2_rep, q1_rep - q2_rep,q1_rep*q2_rep,m], -1)


        # batch_size * seq_len * dim =>      batch_size * seq_len * hidden_size
        logits = self.fc(x)
        out = torch.softmax(logits,1).to(device)
        if is_train:
            loss1= nn.CrossEntropyLoss()
            loss_1 = loss1(out,label.to(device))
            # loss2 = cosent(lam=20)
            # out = out[:,1]
            # loss_2 = loss2(out,label)
            out = torch.argmax(out, 1)
            return loss_1,out
        else:
            out = torch.argmax(out,1)
            return out



Model = M1(vocab_size=args.vocab_size,hidden_dim=args.hidden_dim,num_class=args.class_size).to(device)
train_dataset = LoadData('PAWSX_train.pickle')
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=collate,shuffle=True,drop_last=False)
print('train data has been loaded')
test_dataset = LoadData('PAWSX_dev.pickle')
test_loader = DataLoader(test_dataset,batch_size=50,collate_fn=collate,shuffle=True,drop_last=True)
print('test data has been loaded')
optimizer = optim.Adam(Model.parameters(),lr=1e-5)
total_params = sum(p.numel() for p in Model.parameters())

LOSS = nn.CrossEntropyLoss()

print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in Model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
print('start training ....')
best_acc = 0
best_epoch = 0
for epoch in range(args.epoch):
    process_bar = tqdm(train_loader,leave = False)
    loss = 0
    train_acc = 0
    train_res = []
    for sent1,s1_mask,sent2,s2_mask,mat,label in process_bar:
        loss,output = Model(sent1,s1_mask,sent2,s2_mask,mat,label)
        optimizer.zero_grad()
        label = label.to(device)
        output = output.long().to(device)
        correct_prediction = torch.eq(output, label).to(device)
        train_accuracy = correct_prediction.float().to(device)
        train_acc = torch.mean(train_accuracy, dim=0).item()
        train_res.append(train_acc)
        loss.backward()
        optimizer.step()
    print('epoch={},loss={},train_acc = {}'.format(epoch,loss.item(),mean(train_res)))
    res = []
    f1s = []
    for sent1,s1_mask,sent2,s2_mask,mat,label in test_loader:
        output = Model(sent1,s1_mask,sent2,s2_mask,mat,label,is_train = False)
        label = label.to(device)
        output = output.long().to(device)
        correct_prediction = torch.eq(output, label).to(device)
        test_accuracy = correct_prediction.float().to(device)
        test_acc = torch.mean(test_accuracy, dim=0).item()
        res.append(test_acc)
        f1 = metrics.f1_score(label.cpu(),output.cpu())
        f1s.append(f1)
    if mean(res) > best_acc and epoch>20:
        best_acc = mean(res)
        best_epoch = epoch + 1
        # torch.save(Model.state_dict(), './data/models/bq/2-{}-{}.pth'.format(best_epoch, best_acc))

    print('epoch =', epoch + 1, 'test_acc=', mean(res),'f1=',mean(f1s), ' best acc epoch:', best_epoch, ' best acc:', best_acc)
    # if epoch % 100 == 0:
    #     torch.save(Model.state_dict(), './data/models/bq/2-{}-{}.pth'.format(epoch, mean(res)))


