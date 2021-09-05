import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

input_file = sys.argv[1]
test_file = sys.argv[2]
start = input_file.split(".")[0]
predict_file = start + ".base_prediction.txt"

total_data = []
with open(input_file, "r") as f:
    texts = f.readlines()
    for text in texts:
        text = text.strip('\n').split('\t')
        total_data.append([int(text[0]), int(text[1]), int(text[2]), int(text[3])])
df = pd.DataFrame(total_data, dtype=np.int32)
user_idx = np.unique(np.array(df)[:,0])
item_idx = np.unique(np.array(df)[:,1])
userId2idx = {uId:idx for (idx, uId) in enumerate(user_idx)}
itemId2idx = {iId:idx for (idx, iId) in enumerate(item_idx)}
# 있는 인덱스끼리 연결
df[0] = df[0].apply(lambda x: userId2idx.get(x))
df[1] = df[1].apply(lambda x: itemId2idx.get(x))

num_user = len(user_idx)
num_item = len(item_idx)

R = np.zeros((num_user, num_item), dtype=np.int32)
R_OC = np.zeros((num_user, num_item), dtype=np.int32)
for data in total_data:
    R[userId2idx[data[0]]][itemId2idx[data[1]]] = data[2]
    R_OC[userId2idx[data[0]]][itemId2idx[data[1]]] = 1

train_set = df

class OCCF(nn.Module):
    def __init__(self, num_user, num_item, embedding_size):
        super(OCCF, self).__init__()
        self.embedding_user = nn.Embedding(num_user, embedding_size)
        self.embedding_item = nn.Embedding(num_item, embedding_size)
        self.embedding_user_bias = nn.Embedding(num_user, 1)
        self.embedding_item_bias = nn.Embedding(num_item, 1)
        torch.nn.init.xavier_uniform_(self.embedding_user.weight)
        torch.nn.init.xavier_uniform_(self.embedding_item.weight)
        torch.nn.init.xavier_uniform_(self.embedding_user_bias.weight)
        torch.nn.init.xavier_uniform_(self.embedding_item_bias.weight)
    
    def forward(self, user, item):
        embedded_user = self.embedding_user(user)
        embedded_item = self.embedding_item(item)
        bias_user = self.embedding_user_bias(user).squeeze()
        bias_item = self.embedding_item_bias(item).squeeze()
        return (embedded_user * embedded_item).sum(1) + bias_user + bias_item

class MF_bias(nn.Module):
    def __init__(self, num_user, num_item, mf_embedding_size=8, fn_embedding_size=32):
        super(MF_bias, self).__init__()
        self.last_layer_size = 16
        self.mf_embedding_user = nn.Embedding(num_user, mf_embedding_size)
        self.mf_embedding_item = nn.Embedding(num_item, mf_embedding_size)
        self.fn_embedding_user = nn.Embedding(num_user, fn_embedding_size)
        self.fn_embedding_item = nn.Embedding(num_item, fn_embedding_size)
        torch.nn.init.xavier_uniform_(self.mf_embedding_user.weight)
        torch.nn.init.xavier_uniform_(self.mf_embedding_item.weight)
        torch.nn.init.xavier_uniform_(self.fn_embedding_user.weight)
        torch.nn.init.xavier_uniform_(self.fn_embedding_item.weight)
        self.linear_stack = nn.Sequential(
            nn.Linear(2*fn_embedding_size, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(2*fn_embedding_size, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.before_out = nn.Linear(self.last_layer_size + mf_embedding_size, 1)
    
    def forward(self, user, item):
        mf_embedded_user = self.mf_embedding_user(user)
        mf_embedded_item = self.mf_embedding_item(item)
        fn_embedded_user = self.fn_embedding_user(user)
        fn_embedded_item = self.fn_embedding_item(item)
        mf_mul = torch.mul(mf_embedded_user, mf_embedded_item)
        fn_concat = torch.cat([fn_embedded_user, fn_embedded_item], dim=-1)
        fn_result = self.linear_stack(fn_concat)
        merged = torch.cat([mf_mul, fn_result], dim=-1)
        ratings = self.before_out(merged)
        return ratings.sum(1)

    
def train(model, epoch=15, lr=0.01, wd=0):
    optimizer= torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    for i in range(epoch):
        model.train()
        users = torch.LongTensor(train_set[0].values).to(device)
        items = torch.LongTensor(train_set[1].values).to(device)
        ratings = torch.FloatTensor(train_set[2].values).to(device)
        predict = model(users, items)
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = F.mse_loss(predict, ratings)
        loss += l2_lambda * l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print("epoch %d loss %.3f" % (i+1, loss.item()))

def train_OCCF(model, epoch=15, lr=0.01, wd=0):
    optimizer= torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    for i in range(epoch):
        model.train()
        users = torch.LongTensor(train_set[0].values).to(device)
        items = torch.LongTensor(train_set[1].values).to(device)
        ratings = torch.FloatTensor([1.0 for _ in range(len(train_set[2].values))]).to(device)
        predict = model(users, items)
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = F.mse_loss(predict, ratings)
        loss += l2_lambda * l2_reg        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def predict_R(model):
    return torch.matmul(model.embedding_user.weight, model.embedding_item.weight.t()).to(device)

def predict_score(model):
    model.eval()
    users = torch.LongTensor(test_set[0].values).to(device)
    items = torch.LongTensor(test_set[1].values).to(device)
    score = model(users, items).tolist()
    score = list(map(lambda x: 5 if x >= 5 else 1 if x <=1 else round(x), score))
    return score

model_OCCF = OCCF(num_user, num_item, embedding_size=32)
model_OCCF = model_OCCF.to(device)
train_OCCF(model_OCCF, epoch=300, lr=1e-3,wd=0.001)

predict_OCCF = predict_R(model_OCCF)
less_than = predict_OCCF < 0.3
train_set_adder = []
val_set_adder = []
base_score = 3
for user_index, line in enumerate(less_than):
    item_index = np.where(line.cpu() == True)[0]
    df_exist = np.array(df[(df[0] == user_index)][1])
    item_index = np.setdiff1d(item_index, df_exist)
    list_df = []
    for item_i in item_index:
        list_df.append([user_index, item_i, base_score, 0])
    if list_df:
        list_df = pd.DataFrame(list_df)
        df = pd.concat([df,list_df], axis=0)

train_set = df

if train_set_adder:
    train_adder_df = pd.DataFrame(train_set_adder)
    train_set = pd.concat([train_set, train_adder_df], axis=0)
if val_set_adder:
    val_adder_df = pd.DataFrame(val_set_adder)
    val_set = pd.concat([val_set, val_adder_df], axis=0)


model = MF_bias(num_user, num_item, mf_embedding_size=16, fn_embedding_size=32)
model = model.to(device)
train(model, epoch=960, lr=1e-3,wd=1e-7)

test_dataset = []
with open(test_file, "r") as f:
    texts = f.readlines()
    for text in texts:
        text = text.strip('\n').split('\t')
        test_dataset.append([int(text[0]), int(text[1]), int(text[2]), int(text[3])])

test_set = pd.DataFrame(test_dataset, dtype=np.int32)

test_set[0] = test_set[0].apply(lambda x: userId2idx.get(x, 0))
test_set[1] = test_set[1].apply(lambda x: itemId2idx.get(x, 0))

predict = predict_score(model)

with open(predict_file, 'w') as f:
    for idx, data in enumerate(test_dataset):
        f.write('{}\t{}\t{}\t{}\n'.format(data[0], data[1], predict[idx], data[3]))
