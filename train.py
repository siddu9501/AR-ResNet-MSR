import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import *
from dataset import *

net = ResNet18()
if(torch.cuda.is_available()):
    net = net.cuda()


X, rX, Y = load_dataset()

N = Y.shape[0]
s = np.arange(X.shape[0])
np.random.shuffle(s)
shuffled_x = X[s]
shuffled_y = Y[s]

train_x = shuffled_x[0:int(0.8*N)]
train_y = shuffled_y[0:int(0.8*N)]
test_x = shuffled_x[int(0.8*N):]
test_y = shuffled_y[int(0.8*N):]

train_x = Variable(torch.from_numpy(train_x)).float().cuda()
train_x = train_x.permute(0,3,1,2).contiguous()
test_x = Variable(torch.from_numpy(test_x)).float().cuda()
test_x = test_x.permute(0,3,1,2).contiguous()
labels = Variable(torch.from_numpy(train_y)).long().cuda()
test_y = Variable(torch.from_numpy(test_y)).long().cuda()

BATCH_SIZE = 120
num_epochs = 200
num_batches = train_x.shape[0]/BATCH_SIZE
num_test_batches = test_x.shape[0]/BATCH_SIZE
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01,momentum = 0.9, weight_decay = 0.0001)

for epoch in range(num_epochs):
    if(epoch%75 == 74):
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
    running_loss = 0
    test_running_loss = 0
    for batch in range(int(num_test_batches)):
        batch_x = test_x[BATCH_SIZE*batch:BATCH_SIZE*(batch+1)]
        out = net(batch_x)
        loss = criterion(out,test_y[BATCH_SIZE*batch:BATCH_SIZE*(batch+1)].cuda())
        test_running_loss += loss.data[0]
#         test_running_loss = test_running_loss/num_test_batches
        #test_loss_list.append(test_running_loss)

    for batch in range(int(num_batches)):
        optimizer.zero_grad()
        batch_x = train_x[BATCH_SIZE*batch:BATCH_SIZE*(batch+1)]
        out = net(batch_x)
        loss = criterion(out,labels[BATCH_SIZE*batch:BATCH_SIZE*(batch+1)])
        running_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    print ('Epoch [%d/%d], Loss: %.4f, Test Loss: %.4f' %(epoch+1, num_epochs, running_loss, test_running_loss))


def pred(x):
    out = net(x)
    return np.argmax(out.data.cpu().numpy(), axis=1)

print(np.mean(pred(test_x) == test_y.data.cpu().numpy()))
print(np.mean(pred(train_x) == labels.data.cpu().numpy()))
