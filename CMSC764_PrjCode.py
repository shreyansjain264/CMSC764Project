#Specify desired output location:
outdir = 'project/'

import os
if not os.path.exists('%snn10hCE'%(outdir)):
    os.makedirs('%snn10hCE'%(outdir))
if not os.path.exists('%snn10hMSE'%(outdir)):
    os.makedirs('%snn10hMSE'%(outdir))
if not os.path.exists('%snn10hSL1'%(outdir)):
    os.makedirs('%snn10hSL1'%(outdir))
    
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import math
import torch
import torchvision
import torchvision.transforms as transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = m.weight.data
    
class fnet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out):
        
        super(fnet, self).__init__()
        self.sp = torch.nn.Softplus()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, H5)
        self.linear6 = torch.nn.Linear(H5, H6)
        self.linear7 = torch.nn.Linear(H6, H7)
        self.linear8 = torch.nn.Linear(H7, H8)
        self.linear9 = torch.nn.Linear(H8, H9)
        self.linear10 = torch.nn.Linear(H9, H10)
        self.linear11 = torch.nn.Linear(H10, D_out)

    def forward(self, x):
        h_softplus1 = self.sp(self.linear1(x))
        h_softplus2 = self.sp(self.linear2(h_softplus1))
        h_softplus3 = self.sp(self.linear3(h_softplus2))
        h_softplus4 = self.sp(self.linear4(h_softplus3))
        h_softplus5 = self.sp(self.linear5(h_softplus4))
        h_softplus6 = self.sp(self.linear6(h_softplus5))
        h_softplus7 = self.sp(self.linear7(h_softplus6))
        h_softplus8 = self.sp(self.linear8(h_softplus7))
        h_softplus9 = self.sp(self.linear9(h_softplus8))
        h_softplus10 = self.sp(self.linear10(h_softplus9))
        y_pred = self.linear11(h_softplus10)
        return y_pred
    
def weight_update(m,mp):
    m.linear1.weight.data = m.linear1.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear1.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear1.weight.data)*mp
    m.linear2.weight.data = m.linear2.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear2.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear2.weight.data)*mp
    m.linear3.weight.data = m.linear3.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear3.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear3.weight.data)*mp
    m.linear4.weight.data = m.linear4.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear4.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear4.weight.data)*mp
    m.linear5.weight.data = m.linear5.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear5.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear5.weight.data)*mp
    m.linear6.weight.data = m.linear6.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear6.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear6.weight.data)*mp
    m.linear7.weight.data = m.linear7.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear7.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear7.weight.data)*mp
    m.linear8.weight.data = m.linear8.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear8.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear8.weight.data)*mp
    m.linear9.weight.data = m.linear9.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear9.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear9.weight.data)*mp
    m.linear10.weight.data = m.linear10.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear10.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear10.weight.data)*mp
    m.linear11.weight.data = m.linear11.weight.data + (torch.from_numpy(np.random.choice([-1, 1], size=m.linear11.weight.data.size())).type(torch.cuda.FloatTensor)*m.linear11.weight.data)*mp

# N is # of samples; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out = 40000, 3072, 600, 500, 450, 400, 350, 300, 250, 150, 100, 50, 10

#Normalize data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#Loader for training data. Loader is iterated through once later in code. Batch size = data sample size. 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Iterates through data set, and loads a different dataset each time.
dataiter = iter(trainloader)
images, labels = dataiter.next()

#########maximize for CE##############
#Datatype depends on use of cuda:
dtype1 = torch.cuda.FloatTensor 
dtype2 = torch.cuda.LongTensor 

iter_t = 500
loss_t = np.zeros(iter_t)
    
while max(-1*loss_t) < (10000):
    #Find max approximate
    model = fnet(D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out)

    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-2
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size = 100)

    iter_t = 500
    loss_t = np.zeros(iter_t)

    for t in range(iter_t):     
    
        # get the inputs      
        x = images.view(N,3072)
        y = labels
        x, y = Variable(x.type(dtype1)), Variable(y.type(dtype2), requires_grad=False)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = -loss_fn(y_pred, y)
        loss_t[t] = loss.data[0]


        if max(-1*loss_t) > (10e4) or np.isnan(loss_t[t]) == True:
            th = t
            break

        print(t, loss.data[0])

        #Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = -loss_fn(y_pred, y)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
    loss_t[th] = 0
np.savetxt("%snn10hCE/nn10hCE_%spic_4rp_%si_maxapprox.csv"%(outdir,N,iter_t), loss_t.T, delimiter=",")
torch.save(model.state_dict(), '%snn10hCE/CE_ogmodel.pth'%(outdir)) 

#######Optimize of models for CE#########
#rp = number of times to fit same model, iter_t = optimization iteration count, mtpl = % change in each weight
rp = 4
iter_t = 2000
mtpl = [0.01,0.05,0.1,0.2,0.5]
learning_rate = 1e-2
#empty arrays for results
loss_all_CE = np.zeros([rp,iter_t,len(mtpl)])

#copy previous model, adjust weights
new_mdl = copy.deepcopy(model)
weight_update(new_mdl,mtpl[0])

count = 0
for mp in mtpl:
    print(mp)
    for n in range(rp):

        loss_fn = torch.nn.CrossEntropyLoss()

        learning_rate = 1e-1
        optimizer = torch.optim.LBFGS(new_mdl.parameters(), lr=learning_rate, history_size = 100)
        loss_t = np.zeros(iter_t)

        for t in range(iter_t):

            # get the inputs      
            x = images.view(N,3072)
            y = labels
            x, y = Variable(x.type(dtype1)), Variable(y.type(dtype2), requires_grad=False)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = new_mdl(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            loss_t[t] = loss.data[0]

            if np.isnan(loss_t[t]) == True:
                break

            print(t, loss.data[0])
            #Backprop
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            def closure():
                optimizer.zero_grad()
                output = new_mdl(x)
                loss = loss_fn(y_pred, y)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)
        loss_all_CE[n,:,count] = loss_t
        torch.save(new_mdl.state_dict(),'%snn10hCE/CE_%s_%s.pth'%(outdir,mp,n))
        new_mdl = copy.deepcopy(model)
        weight_update(new_mdl,mp)
    np.savetxt("%snn10hCE/nn10hCE_%spic_4rp_%si_all_%s.csv"%(outdir,N,iter_t,mp), loss_all_CE[:,:,count].T, delimiter=",")
    count = count+1

#########maximize for MSE Loss############
#Datatype depends on use of cuda:
N, D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out = 40000, 3072, 600, 500, 450, 400, 350, 300, 250, 150, 100, 50, 1

dtype1 = torch.cuda.FloatTensor #Uncomment to use GPU
dtype2 = torch.cuda.LongTensor #Uncomment to use GPU

iter_t = 500
loss_t = np.zeros(iter_t)
while max(-1*loss_t) < (10000):
    #Find max approximate
    model = fnet(D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out)

    model.cuda() #Uncomment to use GPU

    loss_fn = torch.nn.MSELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size = 100)

    loss_t = np.zeros(iter_t)

    for t in range(iter_t):     

        # get the inputs      
        x = images.view(N,3072)
        y = labels.reshape(N,1)
        x, y = Variable(x.type(dtype1)), Variable(y.type(dtype1), requires_grad=False)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = -loss_fn(y_pred, y)
        loss_t[t] = loss.data[0]


        if max(-1*loss_t) > (50000) or np.isnan(loss_t[t]) == True:
            th = t
            break

        print(t, loss.data[0])
        t = t + 1
        #Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = -loss_fn(y_pred, y)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
    loss_t[th] = 0
    np.savetxt("%snn10hMSE/nn10hMSE_%spic_4rp_%si_maxapprox.csv"%(outdir,N,iter_t), loss_t.T, delimiter=",")
    torch.save(model.state_dict(), '%snn10hMSE/MSE_ogmodel.pth'%(outdir)) 

#####Build models for MSE############
#rp = number of times to fit same model, iter_t = optimization iteration count, mtpl = % change in each weight
rp = 4
iter_t = 5000
mtpl = [0.01,0.05,0.1,0.2,0.5]
learning_rate = 1e-2
#empty arrays for results
loss_all_MSE = np.zeros([rp,iter_t,len(mtpl)])

#copy previous model, adjust weights
new_mdl = copy.deepcopy(model)
weight_update(new_mdl,mtpl[0])

count = 0
for mp in mtpl:
    print(mp)
    for n in range(rp):

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.LBFGS(new_mdl.parameters(), lr=learning_rate, history_size = 100)
        loss_t = np.zeros(iter_t)

        for t in range(iter_t):
            
            # get the inputs      
            x = images.view(N,3072)
            y = labels.reshape(N,1)
            x, y = Variable(x.type(dtype1)), Variable(y.type(dtype1), requires_grad=False)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = new_mdl(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            loss_t[t] = loss.data[0]
            
            if np.isnan(loss_t[t]) == True:
                break
                
            print(t, loss.data[0])
            #Backprop
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            def closure():
                optimizer.zero_grad()
                output = new_mdl(x)
                loss = loss_fn(y_pred, y)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)
        loss_all_MSE[n,:,count] = loss_t
        torch.save(new_mdl.state_dict(),'%snn10hMSE/MSE_%s_%s.pth'%(outdir,mp,n))
        new_mdl = copy.deepcopy(model)
        weight_update(new_mdl,mp)
    np.savetxt("%snn10hMSE/nn10hMSE_%spic_4rp_%si_all_%s.csv"%(outdir,N,iter_t,mp), loss_all_MSE[:,:,count].T, delimiter=",")
    count = count+1

#######maximize for SmoothL1Loss#########

#Datatype depends on use of cuda:
dtype1 = torch.cuda.FloatTensor #Uncomment to use GPU
dtype2 = torch.cuda.LongTensor #Uncomment to use GPU
N, D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out = 40000, 3072, 600, 500, 450, 400, 350, 300, 250, 150, 100, 50, 1

iter_t = 500
loss_t = np.zeros(iter_t)
while max(-1*loss_t) < (10000):
    #Find max approximate
    model = fnet(D_in, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, D_out)

    model.cuda() #Uncomment to use GPU
    loss_fn = torch.nn.SmoothL1Loss()

    learning_rate = 1e-3
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size = 100)

    iter_t = 500
    loss_t = np.zeros(iter_t)

    for t in range(iter_t):     

        # get the inputs      
        x = images.view(N,3072)
        y = labels.reshape(N,1)
        x, y = Variable(x.type(dtype1)), Variable(y.type(dtype1), requires_grad=False)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = -loss_fn(y_pred, y)
        loss_t[t] = loss.data[0]


        if max(-1*loss_t) > (10e4) or np.isnan(loss_t[t]) == True:
            th = t
            break

        print(t, loss.data[0])
        #Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = -loss_fn(y_pred, y)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
    loss_t[th] = 0;
    np.savetxt("%snn10hSL1/nn10hSL1_%spic_4rp_%si_maxapprox.csv"%(outdir,N,iter_t), loss_t.T, delimiter=",")
    torch.save(model.state_dict(), '%snn10hSL1/SL1_ogmodel.pth'%(outdir)) 

#####Build models for SmoothL1Loss###############
#rp = number of times to fit same model, iter_t = optimization iteration count, mtpl = % change in each weight
rp = 4
iter_t = 2500
mtpl = [0.01,0.05,0.1,0.2,0.5]
learning_rate = 1e-1
#empty arrays for results
loss_all_SL1 = np.zeros([rp,iter_t,len(mtpl)])

#copy previous model, adjust weights
new_mdl = copy.deepcopy(model)

weight_update(new_mdl,mtpl[0])

count = 0
for mp in mtpl:
    print(mp)
    for n in range(rp):

        loss_fn = torch.nn.SmoothL1Loss()

        optimizer = torch.optim.LBFGS(new_mdl.parameters(), lr=learning_rate, history_size = 100)
        loss_t = np.zeros(iter_t)

        for t in range(iter_t):
            
            # get the inputs      
            x = images.view(N,3072)
            y = labels.reshape(N,1)
            x, y = Variable(x.type(dtype1)), Variable(y.type(dtype1), requires_grad=False)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = new_mdl(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            loss_t[t] = loss.data[0]
            
            if np.isnan(loss_t[t]) == True:
                break
                
            
            print(t, loss.data[0])
            #Backprop
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            def closure():
                optimizer.zero_grad()
                output = new_mdl(x)
                loss = loss_fn(y_pred, y)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)
        loss_all_SL1[n,:,count] = loss_t
        torch.save(new_mdl.state_dict(),'%snn10hSL1/SL1_%s_%s.pth'%(outdir,mp,n))
        new_mdl = copy.deepcopy(model)
        
        #update layer weights
        weight_update(new_mdl,mp)
    np.savetxt("%snn10hSL1/nn10hSL1_%spic_4rp_%si_all_%s.csv"%(outdir,N,iter_t,mp), loss_all_SL1[:,:,count].T, delimiter=",")
    count = count+1
