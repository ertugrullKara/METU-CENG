import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class _Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_outputs):
        super(_Net, self).__init__()
        self.netsize = len(hidden_sizes)
        if self.netsize == 0:
            self.fc1 = nn.Linear(input_size, num_outputs)
        elif self.netsize == 1:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_sizes[0], num_outputs)
        elif self.netsize == 2:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_sizes[1], num_outputs)
        elif self.netsize == 3:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_sizes[2], num_outputs)
        elif self.netsize == 4:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(hidden_sizes[3], num_outputs)
    
    def forward(self, x):
        if self.netsize == 0:
            out = self.fc1(x)
        elif self.netsize == 1:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
        elif self.netsize == 2:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
        elif self.netsize == 3:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.relu3(out)
            out = self.fc4(out)
        elif self.netsize == 4:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.relu3(out)
            out = self.fc4(out)
            out = self.relu4(out)
            out = self.fc5(out)
        return out

class Network():
    def __init__(self, input_size=512, hidden_sizes=[100], num_outputs=1,
                num_epochs=5, batch_size=50, learning_rate=1e-4):
        self.train_data = np.load("train.npy")
        self.train_data_gt = np.load("train_gt.npy")
        tensor_x = torch.stack([torch.Tensor(i) for i in self.train_data])
        tensor_y = torch.stack([torch.Tensor([i]) for i in self.train_data_gt])
        self.train_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        
        self.val_data = np.load("valid.npy")
        self.val_data_gt = np.load("valid_gt.npy")
        tensor_x = torch.stack([torch.Tensor(i) for i in self.val_data])
        tensor_y = torch.stack([torch.Tensor([i]) for i in self.val_data_gt])
        self.val_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        
        self._train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=True)
        self._val_data_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                             batch_size=batch_size)
        self._net = _Net(input_size=input_size, hidden_sizes=hidden_sizes, num_outputs=num_outputs)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_sizes = hidden_sizes
    
    def load_model(self, model_name):
        pretrained_dict = torch.load(model_name)
        for k in pretrained_dict.keys():
            print(k, len(pretrained_dict[k]))
        self._net.load_state_dict(pretrained_dict)
    
    def predict(self, test_x=None):
        if test_x is None and test_t is None:
            tensor_x = torch.stack([torch.Tensor(i) for i in self.val_data])
        else:
            tensor_x = torch.stack([torch.Tensor(i) for i in test_x])
        loss_fn = nn.MSELoss()
        features = Variable(tensor_x.view(-1, 512))
        output = self._net(features)
        return output.data.numpy()
        
    
    def train(self, early_stop_epochs=10, verbose=1):
        """
        early_stop_epochs: How many epochs to wait to see if validation loss improves.
        verbose: [0, 1, 2], 0: No output, 1: Full output, 2: Only when stopping.
        """
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self._net.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.9) # momentum
        
        model_name = "models/bestmodel_"
        if not self.hidden_sizes:
            model_name += "{}_".format(0)
        for hidden in self.hidden_sizes:
            model_name += "{}_".format(hidden)
        model_name += "{}_".format(self.batch_size)
        model_name += "{}_".format(self.learning_rate)
        pytorch_extension = ".pt"
        
        best_val_loss = np.inf
        early_stop = 0
        train_loss_hist = []
        val_loss_hist = []
        
        for epoch in range(self.num_epochs):
            tr_loss = 0
            tr_cnt = 0
            for i, (features, labels) in enumerate(self._train_data_loader):
                features = Variable(features.view(-1, 512))
                labels = Variable(labels)
                
                optimizer.zero_grad()
                output = self._net(features)
                loss = loss_fn(output, labels)
                tr_loss += loss.data[0]
                tr_cnt += 1
                loss.backward()
                optimizer.step()
            tr_loss /= tr_cnt
            val_loss = 0
            val_cnt = 0
            for i, (features, labels) in enumerate(self._val_data_loader):
                features = Variable(features.view(-1, 512))
                labels = Variable(labels)
                output = self._net(features)
                valloss = loss_fn(output, labels)
                val_loss += valloss.data[0]
                val_cnt += 1
            val_loss /= val_cnt
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(val_loss)
            if verbose == 1:
                print ('Epoch [%d/%d]  AvgTraining Loss: %.4f' %(epoch+1, self.num_epochs, tr_loss), end="\t\t")
                print ('Epoch [%d/%d]  AvgValidation Loss: %.4f' %(epoch+1, self.num_epochs, val_loss), end="\t")
            if val_loss < best_val_loss:
                if verbose == 1:
                    print ("Validation loss improved.", end="\n\n")
                best_val_loss = val_loss
                torch.save(self._net.state_dict(), model_name + pytorch_extension)
                early_stop = 0
            else:
                if verbose == 1:
                    print("\n")
                if early_stop > early_stop_epochs:
                    if verbose > 0:
                        print("Model did not improve within the last {} epochs. Stopping training.".format(str(early_stop_epochs)))
                        print("TrainedModelName: {}, BestValAcc: {}".format(model_name.split("/")[1], best_val_loss))
                    np.save(model_name, {"train_loss_hist": train_loss_hist,
                            "val_loss_hist": val_loss_hist})
                    return train_loss_hist, val_loss_hist
                early_stop += 1