import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils
from skimage.transform import resize
from evaluate import main


class _ConvNet(nn.Module):
    def __init__(self, first_kernel=16, first_kernel_size=5, second_kernel=24, second_kernel_size=3, last_kernel_size=3):
        super(_ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, first_kernel, first_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(first_kernel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(first_kernel, second_kernel, second_kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(second_kernel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Conv2d(second_kernel, 2, last_kernel_size, stride=1, padding=1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.permute(0, 2, 3, 1)
        return out


def load_data(load_from, img_folder, gt_img_folder):
    ims = list(map(str.strip, open(load_from, 'r').readlines()))
    tr = [] 
    tr_gt = []
    for im in ims:
        img = utils.read_image(img_folder + "/" + im)
        if gt_img_folder:
            gt_img = utils.read_image(gt_img_folder + "/" + im)
        L, _ = utils.cvt2Lab(img)
        if gt_img_folder:
            _, ab = utils.cvt2Lab(gt_img)
        tr.append(L)
        if gt_img_folder:
            tr_gt.append(ab)
    return np.array(tr), np.array(tr_gt)
    
    
class ConvNet():
    def __init__(self, learning_rate=1e-5, batch_size=64, reg=1e-3, num_epochs=500, first_kernel=16, first_kernel_size=5, second_kernel=24, second_kernel_size=3, last_kernel_size=3):
        self.batch_size = batch_size
        self.reg = reg
        try:
            self.train_data = np.load("train.npy")
            self.train_data_gt = np.load("train_gt.npy")
        except:
            self.train_data, self.train_data_gt = load_data("train.txt", "gray", "color_64")
            np.save("train.npy", self.train_data)
            np.save("train_gt.npy", self.train_data_gt)
        try:
            self.val_data = np.load("valid.npy")
            self.val_data_gt = np.load("valid_gt.npy")
        except:
            self.val_data, self.val_data_gt = load_data("valid.txt", "gray", "color_64")
            np.save("valid.npy", self.val_data)
            np.save("valid_gt.npy", self.val_data_gt)
        try:
            self.test_data = np.load("test.npy")
            # self.test_data_gt = np.load("te_gt.npy")
        except:
            self.test_data, _ = load_data("test.txt", "test_gray", None)
            np.save("test.npy", self.test_data)
        # self.dmin = np.min(np.concatenate((self.train_data, self.val_data)))
        # self.dmax = np.max(np.concatenate((self.train_data, self.val_data)))
        self.train_data = (self.train_data) / 100
        self.val_data = (self.val_data) / 100
        self.test_data = (self.test_data) / 100
        
        # self.dgtmin = np.min(np.concatenate((self.train_data_gt, self.val_data_gt)), axis=(0,1,2))
        # self.dgtmax = np.max(np.concatenate((self.train_data_gt, self.val_data_gt)), axis=(0,1,2))
#         self.train_data_gt = (self.train_data_gt + 128) / 255
#         self.val_data_gt = (self.val_data_gt + 128) / 255
        
        tensor_x = torch.stack([torch.Tensor(i) for i in self.train_data])# .cuda()
        tensor_y = torch.stack([torch.Tensor(i) for i in self.train_data_gt])# .cuda()
        self.train_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        
        tensor_x = torch.stack([torch.Tensor(i) for i in self.val_data])# .cuda()
        tensor_y = torch.stack([torch.Tensor(i) for i in self.val_data_gt])# .cuda()
        self.val_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        
        tensor_x = torch.stack([torch.Tensor(i) for i in self.test_data])# .cuda()
        tensor_y = torch.stack([torch.Tensor(i) for i in self.test_data])# .cuda()
        self.test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        
        self._train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                             batch_size=self.batch_size,
                                                             shuffle=True)
        self._val_data_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                           batch_size=self.batch_size)
        
        self._test_data_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                            batch_size=self.batch_size)
        
        self.net = _ConvNet(first_kernel=first_kernel, first_kernel_size=first_kernel_size, second_kernel=second_kernel, 
                            second_kernel_size=second_kernel_size, last_kernel_size=last_kernel_size)
        self.net = self.net# .cuda()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.pytorch_extension = ".pt"
        self.model_name = "bestmodel_"
        self.model_name += "{}_{}_{}".format(learning_rate, batch_size, reg)
       
    def load_model(self, model_name):
        pretrained_dict = torch.load(model_name)
        self.net.load_state_dict(pretrained_dict)
        self.net# .cuda()
        
    def predict(self, test=False):
        if not test:
          tensor_x = torch.stack([torch.Tensor(i) for i in self.val_data])# .cuda()
          x = Variable(tensor_x.unsqueeze(1))
        else:
          tensor_x = torch.stack([torch.Tensor(i) for i in self.test_data])# .cuda()
          x = Variable(tensor_x.unsqueeze(1))
        predictions = self.net(x)

        cielab = np.zeros((100, 256, 256, 3))
        #predictions[...,0] = predictions[...,0]  * ((self.dgtmax - self.dgtmin) + self.dgtmin)[0]
        #predictions[...,1] = predictions[...,1]  * ((self.dgtmax - self.dgtmin) + self.dgtmin)[1]
#         predictions[..., 0:2] = predictions[..., 0:2] * 255 - 128
        predictions = predictions.permute(0, 3, 1, 2)
        up = nn.Upsample(scale_factor=4)
        cielab[..., 1:] = up(predictions).permute(0, 2, 3, 1).cpu().detach().numpy()
        if not test:
            cielab[..., 0] = self.val_data * 100# * (self.dmax - self.dmin) + self.dmin
        else:
            cielab[..., 0] = self.test_data * 100# * (self.dmax - self.dmin) + self.dmin
        predictions = []
        for img in cielab:
            predictions.append(utils.cvt2rgb(img)*255.)
        predictions = np.array(predictions)
        return predictions
        
    def train(self, early_stop_epochs=20, verbose=1):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.learning_rate,
                                       weight_decay=self.reg, momentum=0.9)
        
        best_val_loss = np.inf
        train_loss_hist = []
        val_loss_hist = []
        lr_decay = 0
        
        for epoch in range(self.num_epochs):
            tr_loss = 0
            tr_cnt = 0
            for (x, y) in self._train_data_loader:
                x.unsqueeze_(1)
                # y.unsqueeze_(1)
                b_x = Variable(x)
                b_y = Variable(y)
                output = self.net(b_x)
                loss = loss_fn(output, b_y)
                
                tr_loss += loss.item()
                tr_cnt += 1
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tr_loss /= tr_cnt
            
            val_loss = 0
            val_cnt = 0
            for (x,y) in self._val_data_loader:
                x.unsqueeze_(1)
                # y.unsqueeze_(1)
                b_x = Variable(x)
                b_y = Variable(y)
                output = self.net(b_x)
                valloss = loss_fn(output, b_y)
                
                val_loss += valloss.item()
                val_cnt += 1
            val_loss /= val_cnt
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(val_loss)
            
            predictions = self.predict()
            np.save("estimations_val.npy", predictions)
            main(["estimations_val.npy", "valid.txt"])
            lr_decay += 1
            if lr_decay == 10:
              lr_decay = 0
              self.learning_rate *= 0.95
              optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.learning_rate,
                                       weight_decay=self.reg, momentum=0.9)
            if verbose == 1:
                print ('Epoch [%d/%d]  AvgTraining Loss: %.4f' %(epoch+1, self.num_epochs, tr_loss), end="\t\t")
                print ('Epoch [%d/%d]  AvgValidation Loss: %.4f' %(epoch+1, self.num_epochs, val_loss), end="\t")
            if val_loss < best_val_loss:
                if verbose == 1:
                    print ("Validation loss improved.", end="\n\n")
                best_val_loss = val_loss
                torch.save(self.net.state_dict(), self.model_name + self.pytorch_extension)
                early_stop = 0
            else:
                if verbose == 1:
                    print("\n")
                if early_stop > early_stop_epochs:
                    if verbose > 0:
                        print("Model did not improve within the last {} epochs. Stopping training.".format(str(early_stop_epochs)))
                        print("TrainedModelName: {}, BestValAcc: {}".format(self.model_name, best_val_loss))
                    np.save(self.model_name, {"train_loss_hist": train_loss_hist,
                            "val_loss_hist": val_loss_hist})
                    return train_loss_hist, val_loss_hist
                early_stop += 1
        return train_loss_hist, val_loss_hist

"""
if __name__ == "__main__":
    import scipy.misc
    import warnings
    warnings.filterwarnings('ignore')

    best_acc = 0
    best_net = None
    best_tr_loss_hist = None
    best_val_loss_hist = None
    lr = 1e-6
                            net = ConvNet(learning_rate=lr, first_kernel=fk, 
                                        first_kernel_size=fksize, second_kernel=sk,
                                        second_kernel_size=sksize, 
                                        last_kernel_size=lastsize)
                            net.load_model("models/bestmodel_48_11_48_3_1_0.001.pt")
                            tr_loss_hist, val_loss_hist = net.train()
                        except:
                            continue

                        net.load_model(net.model_name+net.pytorch_extension)
                        predictions = net.predict()
                        np.save("estimations_val.npy", predictions)
                        acc = main(["estimations_val.npy", "valid.txt"])
                        # scipy.misc.imsave('outfile.jpg', predictions[0])
                        if acc > best_acc:
                            best_acc = acc
                            best_net = net
                            best_tr_loss_hist = tr_loss_hist
                            best_val_loss_hist = val_loss_hist                         
"""
