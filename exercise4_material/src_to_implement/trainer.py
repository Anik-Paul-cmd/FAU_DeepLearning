import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os



class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        
        
        if self._cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        os.makedirs('checkpoints', exist_ok=True)
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
                      x,                 # model input (or a tuple for multiple inputs)
                      fn,                # where to save the model (can be a file or file-like object)
                      export_params=True,# store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True, # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'], # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) # variable length axes
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        loss = self._crit(self._model(x), y.float())
        loss.backward()
        self._optim.step()
        return loss.item()
        
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        output = self._model(x)
        loss = self._crit(output, y.float())
        prediction = (output.detach().cpu().numpy() > 0.5).astype(int)
        return loss.item(), prediction
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train()
        average_loss = 0
        data_iter = iter(self._train_dl)
        num_batches = len(self._train_dl)
        while num_batches > 0:
            input, target = next(data_iter)
            if self._cuda:
                input, target = input.cuda(), target.cuda()
    
            loss = self.train_step(input, target)
            average_loss += loss / len(self._train_dl) 

            num_batches -= 1

        return average_loss
        
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()  

        average_loss = 0
        preds = []
        labels = []

        data_iter = iter(self._val_test_dl)
        num_batches = len(self._val_test_dl)

        while num_batches > 0:
            inputs, targets = next(data_iter)
            if self._cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with t.no_grad(): 
                loss, pred = self.val_test_step(inputs, targets)
                average_loss += loss / len(self._val_test_dl)

                if self._cuda:
                    targets = targets.cpu()

                preds.extend(pred)
                labels.extend(targets.cpu().numpy())

            num_batches -= 1

        score = f1_score(labels, preds, average='micro')

        return average_loss, score
        
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_l = []
        val_l = []
        epoch = 0
        while True:
            if epoch == epochs:
                break
            
            print('Epoch: ',(epoch+1))
            train_ls = self.train_epoch()
            val_ls, score = self.val_test()
            if val_l and val_ls < min(val_l):
                self.save_checkpoint(epoch)

            train_l.append(train_ls)
            val_l.append(val_ls)

                    
            patience = self._early_stopping_patience
            index = len(val_l)

            while patience > 0 and index > patience:
                recent_loss = val_l[-1]
                previous_loss = val_l[-(patience + 1)]
    
                if recent_loss > previous_loss:
                    break
    
                patience -= 1
                
            epoch += 1 
            print('Score: %.4f'%(score))

            
        return train_l, val_l
