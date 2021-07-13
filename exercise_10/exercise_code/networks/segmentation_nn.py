"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # input (N, 3, 240, 240)
        # output (N, 23, 240, 240)
        
        self.mobilenet = models.mobilenet_v2(pretrained=True).features # last layer (1280, 1, 1)
        # self.alexnet = models.alexnet(pretrained=True) # last layer softmax for 1000
        # resnet = models.resnet34(pretrained=True)
        
        for parameter in self.mobilenet.parameters():
            parameter.requires_grad = False
        
        """
        self.decoder = nn.Sequential(
            # input torch.Size([N, 1280, 8, 8])
            nn.Upsample(scale_factor=7, mode='bilinear'), # 1280 x 7 x 7
            nn.Conv2d(1280, 320, 1, stride=1), # 320 x 7 x 7
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 160 x 14 x 14
            nn.Conv2d(320, 160, 1, stride=1), # 64 x 14 x 14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 64 x 28 x 28
            nn.Conv2d(160, 128, 1, stride=1), # 32 x 28 x 28
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 32 x 56 x 56
            nn.Conv2d(128, 64, 1, stride=1), # 16 x 56 x 56
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), # 16 x 224 x 224
            nn.Conv2d(64, num_classes, 1, stride=1) # 3 x 224 x 224
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=30, mode='bilinear'),
            nn.Conv2d(1280, num_classes, 1, stride=1)
        )
 
        """
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), # 1280 x 16 x 16
            nn.Conv2d(1280, 320, 1, stride=1), # 320 x 7 x 7
            nn.Upsample(scale_factor=3, mode='bilinear'), # 320 x 48 x 48
            nn.Conv2d(320, 128, 1, stride=1), # 128 x 48 x 48
            nn.Upsample(scale_factor=5, mode='bilinear'), # 128 x 240 x 240
            nn.Conv2d(128, num_classes, 1, stride=1), # 23 x 240 x 240
        )
            

        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x_encoded = self.mobilenet(x)
        x = self.decoder(x_encoded)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x
    """
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.decoder.parameters(), lr=0.0001)
        return optim
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        out = self.forward(x)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = criterion(out, y)
        tensorboard_logs = {'train loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        out = self.forward(x)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = criterion(out, y)
        tensorboard_logs = {'val loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
     """   

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
