"""
Architecture of the CNN model for local sensing.
The forward method will assume the input is (batch_size, s, s, B)
It will give output of (batch_size, ld//4)
"""
#%%
import torch
import torch.nn as nn
from typing import List, Tuple
from collections import OrderedDict

class LocalSensingNet(nn.Module):
    def __init__(self, n_bands: int, s: int, ld: int, num_layers:int =3) -> None:
        super().__init__()
        self.n_bands: int = n_bands
        self.s: int= s
        self.ld: int= ld
        self.num_layers: int = num_layers
        self.kernel_sizes: List[int] = self.compute_kernel_sizes(self.s , self.num_layers)
        self.cnn_block = self.construct_conv_layers() 
        self.activation = nn.Sigmoid() 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (batch_size, s, s, B)
        x = x.permute(0, 3, 1, 2) #(batch_size, B, s, s,)
        x = self.activation(self.cnn_block(x)) # (batch_size, ld//4, 1, 1)
        return torch.squeeze(x, dim=(2,3)) #(batch_size, ld//4)
    def compute_kernel_sizes(self, s, num_layers) -> List[int]:
        kernels = []
        size = s #spatil sizes in the convolutional layers
        for i in range(num_layers-1):
            kernel_size = size//2 + 1
            size = size - (kernel_size - 1) # spatial size of resulting cube
            kernels.append(kernel_size)
        kernels.append(size)
        return kernels
    
    def __setup_layers(self, s:int, ld:int, index:int, n_bands:int) -> Tuple[str, nn.Conv2d]:
        """
        Setup a layer of 2dconv for the CNN.
        Returns layer name, and a Conv2d layer.
        All intermediate layers have same channel length of ld.
        The last layer's output channel will be ld//4.
        index starts from 0
        """
        input_size = n_bands if index==0 else ld #intermediate layer's channel size kept at ld
        output_size = ld if index< self.num_layers-1 else ld//4 #last layer will produce ld//4 channels
        
        conv_layer = nn.Conv2d(in_channels=input_size, 
                               out_channels=output_size,
                               kernel_size=self.kernel_sizes[index])
        return (f"Layer_{index}", conv_layer)
    
    def construct_conv_layers(self) -> nn.Sequential:
        """
        Return the CNN block to be used in the forward method
        """
        layers = OrderedDict()
        for idx in range(self.num_layers):
            layer_name, conv_layer = self.__setup_layers(self.s, self.ld, idx, self.n_bands)
            layers[layer_name] = conv_layer
            
        conv_block = nn.Sequential(layers)
        return conv_block
    
# %%

#testing
# x = torch.randn(32,5,5,109)

# cnn = LocalSensingNet(109, 5, 12, 3)
# out = cnn(x)
# print(out.shape)
# %%
