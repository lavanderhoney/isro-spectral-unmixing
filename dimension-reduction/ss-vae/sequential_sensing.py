"""
Architecture of the LSTM model.
This model's forward method will assume the input is (batch_size,s^2,B)
Will give output of (batch_size, ld//4)
"""
# %%
import torch
import torch.nn as nn

class SequentialSensingNet(nn.Module):
    def __init__(self, n_bands: int, s:int, ld: int, lstm_layers: int=3) -> None:
        super().__init__()
        self.lstm_layers = lstm_layers
        self.ld = ld
        self.n_bands = n_bands
        self.s = s
        
        self.stack = nn.LSTM(input_size=n_bands, hidden_size=ld, 
                        num_layers=lstm_layers-1, batch_first=True)
        self.final = nn.LSTM(input_size=ld, hidden_size=ld//4, num_layers=1, batch_first=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=s*s)
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        x: (batch_size, s^2, n_bands)
        Returns: (batch_size, 1 , ld//4)
        """
        batch_size, seq_len, _ = x.shape
        
    
        # Pass through the first LSTM layer
        out, _ = self.stack(x) # out: batch_size, s^2, ld
        
        # Pass through the final LSTM layer
        out, _ = self.final(out) #out: batch_size, s^2, ld//4
        out = out.permute(0, 2, 1)  # Change shape to (batch_size, ld//4, s^2)
        latent_vector = self.activation(self.avg_pool(out)) # (batch_size, ld//4, 1)
        return torch.squeeze(latent_vector, dim=2)  # Change shape to (batch_size, ld//4)

# %%
# testing  
# x = torch.randn(32, 25, 109)
# lstm = SequentialSensingNet(109, 5, 12, 3)
# out = lstm(x)
# print(out.shape)

