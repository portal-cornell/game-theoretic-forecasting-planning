import torch
import torch.nn as nn
from crowd_nav.utils.transform import MultiAgentTransform

class ForecastNetworkAttention(nn.Module):
    """ Policy network for imitation learning """
    def __init__(self, num_human=5, embedding_dim=64, hidden_dim=64, local_dim=32, history=8, horizon=12):
        super().__init__()
        self.horizon = horizon
        self.history = history
        self.num_human = num_human
        self.transform = MultiAgentTransform(num_human)
        self.hidden_dim = hidden_dim
        self.history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)

        self.human_encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.human_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.human_query = nn.Linear(hidden_dim, hidden_dim)
        self.human_key = nn.Linear(hidden_dim, hidden_dim)
        self.human_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.human_forecaster = nn.Linear(hidden_dim*2, horizon*2)

    
    def forward(self, crowd_obsv):
        if len(crowd_obsv.shape) < 4:
            crowd_obsv = crowd_obsv[None, ...]
        crowd_obsv = crowd_obsv.reshape(-1, self.history+1, 2)
        _, (hn, _) = self.history_encoder(crowd_obsv)
        history_encoded = hn[0].reshape(-1, self.num_human, self.hidden_dim)
        hidden = self.human_head(self.human_encoder(history_encoded))
        
        query = self.human_query(hidden)
        key = self.human_key(hidden)
        value = self.human_value(hidden)
        
        logits = torch.matmul(query, key.permute([0, 2, 1]))
        softmax = nn.functional.softmax(logits, dim=2)
        
        human_attentions = torch.matmul(softmax, value)
        
        forecast = self.human_forecaster(torch.cat([value, human_attentions], dim = -1))
        forecast = forecast.reshape(-1, self.num_human, self.horizon, 2)
        
        return forecast, torch.cat([value, human_attentions], dim=-1)
        
        

