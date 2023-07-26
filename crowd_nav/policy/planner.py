import torch
import torch.nn as nn

class Planner(nn.Module):
    def __init__(self, num_human=5, hidden_dim=64, horizon=12, history = 8, plan_horizon=12, 
            use_forecast=True):
        super().__init__()
        self.use_forecast = use_forecast
        self.horizon = horizon
        self.history = history
        self.plan_horizon = plan_horizon
        self.num_human = num_human
        self.hidden_dim = hidden_dim

        self.robot_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.history_encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        self.forecast_encoder = nn.LSTM(2, hidden_dim, batch_first=True)

        self.human_encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.robot_query = nn.Linear(hidden_dim, hidden_dim)
        self.human_key = nn.Linear(hidden_dim, hidden_dim)
        self.human_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.task_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.robot_plan = nn.Linear(hidden_dim*2, horizon*2)
        
    def forward(self, robot_state, human_history, forecasts=None):
        human_history = human_history.reshape(-1, self.history+1, 2)
        _, (hist_encoded, _) = self.history_encoder(human_history)
        history_encoded = hist_encoded[0].reshape(-1, self.num_human, self.hidden_dim)

        forecasts = forecasts.reshape(-1, self.horizon, 2)
        _, (fut_encoded, _) = self.forecast_encoder(forecasts)
        future_encoded = fut_encoded[0].reshape(-1, self.num_human, self.hidden_dim)

        human_obs = torch.cat([history_encoded, future_encoded], dim = -1)

        human_forecasts_emb = self.human_encoder(human_obs)

        robot_emb = self.robot_encoder(robot_state[:, :4])
        
        query = self.robot_query(robot_emb)
        key = self.human_key(human_forecasts_emb)
        value = self.human_value(human_forecasts_emb)

        logits = torch.matmul(query.view(-1, 1, self.hidden_dim), key.permute([0, 2, 1]))
        softmax = nn.functional.softmax(logits, dim=2)
        human_attentions = torch.matmul(softmax, value)
        
        reparam_robot_state = torch.cat([robot_state[:, -2:] - robot_state[:, :2], robot_state[:, 2:4]], axis=1)
        robot_task = self.task_encoder(reparam_robot_state)
        
        plan = self.robot_plan(torch.cat([robot_task, human_attentions.squeeze(1)], dim = -1))
        plan = plan.view(-1, self.horizon, 2)

        return plan
        
        

