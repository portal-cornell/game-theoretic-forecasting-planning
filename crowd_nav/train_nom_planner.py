from crowd_nav.utils.dataset import ForecastDataset, split_dataset
from crowd_nav.policy.forecast_attn import ForecastNetworkAttention
from crowd_nav.policy.planner import Planner
from utils.training import *
import torch, os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pathlib
torch.manual_seed(2020)


HISTORY, HORIZON, PLAN_HORIZON = 8, 8, 8
model_dir = f'mle_planners/{HISTORY}_{HORIZON}'
pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=f'./logs/{model_dir}')

demo_file = os.path.join('data/demonstration', 'data_imit.pt')
data_imit = torch.load(demo_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ForecastDataset(data_imit, None, device, 
                horizon=HORIZON, history=HISTORY, plan_horizon=PLAN_HORIZON)
validation_split = 0.3
train_loader, valid_loader = split_dataset(dataset, 128, 0.5, validation_split)
planner = Planner(horizon=HORIZON, history = HISTORY)
planner.to(device)

# optimize planner
param = list(planner.parameters()) 
planner_optimizer = optim.Adam(param, lr=1e-3)
planner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            planner_optimizer, 'min', patience=20, threshold=0.01,
            factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)

forecaster = torch.load(f'mle_forecasters/{HISTORY}_{HORIZON}/199.pth')

criterion = nn.MSELoss()

train_losses = []
for i in range(200):
    if (i+1) % 10 == 0:
        print(f'Epoch {i+1}')
    train_loss_all = \
            train_planner_forecaster_disjointly(
                forecaster, planner, train_loader, 
                criterion, planner_optimizer, col_weight=0.0, device = device)
    planner_scheduler.step(train_loss_all)
    writer.add_scalar('Train Loss All', train_loss_all, i)
    print(f'Planner = {round(train_loss_all, 3)}')

    if (i+1) % 10 == 0:
        validation_loss_all = \
            train_planner_forecaster_disjointly(
                forecaster, planner, valid_loader, 
                criterion, planner_optimizer, col_weight=0.0, is_train=False, device = device)
        writer.add_scalar('Validation Loss All', validation_loss_all, i)

    torch.save(planner, f'{model_dir}/{i}.pth')

print('1')