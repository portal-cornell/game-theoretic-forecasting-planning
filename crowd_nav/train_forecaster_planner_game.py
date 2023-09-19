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
import sys
torch.manual_seed(2020)

HISTORY, HORIZON, PLAN_HORIZON = 8, 8, 8

# COL_WEIGHT = round(float(sys.argv[1]), 2)
# print(COL_WEIGHT)
COL_WEIGHT = 0.1

demo_file = os.path.join('data/demonstration', 'data_imit.pt')
data_imit = torch.load(demo_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ForecastDataset(data_imit, None, device, 
                horizon=HORIZON, history=HISTORY, plan_horizon=PLAN_HORIZON)

validation_split = 0.3
train_loader, valid_loader = split_dataset(dataset, 128, 0.5, validation_split)

planner = torch.load(f'mle_planners/{HISTORY}_{HORIZON}/199.pth')
forecaster = torch.load(f'mle_forecasters/{HISTORY}_{HORIZON}/199.pth')
planner.to(device)
forecaster.to(device)

# optimize planner
param = list(planner.parameters()) 
planner_optimizer = optim.Adam(param, lr=1e-3)
planner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            planner_optimizer, 'min', patience=20, threshold=0.01,
            factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)

# optimize forecaster
param = list(forecaster.parameters()) 
forecaster_optimizer = optim.Adam(param, lr=1e-3)
forecaster_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    forecaster_optimizer, 'min', patience=20, threshold=0.01,
    factor=0.5, cooldown=20, min_lr=1e-5, verbose=True)

criterion = nn.MSELoss()

safe_dir = f'safe_planners/{HISTORY}_{HORIZON}_{COL_WEIGHT}'
pathlib.Path(safe_dir).mkdir(parents=True, exist_ok=True)

adv_dir = f'adv_forecasters/{HISTORY}_{HORIZON}_{COL_WEIGHT}'
pathlib.Path(adv_dir).mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=f'./logs/ioc/{HISTORY}_{HORIZON}_{COL_WEIGHT}')

for i in range(200):
    print(i+1)
    loss_dict = \
            train_planner_forecaster_together(
                forecaster, planner, train_loader, criterion, 
                forecaster_optimizer, planner_optimizer, col_weight=COL_WEIGHT, device = device)
    planner_scheduler.step(loss_dict['planner']['total_loss'])
    forecaster_scheduler.step(loss_dict['forecaster']['total_loss'])

    for model in ['planner', 'forecaster']:
        for lt in ['total_loss', 'mse', 'cost_dif', 'plan_cost', 'forecast_cost']:
            key = f'{model}_{lt}'
            writer.add_scalar(f'{key}', loss_dict[model][lt], i)

    torch.save(planner, f'{safe_dir}/{i}.pth')
    torch.save(forecaster, f'{adv_dir}/{i}.pth')