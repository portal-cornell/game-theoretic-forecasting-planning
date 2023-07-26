from crowd_nav.utils.dataset import ForecastDataset, split_dataset
from crowd_nav.snce.model import ProjHead, SpatialEncoder, EventEncoder
from crowd_nav.policy.forecast import ForecastNetwork
from crowd_nav.policy.forecast_attn import ForecastNetworkAttention
from crowd_nav.policy.planner import Planner
import torch, os
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
torch.manual_seed(2020)

HISTORY, HORIZON, PLAN_HORIZON = 8, 8, 8
COL_WEIGHT = 0.1

demo_file = os.path.join('data/demonstration', 'data_imit.pt')
data_imit = torch.load(demo_file)
device = 'cpu'
dataset_forecast = ForecastDataset(data_imit, None, device, horizon=HORIZON, 
                                   history=HISTORY, plan_horizon=PLAN_HORIZON)

validation_split = 0.3
train_loader, valid_loader = split_dataset(dataset_forecast, 128, 1.0, validation_split)

mle_forecaster = torch.load(f'mle_forecasters/{HISTORY}_{HORIZON}/199.pth')
mle_forecaster.eval()
mle_forecaster.to(device)

mle_planner = torch.load(f'mle_planners/{HISTORY}_{HORIZON}/40.pth')
mle_planner.eval()
mle_planner.to(device)

i = 10
adv_forecaster = torch.load(f'adv_forecasters/{HISTORY}_{HORIZON}_{COL_WEIGHT}/{i}.pth')
adv_forecaster.eval()
adv_forecaster.to(device)

safe_planner = torch.load(f'safe_planners/{HISTORY}_{HORIZON}_{COL_WEIGHT}/{i}.pth')
safe_planner.eval()
safe_planner.to(device)

def get_costs(plan, forecasts, threshold=0.6, eps = 0.2, plan_horizon=12):
    distances = torch.linalg.norm(plan.unsqueeze(1)-forecasts, dim=-1)

    is_col = distances < threshold
    is_col = is_col.reshape(-1, 5*HORIZON)
    is_col = torch.sum(is_col, dim=-1)
    is_col = is_col > 0

    l1_col_mask = distances < threshold
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)
    cost = torch.max(l1_loss+l2_loss, dim=-1)[0]

    return cost.tolist(), is_col.tolist()

def get_plan_errors(plan, future):
    errors = torch.linalg.norm(plan-future, dim = -1)
    ade = torch.mean(errors, dim=-1)
    fde = errors[:, -1]
    return ade.tolist(), fde.tolist()

def get_forecast_errors(forecast, future):
    errors = torch.linalg.norm(forecast-future, dim = -1)
    ade = torch.mean(errors, dim=-1)
    fde = errors[:, :, -1]
    return ade.tolist(), fde.tolist()

planners = ['mle', 'safe', 'futures']
forecasters = ['mle', 'adv', 'futures']
forecaster_models = ['mle', 'adv']
planner_models = ['mle', 'safe']

results_dict = {}
for planner in planners:
    for forecaster in forecasters:
        key = f'{planner}_{forecaster}'
        results_dict[f'{key}_cost'] = []
        results_dict[f'{key}_col'] = []

for planner in planner_models:
    results_dict[f'planner_{planner}_ade'] = []
    results_dict[f'planner_{planner}_fde'] = []
    
for forecaster in forecaster_models:
    results_dict[f'forecaster_{forecaster}_ade'] = []
    results_dict[f'forecaster_{forecaster}_fde'] = []

for robot_states, human_states, action_targets, pos_seeds, \
    neg_seeds, pos_hist, neg_hist, neg_mask in valid_loader:
    
    neg_hist = neg_hist.permute([0, 2, 1, 3])    
    neg_seeds = neg_seeds.permute([0, 2, 1, 3]) 
    mask = neg_mask.permute([0, 2, 1, 3]) 

    human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

    with torch.no_grad():
        vel_forecasts, features = mle_forecaster(human_history)
        mle_forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = mle_planner(robot_states, human_history, mle_forecasts)
        mle_plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        vel_forecasts, features = adv_forecaster(human_history)
        adv_forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = safe_planner(robot_states, human_history, adv_forecasts)
        safe_plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

    plans = {
        'futures': pos_seeds,
        'safe': safe_plan,
        'mle': mle_plan
    }

    forecasts = {
        'futures': neg_seeds,
        'adv': adv_forecasts,
        'mle': mle_forecasts
    }

    for planner in planners:
        for forecaster in forecasters:
            key = f'{planner}_{forecaster}'
            cost, col = get_costs(plans[planner], forecasts[forecaster])
            results_dict[f'{key}_cost'].extend(cost)
            results_dict[f'{key}_col'].extend(col)
    
    for planner in planner_models:
        ade, fde = get_plan_errors(plans[planner], pos_seeds)
        results_dict[f'planner_{planner}_ade'].extend(ade)
        results_dict[f'planner_{planner}_fde'].extend(fde)
    
    for forecaster in forecaster_models:
        ade, fde = get_forecast_errors(forecasts[forecaster]*mask, neg_seeds*mask)
        results_dict[f'forecaster_{forecaster}_ade'].extend(ade)
        results_dict[f'forecaster_{forecaster}_fde'].extend(fde)

for forecaster in forecasters:
    print(f'========{forecaster}')
    for planner in planners:
        key = f'{planner}_{forecaster}_cost'
        print(f'{planner} cost = {results_dict[key]}', end =" ")
    print()

torch.save(results_dict, f'{HISTORY}_{HORIZON}_{COL_WEIGHT}_new.pt')
