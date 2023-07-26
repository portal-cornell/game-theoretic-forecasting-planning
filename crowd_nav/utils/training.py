import torch

def train_planner(planner, train_loader, criterion, optimizer,\
         num_human=5, horizon=12, col_weight = 0.1, col_threshold = 0.6):
    planner.train()
    
    loss_sum_all, loss_sum_task, loss_sum_col = 0, 0, 0

    for robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist, neg_mask in train_loader:
        
        plan = planner(robot_states, human_states)

        plan_vel = plan[:, 1:, :] - plan[:, :-1, :]
        pos_vel = pos_seeds[:, 1:, :] - pos_seeds[:, :-1, :]
        
        plan_first_vel = plan[:, 0, :] - robot_states[:, :2]
        a, c = plan_first_vel.shape
        plan_first_vel = plan_first_vel.view((a, 1, c))
        pos_first_vel = pos_seeds[:, 0, :] - robot_states[:, :2]
        pos_first_vel = pos_first_vel.view((a, 1, c))
        
        plan_vel = torch.cat((plan_first_vel, play_vel), 1)
        pos_vel = torch.cat((pos_first_vel, pos_vel), 1)

        # loss_task = criterion(plan, pos_seeds) # uses x,y positions
        loss_task = criterion(plan_vel, pos_vel) # uses vx, vy velocities
        
        col_loss = col_weight*get_chomp_col_loss(plan, neg_seeds.permute([0, 2, 1, 3]))
        loss_sum_col += col_loss.item()

        loss = loss_task + col_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum_all += loss.data.item()
        loss_sum_task += loss_task.item()
        loss_sum_col += col_loss.item()

    num_batch = len(train_loader)

    return loss_sum_all / num_batch, loss_sum_task / num_batch, \
        loss_sum_col / num_batch

def train_forecaster(forecaster, train_loader, criterion, optimizer,\
    num_human=5, horizon=12, col_weight=0.1, col_threshold=0.6, is_train=True, device = 'cpu'):

    if is_train:
        forecaster.train()
    else:
        forecaster.eval()
    loss_sum_all = 0
    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist, neg_mask = batch
        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)
        if is_train:
            outputs, _ = forecaster(human_history)
        else:
            with torch.no_grad():
                outputs, _ = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(outputs, dim = 2)

        mask = neg_mask.permute([0,2,1,3])
        neg_seeds = neg_seeds.permute([0, 2, 1, 3])
        loss = criterion(forecasts*mask, neg_seeds*mask) # uses vx, vy velocities

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch

def train_planner_forecaster_disjointly(forecaster, planner, train_loader, criterion, optimizer,\
         num_human=5, horizon=12, col_weight = 0.0, col_threshold = 0.6, is_train=True, device = 'cpu'):

    forecaster.eval()
    if is_train:
        planner.train()
    else:
        planner.eval()
    
    loss_sum_all, loss_sum_task, loss_sum_col = 0, 0, 0

    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist, neg_mask = batch

        neg_hist = neg_hist.permute([0, 2, 1, 3])
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        with torch.no_grad():
            vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)

        if is_train:
            vel_plan = planner(robot_states, human_history, forecasts)
        else:
            with torch.no_grad():
                vel_plan = planner(robot_states, human_history, forecasts)

        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        loss = criterion(plan, pos_seeds) 
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum_all += loss.data.item()

    num_batch = len(train_loader)

    return loss_sum_all / num_batch

def get_chomp_col_loss(plan, forecasts, threshold = 0.6, eps = 0.2):
    distances = torch.linalg.norm(plan.unsqueeze(1)-forecasts, dim=-1)
    
    l1_col_mask = distances < threshold
    l2_col_mask = (distances > threshold) * (distances < (threshold+eps))

    distances = distances-threshold
    l1_loss = torch.sum((-distances+eps/2)*l1_col_mask, dim = -1)
    l2_loss = torch.sum((distances-eps)**2/(2*eps)*l2_col_mask, dim = -1)

    return torch.mean(torch.max(l1_loss+l2_loss, dim=-1)[0])

def train_planner_forecaster_together(forecaster, planner, train_loader, criterion, forecaster_optimizer,\
          planner_optimizer, num_human=5, horizon=12, col_weight = 0.1, 
          col_threshold = 0.6, planning_horizon=12, device = 'cpu'):

    forecaster.train()
    planner.train()
    
    loss_sum_all, loss_sum_planner, loss_sum_forecaster, loss_sum_col = 0, 0, 0, 0

    loss_types = ['total_loss', 'mse', 'cost_dif', 'plan_cost', 'forecast_cost']

    loss_dict = {
        'planner': {},
        'forecaster': {}
    }
    for lt in loss_types:
        loss_dict['planner'][lt] = 0
        loss_dict['forecaster'][lt] = 0

    for batch in train_loader:
        [b.to(device) for b in batch]
        robot_states, human_states, action_targets, pos_seeds, \
        neg_seeds, pos_hist, neg_hist, neg_mask = batch

        neg_hist = neg_hist.permute([0, 2, 1, 3])    
        neg_seeds = neg_seeds.permute([0, 2, 1, 3])    
        human_history = torch.cat([neg_hist, human_states[:, :, :2].unsqueeze(2)], dim = 2)

        ### Update planner
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()

        vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        plan_cost = get_chomp_col_loss(plan, forecasts)
        gt_cost = get_chomp_col_loss(pos_seeds, forecasts)
        
        planner_cost_dif = plan_cost - gt_cost
        planner_mse = criterion(plan, pos_seeds)
        planner_loss = planner_mse+col_weight*planner_cost_dif

        planner_loss.backward()
        planner_optimizer.step()

        losses = [planner_loss.item(), planner_mse.item(), 
        planner_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['planner'][lt] += loss

        ### Update Forecaster 
        planner_optimizer.zero_grad()
        forecaster_optimizer.zero_grad()
        
        vel_forecasts, features = forecaster(human_history)
        forecasts = human_states[:, :, :2] .unsqueeze(2) + torch.cumsum(vel_forecasts, dim = 2)
        vel_plan = planner(robot_states, human_history, forecasts)
        plan = robot_states[:, :2].unsqueeze(1) + torch.cumsum(vel_plan, dim=1)

        plan_cost = get_chomp_col_loss(plan, forecasts)
        gt_cost = get_chomp_col_loss(pos_seeds, forecasts)
        
        forecaster_cost_dif = plan_cost - gt_cost
        mask = neg_mask.permute([0, 2, 1, 3]) 
        forecaster_mse = criterion(forecasts*mask, neg_seeds*mask)
        forecaster_loss = forecaster_mse+col_weight*forecaster_cost_dif

        forecaster_loss.backward()
        forecaster_optimizer.step()
        
        losses = [forecaster_loss.item(), forecaster_mse.item(), 
        forecaster_cost_dif.item(), plan_cost.item(), gt_cost.item()]

        for loss, lt in zip(losses, loss_types):
            loss_dict['forecaster'][lt] += loss

    num_batch = len(train_loader)
    for model in ['planner', 'forecaster']:
        for lt in loss_types:
            loss_dict[model][lt] /= num_batch
    return loss_dict