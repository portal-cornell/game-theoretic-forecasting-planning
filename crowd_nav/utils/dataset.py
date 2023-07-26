import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from crowd_nav.utils.transform import MultiAgentTransform

class ForecastDataset(Dataset):

    def __init__(self, data, action_space, device, vmin=0.0, 
    horizon=12, history=8, plan_horizon=12):
        '''
        Assumes planning horizon <= horizon
        '''
        robot_data = data[0]
        human_data = data[1]
        action_data = data[2]
        value_data = data[3]

        pos_seq = torch.zeros(robot_data.size(0), horizon, 2)
        neg_seq = torch.zeros(human_data.size(0), horizon, human_data.size(1), 2)
        neg_mask = torch.ones(human_data.size(0), horizon, human_data.size(1), 2)
        
        pos_hist = torch.zeros(robot_data.size(0), history, 2)
        neg_hist = torch.zeros(human_data.size(0), history, human_data.size(1), 2)
        
        num_humans = human_data.size(1)
        
        # remove samples at the end of episodes
        vx = robot_data[1:, 2]
        dx = robot_data[1:, 0] - robot_data[:-1, 0]
        diff = dx - vx * 0.25
        idx_done = (diff.abs() > 1e-6).nonzero(as_tuple=False)
        
        start_state = torch.Tensor([0, -4, 0.0, 0.0, 0, 4.0])
        idx_start = (torch.linalg.norm(robot_data-start_state, dim=1) < 1e-5).nonzero(as_tuple=False)
        
        print(idx_start.shape)
        idx_start = [int(x) for x in idx_start[:, 0]]
        idx_start.append(robot_data.size(0))
        
        
        cur_count = 0
        for start_idx, end_idx in zip(idx_start[:-1], idx_start[1:]):
            # print(human_data[start_idx, :, :2].shape)
            human_history = human_data[start_idx, :, :2].repeat(history, 1, 1)
            robot_history = robot_data[start_idx, :2].repeat(history, 1)
            
            for idx in range(start_idx, end_idx):
                if idx + horizon < end_idx:
                    human_future = human_data[idx+1:idx+horizon+1, :, :2]
                    robot_future = robot_data[idx+1:idx+horizon+1, :2]
                else:
                    ep_length = end_idx-idx-1
                    rem_length = horizon-ep_length
                    
                    human_future = human_data[idx+1:end_idx, :, :2]
                    human_future = torch.cat([human_future,\
                                             torch.zeros(rem_length, num_humans, 2)], dim=0)
                    neg_mask[cur_count, ep_length:, :2] = 0
                    
                    robot_future = robot_data[idx+1:end_idx, :2]
                    px, py, vx, vy = robot_data[end_idx-1, :4]
                    # print(px, py, vx, vy)
                    px, py = px+vx*0.25, py+vy*0.25
                    # print(px, py)
                    # input()
                    robot_end_pos = torch.Tensor([px, py])
                    robot_future = torch.cat([robot_future,\
                                             robot_end_pos.repeat(rem_length, 1)], dim=0)
            
                human_history = torch.roll(human_history, -1, 0)
                human_history[-1] = human_data[idx, :, :2]
                
                robot_history = torch.roll(robot_history, -1, 0)
                robot_history[-1] = robot_data[idx, :2]
                
                neg_hist[cur_count] = human_history
                pos_hist[cur_count] = robot_history
                neg_seq[cur_count] = human_future
                pos_seq[cur_count] = robot_future
                cur_count += 1
        
        # remove bad experience for imitation
        mask = (value_data > vmin).squeeze()
        pos_seq = pos_seq[:, :plan_horizon, :]

        self.robot_state = robot_data[mask].to(device)
        self.human_state = human_data[mask].to(device)
        self.action_target = action_data[mask].to(device)
        self.pos_state = pos_seq[mask].to(device)
        self.neg_state = neg_seq[mask].to(device)
        self.pos_hist = pos_hist[mask].to(device)
        self.neg_hist = neg_hist[mask].to(device)
        self.neg_mask = neg_mask[mask].long().to(device)

    def __len__(self):
        return self.robot_state.size(0)

    def __getitem__(self, idx):
        return self.robot_state[idx], self.human_state[idx], self.action_target[idx], \
    self.pos_state[idx], self.neg_state[idx], self.pos_hist[idx], self.neg_hist[idx], self.neg_mask[idx]

class ImitDataset(Dataset):

    def __init__(self, data, action_space, device, vmin=0.0, horizon=3):

        robot_data = data[0]
        human_data = data[1]
        action_data = data[2]
        value_data = data[3]

        # contrastive seeds, i.e., true positions of the robot (pos) and human neighbors (neg)
        pos_seq = torch.zeros(robot_data.size(0), horizon, 2)
        neg_seq = torch.zeros(human_data.size(0), horizon, human_data.size(1), 2)
        # remove samples at the end of episodes
        vx = robot_data[1:, 2]
        dx = robot_data[1:, 0] - robot_data[:-1, 0]
        diff = dx - vx * 0.25
        idx_done = (diff.abs() > 1e-6).nonzero(as_tuple=False)
        for t in range(horizon):
            dt = t + 1
            pos_seq[:-dt, t] = robot_data[dt:, :2]
            neg_seq[:-dt, t] = human_data[dt:, :, :2]
            for i in range(dt):
                pos_seq[idx_done-i, t] *= 0.0
                neg_seq[idx_done-i, t] *= 0.0

        # remove bad experience for imitation
        mask = (value_data > vmin).squeeze()

        self.robot_state = robot_data[mask].to(device)
        self.human_state = human_data[mask].to(device)
        self.action_target = action_data[mask].to(device)
        self.pos_state = pos_seq[mask].to(device)
        self.neg_state = neg_seq[mask].to(device)

    def __len__(self):
        return self.robot_state.size(0)

    def __getitem__(self, idx):
        return self.robot_state[idx], self.human_state[idx], self.action_target[idx], self.pos_state[idx], self.neg_state[idx]


class TrajDataset(Dataset):

    def __init__(self, data, length_pred, skip_pred, device):

        assert length_pred >= 1                 # TODO: multiple

        num_human = data[0].shape[1]
        state_dim = data[0].shape[2]

        self.transform = MultiAgentTransform(num_human)

        obsv = []
        target = []
        index = []

        for i, episode in enumerate(data):

            # remove starting and ending frame due to unpredictability
            speed = episode[:, :, -2:].norm(dim=2)
            valid = episode[(speed > 1e-4).all(axis=1)]

            length_valid = valid.shape[0]

            human_state = self.transform.transform_frame(valid)[:length_valid-length_pred*skip_pred]

            if length_valid > length_pred*skip_pred:
                upcome = []
                for k in range(length_pred):
                    propagate = episode[(k+1)*skip_pred:length_valid-(length_pred-k-1)*skip_pred, :, :2]
                    upcome.append(propagate)
                upcome = torch.cat(upcome, axis=2)
                obsv.append(human_state.view((length_valid-length_pred*skip_pred)*num_human, -1))
                target.append(upcome.view((length_valid-length_pred*skip_pred)*num_human, -1))
                index.append(torch.arange(5).repeat(length_valid-length_pred*skip_pred)+num_human*i)

        self.obsv = torch.cat(obsv).to(device)
        self.target = torch.cat(target).to(device)
        self.index = torch.cat(index).to(device)

    def __len__(self):
        return self.obsv.shape[0]

    def __getitem__(self, idx):
        return self.obsv[idx], self.target[idx]


def split_dataset(dataset, batch_size, percent_label=1.0, validation_split=0.3, is_random=False):

    dataset_size = len(dataset)
    split = int(validation_split * dataset_size)

    if is_random:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    train_indices, val_indices = indices[:int((dataset_size-split)*percent_label)], indices[-split:]

    logging.info("train_indices: %d - %d", train_indices[0].item(), train_indices[-1].item())
    logging.info("val_indices: %d - %d", val_indices[0].item(), val_indices[-1].item())

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

def get_val_indices(dataset, batch_size, percent_label=1.0, validation_split=0.3, is_random=False):

    dataset_size = len(dataset)
    split = int(validation_split * dataset_size)

    if is_random:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    train_indices, val_indices = indices[:int((dataset_size-split)*percent_label)], indices[-split:]

    return val_indices