import torch
from torch import nn

def compute_loss_and_update(states, eulers, actions, rewards, model, optimizer):
    """计算loss并更新模型"""
    if len(states) == 0:
        return
    
    # 将列表转换为张量
    states_tensor = torch.cat(states, dim=0)  # [T, 1, 400, 640]
    eulers_tensor = torch.stack(eulers, dim=0)  # [T, 3]
    actions_tensor = torch.stack(actions, dim=0)  # [T, 4]
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=states_tensor.device)  # [T]
    
    # 计算折扣回报
    gamma = 0.99
    returns = []
    running_return = 0
    for r in reversed(rewards_tensor):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    returns = torch.stack(returns)  # [T]
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化
    
    # 重新前向传播计算当前策略下的动作
    current_actions = []
    hx = None
    for t in range(len(states)):
        act, _, hx = model(states_tensor[t:t+1], eulers_tensor[t:t+1], hx)
        current_actions.append(act.squeeze(0))
    current_actions = torch.stack(current_actions)  # [T, 4]
    
    # 计算loss：让模型在奖励高的地方产生与历史动作相近的输出
    # 使用MSE loss，并用回报加权
    mse_loss = torch.nn.functional.mse_loss(current_actions, actions_tensor, reduction='none')
    weighted_loss = (mse_loss.mean(dim=1) * returns).mean()
    
    # 策略梯度风格的loss（最大化回报）
    # 注意取负因为我们要最小化loss
    policy_loss = -weighted_loss
    
    # 可选：添加动作平滑正则化
    if len(current_actions) > 1:
        smoothness_loss = torch.mean((current_actions[1:] - current_actions[:-1]) ** 2)
        total_loss = policy_loss + 0.01 * smoothness_loss
    else:
        total_loss = policy_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}, Mean Reward: {rewards_tensor.mean().item():.4f}")