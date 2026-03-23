import torch

class pid:
    def __init__(self, batch_size, kp, ki, kd, sigma_limit, output_limit, device):
        self._batch_size = batch_size
        self.kp = torch.full((self._batch_size, 1), kp, device=device).clone()
        self.ki = torch.full((self._batch_size, 1), ki, device=device).clone()
        self.kd = torch.full((self._batch_size, 1), kd, device=device).clone()
        self.sigma_limit = torch.full((self._batch_size, 1), sigma_limit, device=device).clone()
        self.output_limit = torch.full((self._batch_size, 1), output_limit, device=device).clone()
        self.current_err = torch.zeros(self._batch_size, 1, device=device).clone()
        self.last_err = torch.zeros(self._batch_size, 1, device=device).clone()
        self.last_last_err = torch.zeros(self._batch_size, 1, device=device).clone()
        self.sigma_err = torch.zeros(self._batch_size, 1, device=device).clone()
        self.value = torch.zeros(self._batch_size, 1, device=device).clone()

    # 位置式PID
    def position(self, current, target):
        self.current_err = target-current
        # 积分项
        self.sigma_err = self.sigma_err+self.current_err
        sigma = self.ki*self.sigma_err
        # 积分项限幅
        sigma = torch.clamp(sigma, -self.sigma_limit, self.sigma_limit)
        # 位置式PID
        self.value = self.kp*self.current_err+sigma+self.kd*(self.current_err-self.last_err)
        # 更新历史误差
        self.last_err = self.current_err
        self.last_last_err = self.last_err

        # 限幅
        self.value = torch.clamp(self.value, -self.output_limit, self.output_limit)
        return self.value
    
    # 增量式PID
    def incremental(self, current, target):
        self.current_err = target - current
        # 增量式PID
        delta_value = self.kp*(self.current_err-self.last_err)+self.ki*self.current_err+self.kd*(self.current_err-2*self.last_err +self.last_last_err)
        # 更新历史误差
        self.last_err = self.current_err
        self.last_last_err = self.last_err

        self.value = self.value+delta_value

        # 限幅
        self.value = torch.clamp(self.value, -self.output_limit, self.output_limit)
            
        return self.value
    
    # 复位
    def reset(self):
        self.current_err = torch.zeros_like(self.current_err)
        self.last_err = torch.zeros_like(self.last_err)
        self.last_last_err = torch.zeros_like(self.last_last_err)
        self.sigma_err = torch.zeros_like(self.sigma_err)
        self.value = torch.zeros_like(self.value)
