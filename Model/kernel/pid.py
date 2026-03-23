import torch

class pid:
    def __init__(self, batch_size, kp, ki, kd, sigma_limit, output_limit):
        self._batch_size = batch_size
        self.kp = torch.full((self._batch_size, 1), kp)
        self.ki = torch.full((self._batch_size, 1), ki)
        self.kd = torch.full((self._batch_size, 1), kd)
        self.sigma_limit = torch.full((self._batch_size, 1), sigma_limit)
        self.output_limit = torch.full((self._batch_size, 1), output_limit)
        self.current_err = torch.zeros(self._batch_size, 1)
        self.last_err = torch.zeros(self._batch_size)
        self.last_last_err = torch.zeros(self._batch_size, 1)
        self.sigma_err = torch.zeros(self._batch_size)
        self.value = torch.zeros(self._batch_size, 1)

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
        self.current_err.zero_()
        self.last_err.zero_()
        self.last_last_err.zero_()
        self.sigma_err.zero_()
        self.value.zero_()
