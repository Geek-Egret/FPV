import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from matplotlib.animation import FuncAnimation


class pid:
    def __init__(self, kp, ki, kd, sigma_limit, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sigma_limit = sigma_limit
        self.output_limit = output_limit
        self.current_err = 0.0
        self.last_err = 0.0
        self.last_last_err = 0.0
        self.sigma_err = 0.0
        self.value = 0.0
        self.step = 0

    # 位置式PID
    def position(self, current, target):
        self.current_err = target - current
        # 积分项
        self.sigma_err += self.current_err
        # 位置式PID
        self.value = self.kp*self.current_err + self.ki*self.sigma_err + self.kd*(self.current_err-self.last_err)
        # 更新历史误差
        self.last_err = self.current_err
        self.last_last_err = self.last_err

        # 限幅
        if(self.value > self.output_limit):
            self.value = self.output_limit
        elif(self.value < -self.output_limit):
            self.value = -self.output_limit

        self.step += 1

        return self.value
    
    # 增量式PID
    def incremental(self, current, target):
        self.current_err = target - current
        # 增量式PID
        delta_value = self.kp*(self.current_err-self.last_err) + self.ki*self.current_err + self.kd*(self.current_err - 2*self.last_err +self.last_last_err)
        # 更新历史误差
        self.last_err = self.current_err
        self.last_last_err = self.last_err

        self.value += delta_value

        # print(self.current_err)
        # print(delta_value)

        # 限幅
        if(self.value > self.output_limit):
            self.value = self.output_limit
        elif(self.value < -self.output_limit):
            self.value = -self.output_limit

        self.step += 1

        return self.value
    
    # 复位
    def reset(self):
        self.current_err = 0.0
        self.last_err = 0.0
        self.last_last_err = 0.0
        self.sigma_err = 0.0
        self.value = 0.0
        self.step = 0