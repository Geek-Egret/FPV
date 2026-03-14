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
    
    # 图表
    def set_plot(self, name="realtime pid values"):
        self.x_data = deque(maxlen=self.step)  # 时间戳
        self.y_data = deque(maxlen=self.step)  # 数据值

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label=name)

        self.ax.set_xlabel('time (dt)', fontsize=12)
        self.ax.set_ylabel('value', fontsize=12)
        self.ax.set_title(name, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 设置坐标轴范围
        self.ax.set_xlim(0, self.step)
        self.ax.set_ylim(-2, 2)

    def start_plot(self, dt):
        def animate(frame):
            # 获取新数据
            new_value = self.value
            # 添加时间戳
            current_x = len(self.x_data)
            self.x_data.append(current_x)
            self.y_data.append(new_value)
            if len(self.x_data) > 0:
                # 更新数据
                self.line.set_data(list(self.x_data), list(self.y_data))
                # 动态调整y轴范围
                if len(self.y_data) > 1:
                    y_min = min(self.y_data)
                    y_max = max(self.y_data)
                    padding = (y_max - y_min) * 0.1
                    self.ax.set_ylim(y_min - padding, y_max + padding)
                # 动态调整x轴范围
                if len(self.x_data) == self.step:
                    self.ax.set_xlim(self.x_data[0], self.x_data[-1])
            return self.line,
        
        self.ani = FuncAnimation(
            self.fig, animate, 
            interval=1000.0*dt, 
            blit=True, 
            cache_frame_data=False
        )
        plt.show(block=False)