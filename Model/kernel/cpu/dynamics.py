import numpy as np
import torch 
import util

class solver:
    """
        mass:kg
        pos:init position torch.tensor(x, y, z)
        quat:init quaternion torch.tensor(w, x, y, z)
    """
    def __init__(self, dt, mass, init_pos, init_quat):
        self.dt = dt
        self.mass = mass
        self.pos = init_pos
        self.quat = init_quat

    def get_pos(self):
        return self.pos
    
    def get_quat(self):
        return self.quat
    
    def set_acc(self, acc_x, acc_y, acc_z):
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.is_acc_set = True

    def set_vel(self, vel_x, vel_y, vel_z):
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.is_vel_set = True

    def update_pos_quat(self, ):
        
