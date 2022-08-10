import time
import numpy as np


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


class EvalTimer(object):
    def __init__(self):
        self.num = 0
        self.total = 0
        self.tmp = 0

    def tic(self):
        self.tmp = time.time()

    def toc(self):
        self.total = time.time() - self.tmp
        self.num += 1

    def print_time(self):
        print('Total %d images, take %f  FPS: %f'%(self.num, self.total, self.total/self.num))

def calculate_psnr(img1, img2):
    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    mse = np.average((img1 - img2) * (img1 - img2))
    psnr = 20 * np.log10(255.) - 10 * np.log10(mse)
    return psnr


class WarmUpScheduler(object):
    def __init__(self, optimizer, base_lr, target_lr, warm_up_iters):
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warm_up_iters = warm_up_iters
        self.iters = 0

    def step(self):
        self.iters += 1
        lr = ((self.target_lr - self.base_lr) / self.warm_up_iters) * self.iters + self.base_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


class PolynomialScheduler(object):
    def __init__(self, optimizer, base_lr, target_lr, power, max_epoch):
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.power = power
        self.max_epoch = max_epoch
        self.epoch = 0

    def step(self):
        lr = (self.base_lr - self.target_lr) * (1 - self.epoch/self.max_epoch)**self.power + self.target_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
        self.epoch += 1
