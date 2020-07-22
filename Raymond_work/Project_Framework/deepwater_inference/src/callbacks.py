import keras
from keras import backend as K
import numpy as np


class ReduceLRCallback(keras.callbacks.Callback):
    def __init__(self,
                 factor=0.3,
                 cooldown=5,
                 patience=5,
                 min_delta=1e-4,
                 min_lr=0,
                 verbose=0,
                 ):
        super().__init__()
        self.cooldown = cooldown
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.monitor = 'val_loss'
        self.cooldown_counter = 0
        self.wait = 0
        self.best = np.Inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = float(K.get_value(self.model.optimizer.lr))
        current = logs.get(self.monitor)
        if current is None:
            print('Reduce LR on plateau conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
                if self.verbose:
                    print(f'in cooldown {self.cooldown_counter}')

            elif self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                if self.verbose:
                    print(f'improvement to {self.best}')

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        print('\nEpoch %05d: reducing learning '
                              'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                else:
                    if self.verbose:
                        print(f'I am patient {self.wait}/{self.patience}')

    def in_cooldown(self):
        return self.cooldown_counter > 0


import keras
from keras import backend as K
import numpy as np


class StopRottingCallback(keras.callbacks.Callback):
    def __init__(self,
                 factor=0.3,
                 cooldown=5,
                 min_delta=1e-4,
                 patience=5,
                 ):
        super().__init__()
        self.cooldown = cooldown
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.monitor = 'val_loss'
        self.cooldown_counter = 0
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        # get learning rate
        lr = K.get_value(self.model.optimizer.lr)
        print(f"current learning rate: {lr}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = float(K.get_value(self.model.optimizer.lr))
        current = logs.get(self.monitor)
        if current is None:
            print('Reduce LR on plateau conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
                if self.verbose:
                    print(f'in cooldown {self.cooldown_counter}')

            elif self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                if self.verbose:
                    print(f'improvement to {self.best}')

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        print('\nEpoch %05d: reducing learning '
                              'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                else:
                    if self.verbose:
                        print(f'I am patient {self.wait}/{self.patience}')

    def in_cooldown(self):
        return self.cooldown_counter > 0
