class MetricsTracker:
    """ Utility class for tracking metrics. Uses Welford's algorithm for running mean.
    
    Example usage:
        tracker = MetricsTracker('loss', 'accuracy')
        tracker.update('loss', 0.5)
        tracker.update({'accuracy': 0.8, 'loss': 0.4})
        print(tracker.get_average('loss'))
        print(tracker)  # Displays all averages.
    """
    
    def __init__(self):
        self.counts = {}
        self.averages = {}

    def add(self, *metrics):
        for metric in metrics:
            if metric not in self.averages:
                self.averages[metric] = 0.0
                self.counts[metric] = 0

    def reset(self, *metrics):
        if not metrics:
            self.counts = {}
            self.averages = {}
        else:
            for metric in metrics:
                if metric in self.averages:
                    self.counts[metric] = 0
                    self.averages[metric] = 0.0
                else:
                    print(f"Variable {metric} is not being tracked.")
  
    def _update_metric(self, metric, value):
        self.counts[metric] += 1
        self.averages[metric] += (value - self.averages[metric]) / self.counts[metric]

    def update(self, metric, value=None):
        if isinstance(metric, dict):
            for k, v in metric.items():
                if k not in self.averages:
                    print(f"Variable {k} is not being tracked. Use add method to track.")
                    continue
                self.update(k, v)
        else:
            if metric not in self.averages:
                print(f"Variable {metric} is not being tracked. Use add method to track.")
                return
            self._update_metric(metric, value)

    def get_average(self, *metrics):
        metrics = {metric: self.averages.get(metric, None) for metric in metrics}
        if len(metrics) == 1:
            return next(iter(metrics.values()))
        return metrics
    
    def get_all_averages(self):
        return {var: avg for var, avg in self.averages.items() if self.counts[var] > 0}

    def __str__(self):
        averages = self.get_all_averages()
        return ", ".join(
            f"{name}: {avg:.4f}" for name, avg in averages.items()
        )

""" 
Taken from: https://github.com/lehduong/torch-warmup-lr/blob/master/torch_warmup_lr/wrappers.py
Copied due to install issues
 """
from torch.optim.lr_scheduler import _LRScheduler
import math 

class WarmupLR(_LRScheduler):
    def __init__(self, scheduler, init_lr=1e-3, num_warmup=1, warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".format(warmup_strategy))
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate 
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key !='_scheduler')}
        wrapped_state_dict = {key: value for key, value in self._scheduler.__dict__.items() if key != 'optimizer'} 
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])

    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end)/2.0*cos_out
    
    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end 

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start 

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate 
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(group['warmup_initial_lr'], 
                                                group['warmup_max_lr'],
                                                step_num/self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._step_count += 1 
        else:
            self._scheduler.step(*args)