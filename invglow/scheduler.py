class ScheduledOptimizer(object):
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer


    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, *args, **kwargs):
        self.optimizer.add_param_group(*args, **kwargs)


    def state_dict(self,):
        return self.optimizer.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.optimizer.load_state_dict(*args, **kwargs)

