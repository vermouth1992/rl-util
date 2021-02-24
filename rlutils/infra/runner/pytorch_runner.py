import torch

from .base import BaseRunner, OffPolicyRunner, OnPolicyRunner


class PytorchRunner(BaseRunner):
    def setup_global_seed(self):
        super(PytorchRunner, self).setup_global_seed()
        torch.random.manual_seed(self.seeder.generate_seed())
        torch.cuda.manual_seed_all(self.seeder.generate_seed())
        torch.backends.cudnn.benchmark = True


class PytorchOffPolicyRunner(OffPolicyRunner, PytorchRunner):
    pass


class PytorchOnPolicyRunner(OnPolicyRunner, PytorchRunner):
    pass
