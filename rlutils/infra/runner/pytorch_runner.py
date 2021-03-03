import rlutils.pytorch.utils as ptu
import torch

from .base import BaseRunner, OffPolicyRunner, OnPolicyRunner


class PytorchRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(PytorchRunner, self).__init__(*args, **kwargs)
        ptu.set_device('cuda')

    def setup_global_seed(self):
        super(PytorchRunner, self).setup_global_seed()
        torch.random.manual_seed(self.seeder.generate_seed())
        torch.cuda.manual_seed_all(self.seeder.generate_seed())
        torch.backends.cudnn.benchmark = True


class PytorchOffPolicyRunner(OffPolicyRunner, PytorchRunner):
    pass


class PytorchOnPolicyRunner(OnPolicyRunner, PytorchRunner):
    pass
