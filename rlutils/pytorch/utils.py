"""
Handle global pytorch device and data types
"""

import numpy as np
import torch

device = None


def set_default_device(d):
    global device
    print(f'Setting global Pytorch device to {d}')
    if d == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA is not available in this machine. Setting to cpu.')
            d = 'cpu'
    device = d


def get_cuda_device(id=None):
    if not torch.cuda.is_available():
        raise ValueError('Cuda is not available')
    else:
        if id is None:
            cuda_device = 'cuda'
        else:
            assert isinstance(id, int)
            total_device_count = torch.cuda.device_count()
            if id < 0 or id >= torch.cuda.device_count():
                raise ValueError(f'id {id} exceeds total device count {total_device_count}')
            else:
                cuda_device = f'cuda:{cuda}'
    return cuda_device


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def convert_dict_to_tensor(data, device=None):
    tensor_data = {}
    for key, d in data.items():
        if not isinstance(d, np.ndarray):
            d = np.array(d)
        tensor_data[key] = torch.as_tensor(d).to(device, non_blocking=True)
    return tensor_data


cpu = torch.device('cpu')
cuda = []
for i in range(torch.cuda.device_count()):
    cuda.append(torch.device(f'cuda:{i}'))


def print_version():
    print(f'Pytorch version: {torch.__version__}, git version: {torch.version.git_version}')


from torch.utils import data
from typing import Dict


class DictDataset(data.Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.batch_size = None
        for key, tensor in tensors.items():
            if self.batch_size is None:
                self.batch_size = tensor.shape[0]
            else:
                assert self.batch_size == tensor.shape[0]

    def __getitem__(self, item):
        return {key: data[item] for key, data in self.tensors.items()}

    def __len__(self):
        return self.batch_size


def create_dict_data_loader(tensors: Dict[str, torch.Tensor], batch_size):
    dataset = DictDataset(tensors)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
