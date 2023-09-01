import torch
    

def check_gpus(ids):
    if torch.cuda.device_count() >= len(ids):
        for i in ids:
            if i >= torch.cuda.device_count():
                return False
        return True
    else:
        return False
    

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        print(f'cuda:{i} is not available, return cpu')
        return torch.device('cpu')
    