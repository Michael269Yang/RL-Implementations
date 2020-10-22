import torch
import numpy as np
import random

# call seed functions
# seednum: seeding number (default - 1)
def seed(seednum = 1):
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.backends.cudnn.deterministic = True