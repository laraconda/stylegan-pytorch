"""
Determines on what device (cuda/cpu) the networks will run. The `device` variable is
not declared on settings.py to improve readability when importing.

"""

import torch

#: Type of the chosen device where the heavy computation is going to take place.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
