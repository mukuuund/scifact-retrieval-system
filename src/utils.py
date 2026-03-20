import os
def ensure_dir(p):
    if not os.path.exists(p): os.makedirs(p)
