import numpy as np

def read_bin(file_path):
    """
    Returns only the (x,y,z) from .bin file
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]
