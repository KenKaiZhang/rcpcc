import numpy as np

def cartesian_2_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # azimuth
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))  # elevation
    return r, theta, phi


def spherical_2_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def to_range_image(points, H=64, W=1024, v_fov=(-24.9, 2.0), h_fov=(0.0, 360.0)):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r, theta, phi = cartesian_2_spherical(x, y, z)

    # Wrap azimuth to [0, 2pi]
    theta = theta % (2 * np.pi)

    # Convert FOVs to radians
    v_fov_rad = np.radians(v_fov)
    h_fov_rad = np.radians(h_fov)

    # Normalize angles to [0, 1]
    azimuth = (theta - h_fov_rad[0]) / (h_fov_rad[1] - h_fov_rad[0])
    elevation = (phi - v_fov_rad[0]) / (v_fov_rad[1] - v_fov_rad[0])

    # Convert to pixel indices
    i = (azimuth * (W - 1)).astype(np.int32)
    j = ((1.0 - elevation) * (H - 1)).astype(np.int32)

    # Filter valid indices
    valid = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    i, j, r = i[valid], j[valid], r[valid]

    # Fill range image
    range_img = np.zeros((H, W), dtype=np.float32)
    range_img[j, i] = r

    return range_img


def to_point_cloud(range_image, v_fov=(-24.9, 2.0), h_fov=(0.0, 360.0)):
    H, W = range_image.shape
    
    rows, cols = np.where(range_image > 0)
    ranges = range_image[rows, cols]
    
    v_fov_rad = np.radians(v_fov)
    h_fov_rad = np.radians(h_fov)
    
    azimuth_norm = cols / (W - 1)
    elevation_norm = rows / (H - 1)
    
    thetas = azimuth_norm * (h_fov_rad[1] - h_fov_rad[0]) + h_fov_rad[0]
    phis = elevation_norm * (v_fov_rad[1] - v_fov_rad[0]) + v_fov_rad[0]
    
    x, y, z = spherical_2_cartesian(ranges, thetas, phis)
    point_cloud = np.vstack((x,y,z)).T
    
    return point_cloud