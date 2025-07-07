import numpy as np
from scipy.optimize import least_squares
from scipy.fftpack import dct


def surface_model_residuals(params, i, j, r):
    """
    Residual function for surface model: r̂ = -d / (a*i + b*j + c)
    Returns: residuals = observed_r - predicted_r
    """
    a, b, c, d = params
    denom = a * i + b * j + c
    eps = 1e-6  # To avoid division by zero
    pred_r = -d / (denom + eps)
    return r - pred_r  # residuals to be minimized


def fit_surface_model_noninverse(i_coords, j_coords, r_values):
    """
    Fit the surface model r̂ = -d / (a*i + b*j + c) using nonlinear least squares.
    Returns the fitted parameters (a, b, c, d), or None if fitting fails.
    """
    # Filter out invalid points (e.g., zero range)
    valid = r_values > 0
    i = i_coords[valid]
    j = j_coords[valid]
    r = r_values[valid]

    if len(r) < 6:
        return None  # Not enough valid points

    # Initial guess for parameters: small values
    initial_guess = [1e-3, 1e-3, 1.0, 1.0]

    # Run nonlinear least squares fitting
    result = least_squares(surface_model_residuals, initial_guess, args=(i, j, r))

    if not result.success:
        return None  # Optimization failed

    return result.x  # [a, b, c, d]


def surface_encode(range_image, block_size=4, delta_r=0.1):
    
    H, W = range_image.shape
    surface_blocks = []
    mask = np.zeros_like(range_image, dtype=bool)

    for row in range(0, H, block_size):
        for col in range(0, W, block_size):
            
            # Extract macroblock from range image
            block = range_image[row : row + block_size, col : col + block_size]
            block_h, block_w = block.shape

            # Skip blocks with too few valid points
            if np.count_nonzero(block) < block_h * block_w * 0.75:
                continue

            # Generate flattened (i, j, r) arrays for the block
            i_coords, j_coords = np.meshgrid(
                np.arange(block_w) + col, np.arange(block_h) + row
            )
            i_flat = i_coords.flatten()
            j_flat = j_coords.flatten()
            r_flat = block.flatten()

            # Fit the surface model directly using the paper’s formula
            params = fit_surface_model_noninverse(i_flat, j_flat, r_flat)
            if params is None:
                continue  # Skip if fitting failed

            a, b, c, d = params

            # Predict r̂ using fitted model
            denom = a * i_flat + b * j_flat + c
            pred_r = -d / (denom + 1e-6)  # Avoid divide-by-zero

            # Compute residuals and check if they are within threshold
            residuals = np.abs(r_flat - pred_r)
            if np.all(residuals < delta_r):
                # Accept this macroblock as a fitted surface
                surface_blocks.append((row, col, block_w, (a, b, c, d)))
                mask[row : row + block_h, col : col + block_w] = True

    return surface_blocks, mask


def unfit_encode(unfit_image, quantization_step=0.1):
    H, W = unfit_image.shape
    
    # Mask for unfit points
    unfit_mask = unfit_image > 0
    
    # Column-wise 1D DCT
    temp_col_dct = np.zeros_like(unfit_image)
    for col in range(W):
        col_data = unfit_image[:, col]
        values = col_data[col_data > 0] # Extract non-zero (unfit) points
        if len(values) != 0:
            dct_values = dct(values, norm="ortho")  # Apply 1D DCT
            temp_col_dct[:len(dct_values), col] = dct_values    # Store at TOP edge
    
    # Row-wise 1D DCT        
    sadct_coeffs = np.zeros_like(temp_col_dct)
    for row in range(H):
        row_data = temp_col_dct[row, :] # Extract non-zero elements
        values = row_data[row_data != 0]
        if len(values) != 0:
            dct_values_row = dct(values, norm="ortho")  # Apply 1D DCT
            sadct_coeffs[row, :len(dct_values_row)] = dct_values_row    # Store at LEFT edge
            
    quantized_coeffs = np.round(sadct_coeffs / quantization_step)
    
    return quantized_coeffs, unfit_mask
    
