import numpy as np
import matplotlib.pyplot as plt


def visualize_range_image(range_image):
    plt.imshow(range_image, cmap="plasma")
    plt.title("Range Image")
    plt.xlabel("Azimuth")
    plt.ylabel("Elevation")
    plt.show()


def visualize_surface_encoding(range_image, surface_blocks, mask):
    H, W = range_image.shape
    reconstructed = np.zeros_like(range_image)

    # Reconstruct using surface model
    for row, col, length, (a, b, c, d) in surface_blocks:
        for i in range(row, row + 4):
            for j in range(col, col + length):
                if i >= H or j >= W:
                    continue
                denom = a * j + b * i + c
                if denom == 0:
                    continue
                reconstructed[i, j] = -d / (denom + 1e-6)

    # Compute error
    residual = np.abs(range_image - reconstructed)
    residual[~mask] = 0

    # Plot 2x2 without colorbars
    fig, axes = plt.subplots(2, 2, figsize=(25, 10))

    titles = [
        "Original LiDAR Image",
        "Fitted Surface Estimate",
        "Difference (Error)",
        "Surface-Encoded Regions",
    ]
    images = [range_image, reconstructed, residual, mask.astype(float)]
    cmaps = ["plasma", "plasma", "hot", "gray"]

    for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_unfit_encoding(unfit_image, quantized_coeffs, unfit_mask):
    _, axes = plt.subplots(3, 1, figsize=(25, 10))

    # Original Unfit Image
    axes[0].imshow(unfit_image, cmap="plasma", origin="upper")
    axes[0].set_title("Original Unfit Image")
    axes[0].axis("off")

    # SA-DCT Coefficients (before quantization, if available, or just the quantized for display)
    axes[1].imshow(quantized_coeffs, cmap="viridis", origin="upper")
    axes[1].set_title("Quantized SA-DCT Coefficients")
    axes[1].axis("off")
    
    # Unfit Mask (to show where unfit points originally were)
    axes[2].imshow(unfit_mask, cmap="gray", origin="upper")
    axes[2].set_title("Unfit Points Mask")
    axes[2].axis("off")


    plt.tight_layout()
    plt.show()
    
    
def visualize_stages(range_image, surface_blocks, unfit_image):
    H, W = range_image.shape
    reconstructed_surface = np.zeros_like(range_image)

    # Reconstruct surface for visualization
    for row, col, length, (a, b, c, d) in surface_blocks:
        for i in range(row, row + 4):
            for j in range(col, col + length):
                if i >= H or j >= W:
                    continue
                denom = a * j + b * i + c
                if denom == 0:
                    continue
                reconstructed_surface[i, j] = -d / (denom + 1e-6)

    _, axes = plt.subplots(3, 1, figsize=(25, 13)) # 3 row, 1 columns

    plot_data = [
        (range_image, "Original Range Image", "plasma"),
        (reconstructed_surface, "Fitted Surface Estimate", "plasma"),
        (unfit_image, "Unfit Image", "plasma"),
    ]

    for i, (img, title, cmap) in enumerate(plot_data):
        ax = axes[i] # Use axes[i] for 1D array of subplots
        im = ax.imshow(img, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.axis("off")
        # Add colorbar for range images
        if cmap == "plasma":
            plt.colorbar(im, ax=ax, label="Range (m)")

    plt.tight_layout()
    plt.show()
    
    
def visualize_point_cloud(points, title="Point Cloud"):
    
    if points.shape[0] == 0:
        print("No points to visualize")
        return
    
    fig = plt.figure(figsize=(25, 13))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points
    # Using a small alpha for better visibility in dense clouds
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Set equal aspect ratio for proper 3D representation
    # This ensures that units on all axes have the same length
    max_range = np.array([points[:,0].max()-points[:,0].min(),
                          points[:,1].max()-points[:,1].min(),
                          points[:,2].max()-points[:,2].min()]).max() / 2.0

    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()