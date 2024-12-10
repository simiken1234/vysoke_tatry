from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import random


# def compress_image_to_colors(image_path, num_colors):
#     # Load the image
#     image = Image.open(image_path)
#     image_np = np.array(image)
#
#     # Reshape the image to a 2D array of pixels
#     pixels = image_np.reshape(-1, 3)
#
#     # Use KMeans clustering to compress the image to a defined number of colors
#     kmeans = KMeans(n_clusters=num_colors, random_state=42)
#     kmeans.fit(pixels)
#     compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
#
#     # Reshape back to the original image dimensions
#     compressed_image = compressed_pixels.reshape(image_np.shape).astype(np.uint8)
#
#     # Convert the compressed image back to an Image object
#     compressed_image_pil = Image.fromarray(compressed_image)
#
#     return compressed_image_pil, kmeans.cluster_centers_


def compress_image_with_dbscan_and_kmeans_downsampled(
        image_path, num_colors, dbscan_eps=10, dbscan_min_samples=5, dbscan_sample_size=10000):
    """
    Compress an image by performing DBSCAN clustering on a downsampled set of pixels,
    followed by KMeans clustering within each DBSCAN cluster.

    Parameters:
        image_path (str): Path to the image.
        num_colors (int): Total number of colors to reduce the image to.
        dbscan_eps (float): Maximum distance between points for DBSCAN.
        dbscan_min_samples (int): Minimum points to form a cluster in DBSCAN.
        dbscan_sample_size (int): Number of pixels to use for DBSCAN (downsampled subset).

    Returns:
        compressed_image_pil (PIL.Image): The compressed image.
        cluster_centers (np.ndarray): Array of cluster centers in RGB.
    """
    # Load the image and reshape pixels
    image = Image.open(image_path)
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    # Randomly downsample pixels for DBSCAN
    total_pixels = len(pixels)
    sampled_indices = random.sample(range(total_pixels), min(dbscan_sample_size, total_pixels))
    sampled_pixels = pixels[sampled_indices]

    # Perform DBSCAN clustering on the downsampled pixels
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    sampled_labels = dbscan.fit_predict(sampled_pixels)

    # Assign the rest of the pixels to DBSCAN clusters using nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(sampled_pixels)
    _, nearest_indices = nbrs.kneighbors(pixels)
    dbscan_labels = sampled_labels[nearest_indices.flatten()]

    # Unique DBSCAN clusters
    unique_labels = np.unique(dbscan_labels)
    num_dbscan_clusters = len(unique_labels[unique_labels != -1])  # Ignore noise (-1)

    # Handle case where DBSCAN clusters exceed required clusters
    if num_dbscan_clusters >= num_colors:
        compressed_pixels = np.array([
            np.mean(pixels[dbscan_labels == label], axis=0) for label in unique_labels if label != -1
        ])
        compressed_image = compressed_pixels[dbscan_labels]
        compressed_image = compressed_image.reshape(image_np.shape).astype(np.uint8)
        compressed_image_pil = Image.fromarray(compressed_image)
        return compressed_image_pil, compressed_pixels

    # Estimate volume of each DBSCAN cluster using convex hulls
    cluster_volumes = []
    cluster_points = []
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        cluster = pixels[dbscan_labels == label]
        cluster_points.append(cluster)
        if len(cluster) > 3:  # Convex hull requires at least 4 points
            hull = ConvexHull(cluster)
            cluster_volumes.append(hull.volume)
        else:
            cluster_volumes.append(0)  # Small cluster gets negligible volume

    # Normalize volumes and allocate subclusters
    total_volume = sum(cluster_volumes)
    subclusters_per_cluster = [
        max(1, math.ceil(volume / total_volume * num_colors)) if total_volume > 0 else 1
        for volume in cluster_volumes
    ]

    # Adjust to ensure total subclusters do not exceed num_colors
    while sum(subclusters_per_cluster) > num_colors:
        largest_cluster = np.argmax(subclusters_per_cluster)
        subclusters_per_cluster[largest_cluster] -= 1

    # Perform KMeans clustering within each DBSCAN cluster
    final_pixels = np.zeros_like(pixels)
    cluster_centers = []
    for cluster, num_subclusters in zip(cluster_points, subclusters_per_cluster):
        if num_subclusters > 1:
            kmeans = KMeans(n_clusters=num_subclusters, random_state=42)
            kmeans.fit(cluster)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            cluster_centers.extend(centers)
            for i, center in enumerate(centers):
                final_pixels[dbscan_labels == i] = center
        else:
            # Use mean color for single subcluster
            mean_color = np.mean(cluster, axis=0)
            final_pixels[dbscan_labels == i] = mean_color
            cluster_centers.append(mean_color)

    # Reshape back to the original image dimensions
    compressed_image = final_pixels.reshape(image_np.shape).astype(np.uint8)
    compressed_image_pil = Image.fromarray(compressed_image)

    return compressed_image_pil, np.array(cluster_centers)


def plot_original_and_clustered_pixels(image_path, num_colors, sample_rate=10):
    """
    Plots the original and clustered pixels in 3D RGB space.

    Parameters:
        image_path (str): Path to the image.
        num_colors (int): Number of clusters for KMeans.
        sample_rate (int): Plot every n-th pixel to reduce computational load.
    """
    # Load the image and reshape pixels
    image = Image.open(image_path)
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    # Subsample the pixels to reduce computational load
    sampled_pixels = pixels[::sample_rate]

    # Perform KMeans clustering on the full data
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # Use the sampled pixels to visualize clusters
    sampled_labels = kmeans.predict(sampled_pixels)
    cluster_centers = kmeans.cluster_centers_

    # Plot original pixels in RGB space
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(
        sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2],
        c=sampled_pixels / 255, s=1, alpha=0.5  # Use original colors
    )
    ax.set_title("Original Pixels in RGB Space")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")

    # Plot clustered pixels in RGB space
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(
        sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2],
        c=sampled_labels / num_colors, cmap='tab10', s=1, alpha=0.5  # Color by cluster
    )
    ax.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        c='black', s=100, label='Cluster Centers', alpha=1
    )
    ax.set_title(f"Clustered Pixels in RGB Space ({num_colors} Colors)")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
image_path = "C:/Users/Simon/PycharmProjects/vysoke_tatry/input_images/Claude-Monet-1899.jpg"
num_colors = 10  # Adjust the number of colors

#plot_original_and_clustered_pixels(image_path, num_colors, sample_rate=10)

#compressed_image, color_palette = compress_image_to_colors(image_path, num_colors)
compressed_image, color_palette = compress_image_with_dbscan_and_kmeans_downsampled(
    image_path, num_colors, dbscan_eps=5, dbscan_min_samples=50
)

# Display the compressed image
plt.imshow(compressed_image)
plt.axis('off')
plt.title(f"Compressed to {num_colors} Colors")
plt.show()

# Display the color palette
plt.figure(figsize=(8, 2))
for i, color in enumerate(color_palette):
    plt.subplot(1, num_colors, i + 1)
    plt.imshow([[color / 255]])  # Normalize colors to [0, 1] for display
    plt.axis('off')
plt.suptitle("Color Palette")
plt.show()

