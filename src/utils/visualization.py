import numpy as np
from utils.utils import get_logger

logger = get_logger(__name__)


def save_image(array, save_path="tmp.png"):
    from PIL import Image
    Image.fromarray(array).save(save_path)
    logger.debug(f"saved {save_path}")


def save_array(array, save_name="tmp.npy"):
    np.save(save_name, array)


def visualize_voxel(voxel_maps, voxel_size=0.1):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for voxel_map in voxel_maps:
        ax.voxels(voxel_map.astype(bool),
                  facecolors='cyan',
                  edgecolors='gray',
                  alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization')
    plt.tight_layout()
    plt.show()


def visualize_voxel_from_file(path):
    """Load a .npy voxel/pixel map and visualize it."""
    load_map = np.load(path)
    if load_map.ndim == 2:
        load_map = np.stack([load_map] * load_map.shape[0], axis=2)
    visualize_voxel([load_map])


def visualize_point_cloud(ply_path, remove_outliers=True, nb_neighbors=30, std_ratio=1.0):
    """Load a .ply point cloud and visualize with optional outlier removal."""
    import open3d as o3d
    import matplotlib.pyplot as plt
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=1, label='raw')
    if remove_outliers:
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        clean_points = np.asarray(pcd_clean.points)
        ax.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], c='blue', s=1, label='filtered')
    ax.legend()
    plt.tight_layout()
    plt.show()


def save_map_to_image(array, path, save_path="tmp.png"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'red', 'blue', 'cyan'])
    display = array.copy()
    for (r, c) in path:
        display[int(r), int(c)] = 2
    display[int(path[0][0]), int(path[0][1])] = 3
    display[int(path[-1][0]), int(path[-1][1])] = 4
    plt.matshow(display, cmap=cmap)
    plt.colorbar()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug(f"Saved map image to {save_path}")


def save_video_images(controller_infos, keyword, save_path="tmp.mp4"):
    import cv2
    images = [controller_infos[k][keyword] for k in controller_infos]
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 24, (width, height))
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
