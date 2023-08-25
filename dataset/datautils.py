import numpy as np
from scipy.sparse import csr_matrix
import laspy
from sklearn.neighbors import NearestNeighbors
import torch

from einops import rearrange


def bubble_to_laz_file(bubble_path, output_file_name):

    data = torch.load(bubble_path)
    point_tokens = data[0]
    points = rearrange(point_tokens, 'rings points xyz ->  (rings points) xyz')

    labels = data[1]
    labels = rearrange(labels, 'rings labels -> (rings labels)')
    save_as_laz_file(points, output_file_name, labels)

def save_as_laz_file(points, filename, classification=None):
    las = laspy.create(file_version="1.4", point_format=6)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if classification is not None:
        las.classification = classification
    las.write(filename)


def get_data_from_laz_file(laz_file, classification=True):
    las = laspy.read(laz_file)
    x = las.x
    y = las.y
    z = las.z

    if classification:
        classification = las.classification
        return np.stack([x, y, z], axis=1), classification
    else:
        return np.stack([x, y, z], axis=1)


def down_sample_point_data(voxel_size, pc_input):
    """
    voxel_size : float
    pc_input : numpy array of shape [n_points x 6]
        Contains [x,y,z,r,g,b] info on the input point cloud

    Returns
    -------
    pc_down_sampled : numpy array of shape [n_points_down_sampled_pc x 6]
    voxel_map : list containing n_points_pc_down_sampled times sub integer lists
        Each sub list contains the indices of the pc_input that were averaged to determine the position and colour
        of each respective point in the pc_down_sampled
    """

    min_coord_vector = np.amin(pc_input[:, 0:3], axis=0)
    n_points_original_pc = len(pc_input)

    voxel_coordinates_original_pc = (pc_input[:, 0:3] - min_coord_vector) // voxel_size
    down_sampled_voxel_coordinates, indices = np.unique(voxel_coordinates_original_pc, axis=0, return_inverse=True)
    n_points_down_sampled_pc = len(down_sampled_voxel_coordinates)
    del voxel_coordinates_original_pc, down_sampled_voxel_coordinates

    # variables required to build the sparse matrix for determining the pc_down_sampled
    row_ind = []
    col_ind = []

    voxel_map = [[] for _ in range(n_points_down_sampled_pc)]
    for index_original_pc in range(len(indices)):
        index_down_sampled_pc = indices[index_original_pc]
        voxel_map[index_down_sampled_pc].append(index_original_pc)
        row_ind.append(index_down_sampled_pc)
        col_ind.append(index_original_pc)

    sparse_matrix = csr_matrix((np.ones(n_points_original_pc), (row_ind, col_ind)),
                               shape=(n_points_down_sampled_pc, n_points_original_pc))
    del row_ind, col_ind

    denominator = sparse_matrix.sum(axis=1).reshape(n_points_down_sampled_pc, 1)
    pc_down_sampled = np.asarray((sparse_matrix @ pc_input) / denominator)

    return pc_down_sampled, voxel_map


def get_down_sampled_points_and_classification(points, classification, voxel_size):
    down_sampled_points, _ = down_sample_point_data(voxel_size, points)
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
    _, indices = neighbours.kneighbors(down_sampled_points)
    down_sampled_classification = classification[indices.flatten()]

    return down_sampled_points, down_sampled_classification


def bounding_box_calculator(point_cloud):
    """Function to calculate the bounding box of a point cloud

    Parameters
    ----------
    point_cloud : numpy array
        nPoints x 3 array containing [x,y,z] information

    Returns
    -------
    type: array of 6 floats
    Values corresponding to the dimensions of the bounding box in the order: [min_x, max_x, min_y, max_y, min_z, max_z]

    """

    # Calculating the bounding box of the point cloud
    max_bounds = np.amax(point_cloud, axis=0)[0:3]
    min_bounds = np.amin(point_cloud, axis=0)[0:3]

    max_x = max_bounds[0]
    min_x = min_bounds[0]
    max_y = max_bounds[1]
    min_y = min_bounds[1]
    max_z = max_bounds[2]
    min_z = min_bounds[2]

    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])


def calc_train_bubble_centres(bounding_box_dimensions, bubble_slide_step):
    """Function to determine a list of train bubble centres

    Parameters
    ----------
    bounding_box_dimensions : numpy array
        An array of 6 floats corresponding to bounding box dimensions of a point cloud:
        [min_x, max_x, min_y, max_y, min_z, max_z]
    bubble_slide_step : float
        Resolution at which inference centres will be sampled.

    Returns
    -------
    An array of train bubble centres -> np.array([[c1], [c2], [c3]])

    """

    # splitting the x,y and z ranges of the bounding box based on bubble_slide_step
    create_range = lambda min_v, max_v, step: np.arange(min_v, max_v + step, step)
    x = create_range(bounding_box_dimensions[0], bounding_box_dimensions[1], bubble_slide_step)
    y = create_range(bounding_box_dimensions[2], bounding_box_dimensions[3], bubble_slide_step)
    z = create_range(bounding_box_dimensions[4], bounding_box_dimensions[5], bubble_slide_step)

    # creating a 3d grid of the bounding box
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # converting the 3d grid into a list of points
    train_bubble_centres = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).transpose()

    return np.asarray(train_bubble_centres)
