import numpy as np
from scipy.sparse import csr_matrix
import laspy
from sklearn.neighbors import NearestNeighbors


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


# points, classification = get_data_from_laz_file("train-data/pc_46_0_out.laz")
# down_sampled_points, down_sampled_classification = get_down_sampled_points_and_classification(points, classification,
#                                                                                               0.08)
# save_as_laz_file(down_sampled_points, "test.laz", down_sampled_classification)