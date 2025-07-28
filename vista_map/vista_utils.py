import torch
import numpy as np
from voxel_traversal.traversal_utils import VoxelGrid
from vista_map.coverage.utils import VoxelCoverage

class VistaCoverage(VoxelGrid):
    def __init__(self, param_dict, pointcloud, ndim, device):
        super().__init__(param_dict, ndim, device)

        self.far_clip = param_dict['far_clip']

        self.base_directions = None
        self.base_indices = None

        self.val_ndim = param_dict['val_ndim']
        self.coverage = VoxelCoverage(device, param_dict['discretizations'].prod().item(), 3)

        self.pointcloud = pointcloud        # point cloud is a dictionary (rather than a tensor so we don't have to hard code anything in here.)

        points = self.pointcloud['points']
        values = self.pointcloud['values']

        if 'origins' not in self.pointcloud.keys():
            cameras = self.pointcloud['cameras']
            self.populate_voxel_grid_from_pointcloud_cameras(points, values, cameras)
        else:
            origins = self.pointcloud['origins']
            directions = points - origins

            # Near clip
            if 'near_clip' in self.pointcloud.keys():
                self.near_clip = self.pointcloud['near_clip']
                # Normalize directions
                directions_normalized = directions / torch.norm(directions, dim=-1, keepdim=True)
                origins = origins + directions_normalized * self.near_clip
            self.populate_voxel_grid_from_pointcloud_rays(points, values, origins, directions)

    def populate_voxel_grid_from_pointcloud_cameras(self, points, values, cameras):
        assert values.shape[-1] == self.val_ndim
        self.populate_voxel_grid_from_points(points, values)

        # Uses the viewing directions to calculate a covariance for each voxel
        output = self.calculate_view_diversity_cameras(cameras)
        free_and_occupied = output['traversed_voxels']

        # Calculate the totality of the occupied and unobserved
        self.voxel_grid_binary = torch.logical_or(self.voxel_grid_binary, ~free_and_occupied)

    def populate_voxel_grid_from_pointcloud_rays(self, points, values, origins, directions):
        assert values.shape[-1] == self.val_ndim
        self.populate_voxel_grid_from_points(points, values)

        # Uses the viewing directions to calculate a covariance for each voxel
        output = self.calculate_view_diversity_rays(origins, points, directions)
        free_and_occupied = output['traversed_voxels']

        # Calculate the totality of the occupied and unobserved
        self.voxel_grid_binary = torch.logical_or(self.voxel_grid_binary, ~free_and_occupied)

    def calculate_view_diversity_cameras(self, cameras):

        # Compute the voxel intersections from the training views
        K = cameras["K"]    # tensor
        c2w = cameras["c2w"]        # 3D tensor of shape (n_cameras, 4, 4)
        far_clip = cameras["far_clip"]  # float
        near_clip = cameras["near_clip"]  # float
        _, _, output = self.camera_voxel_intersection(K, c2w, far_clip, near_clip)
        
       # Make sure that the terminated voxel intersections are in bounds
        voxel_index = output['terminated_voxel_index']
        ray_directions_total = output['rays_d']
        directions = ray_directions_total[output['terminated_ray_index']]

        assert voxel_index.shape[0] == directions.shape[0]

        # Normalize the directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        self.base_directions = directions
        self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

        self.coverage.load_data(self.base_directions, self.base_indices)

        return output

    def calculate_view_diversity_rays(self, origins, ends, directions):
        # Compute the voxel intersections from the training views
        output = self.compute_voxel_ray_intersection(origins, directions, torch.zeros(origins.shape[0], dtype=torch.int32, device=self.device), 1)
        
        #NOTE: Is it OK for rays to run into the far clip in the grid? Probably
        # safer to make the far clip very large?
        # Make sure that the voxel intersections are in bounds
        voxel_index, in_bounds = self.compute_voxel_index(ends)
        voxel_index = voxel_index[in_bounds]

        directions = directions[in_bounds]

        # Normalize the directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        self.base_directions = directions
        self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

        self.coverage.load_data(self.base_directions, self.base_indices)

        return output

    # @torch.compile
    def compute_termination_values(self, indices, directions, extras=None):
        terminated_values = torch.zeros((len(directions), self.val_ndim + 1), device=self.device)

        voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]

        terminated_values[:, :voxel_values.shape[-1]] = voxel_values

        if self.base_directions is not None:
            print('Calculating View Diversity Values')
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)

            ### THIS IS FOR SINGLE CAMERA ###
            # indices = torch.tensor( np.ravel_multi_index( indices.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

            # query = torch.zeros((self.discretizations.prod().item(), 3), device=self.device)
            # query[indices] = directions

            # coverage_metric = self.coverage.compute_existing_coverage(query)        # n_voxels x n_dim

            ### THIS IS FOR BATCHED CAMERAS ###
            n_cameras = extras['n_cameras']
            indices = torch.tensor( np.ravel_multi_index( indices.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices
            indices = torch.stack([extras['camera_ids'], indices], dim=-1) #cat with camera indices
             
            query = torch.zeros((n_cameras, self.discretizations.prod().item(), 3), device=self.device, dtype=torch.float16) # n_cameras x n_voxels x ndim
            query[indices[:, 0], indices[:, 1]] = directions

            coverage_metric = self.coverage.compute_existing_coverage_batched(query)       # n_cameras x n_voxels
            terminated_values[:, -1] = (coverage_metric[indices[:, 0], indices[:, 1]] + 1.) / 2. # normalizes the coverage metric to be between 0 and 1. 

        return terminated_values
