import torch
import open3d as o3d
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from vista_map.vista_utils import VistaCoverage
from voxel_traversal.traversal_utils import VoxelGrid

"""
This script visualizes the geometric information gain metric on an example of the Stanford bunny.
"""

def look_at(location, target, up):
    z = (location - target)
    z_ = z / torch.norm(z)
    x = torch.cross(up, z, dim=-1)
    x_ = x / torch.norm(x)
    y = torch.cross(z, x, dim=-1)
    y_ = y / torch.norm(y)

    R = torch.stack([x_, y_, z_], dim=-1)
    return R

# Torch stuff
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# Load mesh file
obj_path = './vista_map/data/stanford-bunny.obj'

mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=np.zeros(3))
mesh.translate(-mesh.get_center())
points = np.asarray(mesh.vertices)

# Define params
bounding_box = mesh.get_axis_aligned_bounding_box()
cmap = matplotlib.cm.get_cmap('turbo')

K = torch.tensor([
    [100., 0., 100.],
    [0., 100., 100.],
    [0., 0., 1.]
], device=device)
far_clip = 1.

param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
    'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'K': K,
    'far_clip': far_clip,
    'val_ndim': 4,
}

normalized_z = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
colors = cmap( normalized_z )[:, :3]
values = np.concatenate([colors, normalized_z.reshape(-1, 1)], axis=-1)

# Initialize voxel grid
vgrid = VoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device, dtype=torch.float16), torch.tensor(values, device=device, dtype=torch.float16))

# Define "training view" cameras
N_cameras = 100
t = torch.linspace(0., np.pi / 4, N_cameras, device=device)

depth_images = []
directions = []
origins = []

c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
c2w[:, :3, 3] = torch.stack([0.25*torch.cos(t), 0.25*torch.sin(t), torch.zeros_like(t)], dim=-1)

target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)
c2w[:, :3, :3] = look_at(c2w[:, :3, 3], target, up)

tnow = time.time()
torch.cuda.synchronize()
image, depth, output = vgrid.camera_voxel_intersection(K, c2w, far_clip)
torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)

# Segment out the rays that intersected the voxel grid
values = image.reshape(-1, image.shape[-1])
depths = depth.reshape(-1)
directions = output['rays_d'].reshape(-1, 3)
origins = output['rays_o'].reshape(-1, 3)

# For depths greater than 0
mask = depths == 0
depths[mask] = far_clip

# Termination points
directions = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)
xyzs = origins + depths[:, None] * directions

pointcloud = {
    'points': xyzs,
    'values': values,
    'origins': origins
}


vista = VistaCoverage(param_dict, pointcloud, 3, device)

# Define test time viewing cameras 
N_cameras = 100
t = torch.linspace(0., 2*np.pi, N_cameras, device=device)

c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
c2w[:, :3, 3] = torch.stack([0.25*torch.cos(t), 0.25*torch.sin(t), -0.1 + 0.2*torch.arange(len(t), device=device)/len(t)], dim=-1)

target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)
c2w[:, :3, :3] = look_at(c2w[:, :3, 3], target, up)

tnow = time.time()
torch.cuda.synchronize()
image, depth, _ = vista.camera_voxel_intersection(K, c2w, far_clip)
torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)

# Concatenate information gain image with depth image
combined_images_list = []
cmap = matplotlib.cm.get_cmap('turbo')
for i, (img, dep) in enumerate(zip(image, depth)):
    dep = dep / far_clip
    img = img[..., -1]
    combined_image = torch.concatenate([img, dep], axis=1)

    mask = (combined_image == 0).cpu().numpy()
    combined_image = cmap(combined_image.cpu().numpy())[..., :3]
    combined_image[mask] = 0
    
    combined_images_list.append((combined_image * 255).astype(np.uint8))

# Generate gif of information gain metric
fig_dir = Path("vista_map/images/")

num_images = len(combined_images_list)
fig_comb, axs_comb = plt.subplots()

im_comb = axs_comb.imshow(combined_images_list[0], animated=True)
def animate(i):
    im_comb.set_array(combined_images_list[i])
    return im_comb,

ani_comb = animation.FuncAnimation(fig_comb, animate, frames = num_images, interval=200, blit=True, repeat_delay=10)
ani_comb.save(f"{fig_dir}/vista_coverage.gif")
