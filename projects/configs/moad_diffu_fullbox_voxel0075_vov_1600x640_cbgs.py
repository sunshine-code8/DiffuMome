_base_ = ['./moad_diffu_voxel0075_vov_1600x640_cbgs.py']

# Full-box diffusion variant:
# - query/query_pos stay MOAD-style;
# - noisy reference state is expanded from center-only [x, y, z] to a bounded
#   10D box state [x, y, w, l, z, h, sin, cos, vx, vy];
# - at test time the center comes from feature proposals, while the other box
#   dimensions are initialized from Gaussian noise before DDIM denoising.
model = dict(
    pts_bbox_head=dict(
        reference_dim=10,
        test_gaussian_box_dims=True,
    )
)

# Conservative smoke/full-box setting. Increase samples_per_gpu back to 2 if
# memory is comfortable after the first epoch.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(lr=1e-5)
