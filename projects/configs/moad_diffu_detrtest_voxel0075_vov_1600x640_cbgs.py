_base_ = ['./moad_diffu_voxel0075_vov_1600x640_cbgs.py']

# DiffuDETR-style test-time denoising switches, kept as a separate config so
# the main MOAD-preserving experiment remains the minimal-change baseline.
model = dict(
    pts_bbox_head=dict(
        box_renewal=True,
        use_ensemble=True,
    )
)
