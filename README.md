# MapRefine
SLAM map postprocess

# Reference

1. Ground Segmentation https://github.com/lorenwel/linefit_ground_segmentation

2. Traditional Moving Object Segmentation https://github.com/irapkaist/removert
3. Learning-based Moving Object Segmentation https://github.com/PRBonn/LiDAR-MOS

## Note of Xavier

- Limited performance due to only front-view lidar and long range data. 
- Since our data-collection car only passed through one scene once, relative dynamic objects are hardly removed in each scan. 
- When selecting local sub-map, a precision detection range of lidar is required.
- Flyback mode is needed, while currently we have not enabled it, so different depth map are calculated odd or unoder frames.
