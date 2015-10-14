# Individual Project

## Processing Pipeline

1. Depth Map Conversion (Raw Vertex & Normal Map)
    * converts raw depth from Kinect into __floating point depth in metres__
    * (optional ``AlignPointClouds``) converts to an __orientation point cloud__ which consists of 3D points/ vertices in the camera coordination system, and __surface normals__ (orientation of the surface) at these points

2. Camera Tracking
    * calculates the global/ world camera pose (location & orientation) using an __iterative alignment algorithm__ (to track this pose as the sensor moves in each frame)
    * ``NuiFusionAlignPointClouds``
        * align point clouds calculated from the reconstruction with new incoming point clouds from the Kinect camera depth, or standalone (e.g. align two separate cameras viewing the same scene)
    * `` AlignDepthToReconstruction``
        * more accurate camera tracking results when working with a reconstruction volume, but worse with moving objects
        * re-align camera with the last tracked pose if tracking breaks with this method

3. Fusing / Integrating Depth Data
    * fuse depth data from the known sensor pose into a single volumetric representation of the space around the camera
    * per-frame
    * continuously
        * with running avg to reduce noise, but still be able to __handle some dynmaic change in the scene__ _- small objects being removed_
    * when viewing from a different viewpoint, any gaps or holes where depth data is not present can be filled in 
    * surfaces are __continuously refined__ with __newer high resolution data__ as the camera approaches the surface more closely

4. Raycasting (3D Rendering)
    * reconstruction volume can be raycast from a sensor pose (typically the current Kinect sensor pose) - results in a new point cloud 
    * the resultant point cloud can be __shaded__ for a rendered visible image of the 3D reconstruction volume


## Tracking

* needs to be enough depth variation - __cluttered scenes are the best__
    * pointing to a single planar wall would not work
* 

## Reconstruction Volume

* scan up to 8 m<sup>3</sup>; `__OR__`
* voxel resolution of 1 to 2 mm per voxel
