# Individual Project

## Processing Pipeline

1. Depth Map Conversion (Raw Vertex & Normal Map)
    * converts raw depth from Kinect into __floating point depth in metres__
    * (optional ``AlignPointClouds``) converts to an __orientation point cloud__ which consists of 3D points/ vertices in the camera coordination system, and __surface normals__ (orientation of the surface) at these points

2. Camera Tracking
    * calculates the global/ world camera pose (location & orientation) using an __iterative alignment algorithm__ (to track this pose as the sensor moves in each frame)
    * see _[Tracking]_(https://github.bath.ac.uk/awll20/individual_project#tracking)


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
* ``NuiFusionAlignPointClouds``
    * can be used with a reconstruction volume or standalone
    * align point clouds calculated from the reconstruction with new incoming point clouds from the Kinect camera depth, or standalone (e.g. align two separate cameras viewing the same scene)
* `` AlignDepthToReconstruction``
    * requires a reconstruction volume
    * __more accurate__ camera tracking results when working with a reconstruction volume, but worse with moving objects
    * re-align camera with the last tracked pose if tracking breaks with this method
    * outputs an image of type ``NUI_FUSION_IMAGE_TYPE_FLOAT``
        * describes how well each depth pixel aligns with the reconstruction model
        * may be processed further (color rendering, as input to other vision algorithms, e.g. object segmentation)
        * values are normalised -1 to 1
            * represents the alignment cost/ energy for each pixel
            * larger magnitude values (+/-) represent more discrepancy/ more information at that pixel
            * 0 - valid depth exists, but no reconstruction model exists behind the depth pixels (indicating __perfect alignment__)
            * 1 - no valid depth exists
 * constraints
     * small, slow movement in both translation and rotation - best for maintaining stable tracking
     * __dropped frames__ can adversely affect tracking
         * can effectively lead to __twice__ the translational and rotational movement between processed frames


## Reconstruction Volume
* made up of small cubes in space (__voxels__)
    * scan up to 8 m<sup>3</sup>; *__OR__*
    * voxel resolution of 1 to 2 mm per voxel
* call ``NuiFusionCreateReconstruction`` to create a volume
    * pass a ``NUI_FUSION_RECONSTRUCTION_PARAMETERS`` structure to specify size
    * max size depends on the amount of RAM of the system
* ``voxelPerMeter`` scales the size that 1 voxel represents in the real world
    * e.g. to use a cubic of 384\*384\*384 to represent a 3m cube in real life,  ``voxelPerMeter`` should be set to 128vpm (384/128 = 3), or 256vpm (384/128 = 1.5) to represent a 1.5m cube in real life
* with a fied number of voxels, a volume which represents a __very large__ real world volume with a __very high resolution__ cannot be created
* reconstruction resolution is limited
    * with a maximum contiguous memory block of about 1GB, the max reconstrution resolution is about 640<sup>3</sup> (262 144 000 voxels)
    * can use multiple volumes or devices to deal with very large real world volume of high resolution


## Sample Projects 

* C++
    * Kinect Fusion Basics - D2D
    * Kinect Fusion Color Basics - D2D
    * Kincect Fusion Explorer - D2D
* C#
    * Kinect Fusion Basics - WPF
    * Kinect Fusion Color Basics - WPF
    * Kincect Fusion Explorer - WPF
    * Kinect Fusion Explorer Mullti-Static Cameras - WPF
    * Kinect Fusion Head Scanning - WPF
* Kinect Fusion Projects are basic examples
* Kinect Fusion Explorer Projects are more API-centric

