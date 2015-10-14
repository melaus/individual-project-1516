# Individual Project

## Processing Pipeline

1. Depth Map Conversion (Raw Vertex & Normal Map)
    * converts raw depth from Kinect into _floating point depth in metres_
    * (optional ``AlignPointClouds``) converts to an _orientation point cloud_ which consists of 3D points/ vertices in the camera coordination system, and _surface normals_ (orientation of the surface) at these points
2. Camera Tracking
    * calculates the global/ world camera pose (location & orientation) using an _iterative alignment algorithm_ (to track this pose as the sensor moves in each frame)
    * 
