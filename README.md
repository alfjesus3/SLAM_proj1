# SLAM_proj1
This project is based on the twitchslam project by George Hotz (https://github.com/geohot/twitchslam).
It is a sparse features based SLAM implementation which was tested in Ubuntu 18.04 LTS.
The raw video test was https://vimeo.com/324251099 

<img width=450px src="https://github.com/alfjesus3/SLAM_proj1/blob/master/slam.png" />

Usage
-----
```
F=500		# Focal length (in pixels)
Vi!=None	# Visualize the raw input video with SDL2	

# Good Example
F=500 Vi=1 ./slam.py 

```

Libraries Used
-----

* SDL2 for 2-D display
* cv2 for feature extraction
* pangolin for 3-D display
* g2opy for optimization


