# Crowd Trajectory Clustering

## Introduction
This project provide algorithm for trajectory extraction and trajectory clustering for high density motion video.
## How to install and run the project

```sh
pip install -r requirements.txt
python3 main.py
```

| Parameter | Value |
| ------ | ------ |
| -p / --video_folder_path | Video Folder Path |
| -n / --video_name | Video Name|
| -f / --frame_limit | Frame Limit for trajectory extraction |
| -m / --method | DBScan for clustering / KMeans for Clustering |
| -pf / --plot_filtered | Plot Filtered |
| -pc / --plot_clustered | Plot Clustered |
| -at / --area_threshold | Area Threshold |
| -ft / --flow_threshold | Flow Threshold |
| -pt / --position_threshold | Position Threshold |

## How to use the project
Press ESC after a plot window pop out to continue the process
