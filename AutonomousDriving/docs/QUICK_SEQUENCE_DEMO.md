# Quick Squence Demo for 3DTrans

Simliar to `OpenPCDet` repo, `3DTrans` repo also provides a quick sequence demo to show the groundtruth or prediction results of on the custom point cloud data to visualize the continuous results. 

You need to follow the [INSTALL.md](INSTALL.md) to install the `3DTrans` repo successfully. 

1. Make sure you have already installed the [`Open3D`](https://github.com/isl-org/Open3D). 
If not, you could install it as follows:
```
   pip install open3d==0.15.2
```  
  
2. Run the demo with your custom point cloud data and groundturth infos as follows:
```shell
    python show_squence_demo/demo.py --data_path ${POINT_CLOUD_DATA} --seq_id ${SEQUENCE_ID} 
    --func once or nuscenes
```
Here `${SEQUENCE_ID}` is the sequence id of the data, such as **"000076"** of ONCE, or **"n015-2018-07-18-11-07-57+0800"**  of nuScenes.     
Here `${POINT_CLOUD_DATA}` can be in any of the following format:  
**ONCE**:      

```    
                                           
├──once_data                                                                       
│   │── XXX(squence_id)                                            
│   │   │── XXX(squence_id).json                                        
│   │   │── lidar_roof   
│   │   │   │── *.bin  
```
                              
**nuScenes**:  
```                                                       
├──nuscenes_data                                                           
│   ├── nuscenes_infos_10sweeps_train.pkl                                          
│   ├── samples                                    
│   │   ├── LIDAR_TOP                                     
│   │   │   │── *.pcd.bin
```

Then you could see the groundtruth results with visualized point cloud as follows:

<p align="center">
  <img src="sequence_demo.gif" width="99%">
</p>
