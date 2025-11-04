# LFE-MVSNet 

  

## Installation  
  
Clone repo:  
```  
git clone https://github.com/Skyproud/LFE-MVSNet.git
cd LFE-MVSNet
```  
  
The code needs Python >= 3.6, PyTorch >= 1.9.0 and CUDA >= 11.1. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version in the `requirements.txt` according to your computer.  
```  
conda create -n gbinet python=3.8  
conda activate gbinet  
pip install -r requirements.txt  
```  
  
## Datasets  
  
### DTU  
  
Download the [DTU dataset](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet) and extract the archive. You could use [gdown](https://github.com/wkentaro/gdown) to download it form Google Drive. You could refer to [MVSNet](https://github.com/YoYo000/MVSNet) for the detailed documents of the file formats.  
  
Download the original resolution [depth maps](https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-) provided by [YaoYao](https://github.com/YoYo000/MVSNet/issues/106). Extract it and rename the folder to `Depths_raw`.  
  
Download the original resolution `Rectified` [images](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) from the [DTU website](https://roboimagedata.compute.dtu.dk/?page_id=36). Extract it and rename the folder to `Rectified_raw`.  
  
Merge the three folders together and you should get a dataset folder like below:  
  
```  
dtu  
├── Cameras  
├── Depths  
├── Depths_raw  
├── Rectified  
└── Rectified_raw  
```  
  
### BlendedMVS  
  
Download the [low-res set](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) from [BlendedMVS](https://github.com/YoYo000/BlendedMVS). Extract the file and you should get a data folder like below:  
  
```  
BlendedMVS  
└── low_res  
```  
  
### Tanksandtemples  
  
Download the [Tanks and Temples testing set](https://drive.google.com/open?id=1YArOJaX9WVLJh4757uE8AEREYkgszrCo) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet). For the `intermediate` subset, remember to replace the cameras by those in `short_range_caemeras_for_mvsnet.zip` in the `intermediate` folder, see [here](https://github.com/YoYo000/MVSNet/issues/14). You should get a dataset folder like below:  
  
```  
tankandtemples  
├── advanced  
│ ├── Auditorium  
│ ├── Ballroom  
│ ├── Courtroom  
│ ├── Museum  
│ ├── Palace  
│ └── Temple  
└── intermediate  
├── Family  
├── Francis  
├── Horse  
├── Lighthouse  
├── M60  
├── Panther  
├── Playground  
└── Train  
```
### CDUT_L_DATASET 

You can download the[CDUT_L_DATASET](https://pan.baidu.com/s/1_9NL3gei411baIWRcdLy4g?pwd=8kmt)
 

