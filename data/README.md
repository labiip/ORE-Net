# Tooth_Registration_Dataset

The **Tooth_Registration_Dataset** is designed for CBCT-IOS tooth point cloud registration tasks.  
It contains training, validation, and test sets of paired point clouds with their corresponding ground-truth transformations.

## Dataset Structure
Tooth_Registration_Dataset/
├── train/ # Training set
│ ├── **_sample0.pkl
│ ├── **_sample1.pkl
│ └── ...
├── val/ # Validation set
│ ├── **_sample0.pkl
│ ├── **_sample1.pkl
│ └── ...
├── test/ # Test set
│ ├── **_sample0.pkl
│ ├── **_sample1.pkl
│ └── ...

Each `.pkl` file stores one point cloud pair and its ground-truth transformation.

## Data Format

Taking `300_up_sample1.pkl` as an example, the file contains a Python dictionary with the following keys:

- **`src_pcd`** (`src_points`)  
  Source point cloud, representing the input to be registered.  

- **`tgt_pcd`** (`ref_points`)  
  Target point cloud, representing the reference for registration.  

- **`gt_pose`** (`transform`)  
  Ground-truth rigid transformation matrix (a 4×4 homogeneous transformation) mapping the source point cloud to the target.  