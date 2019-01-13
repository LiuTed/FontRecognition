# Font Recognition
**Pattern Recognition Project**  
Font data can be downloaded at: http://archive.ics.uci.edu/ml/datasets/Character+Font+Images

## How to prepare runtime environment
- Install python3 with pip (3.5 suggested)
- Run `pip install -r requirements.txt` to install needed libs
- You may need to install `tensorflow` or `tensorflow-gpu` (depend on your environment) if want to run `CNN.py`. 

## Jobs
- Proposal: Together, Done
- Preprocessing: Read from csv file and image reconstruction
    -Liu, Done
- Feature Extraction: Extract feature from image and (optional) encoding
    - HOG: Done
    - SIFT: patent encumbered, replaced by Daisy Descriptor, Done
- **Model implementation & training**
    - Bayesian Inference: Li
    - Multiclass SVM: Li
    - Decision Tree (Random Forest): Chen
    - K-Means: Chen
    - KNN: Liu
    - Mean-shift: Liu
    - DCNN: Liu
- Report: Together, Done

## Usage
- **Put the font data at `../fonts`**, or modify the path in each python file to the correct path containing data
- Different method may need different type of data, but you can call `python utils.py` to convert the csv files to binary files (default stored in `../data`) which suitable for most methods
- `CNN`, `KNN`, `mean-shift`, `NB`, `SVM` contains the code of methods
- `feature_PCA_DAISY`, `net_def`, `summary`, `utils` are helper files
- 
