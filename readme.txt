This is a readme file for the judge of this assignment.

Our contribution:
    Proposal: Together
    Preprocessing: Read from csv file and image reconstruction -Liu
    Feature Extraction: Extract feature from image and (optional) encoding
        HOG: Done Li
        SIFT: patent encumbered, replaced by Daisy Descriptor, Li
    Model implementation & training
        Bayesian Inference: Li
        Multiclass SVM: Li
        Decision Tree (Random Forest): Chen
        K-Means: Chen
        KNN: Liu
        Mean-shift: Liu
        DCNN: Liu
    Report: Together

Before running the code, you will have to:
    1. download the fonts dataset at http://archive.ics.uci.edu/ml/datasets/Character+Font+Images and unzip them
    2. Install the following modules:
    -tensorflow or tensorflow-gpu according to your environment
    -numpy
    -Pillow
    -matplotlib
    -scikit-image
    -sklearn
    3. change the path of the dataset in the code with your own.

About who are the main functions:
    self-written modules: summary.py, utils.py, net_def.py. 
    main functions: all of the rest.

The corrispondence between main functions and figures/tables:
    fig1-4: feature_PCA_DAISY.py
    fig5-8£ºNB.py
    fig9-11£ºSVM.py
    table I-II: KNN.py
    fig12(a): dt_md_on_accuracy.py
    fig12(b): dt_mid_on_accuracy.py
    fig13: rf_N_on_accuracy.py
    fig14: performance_of_dt&rf.py
    table III and fig 15: CNN.py