# Font Recognition
**Pattern Recognition Project**  
Font data can be downloaded at: http://archive.ics.uci.edu/ml/datasets/Character+Font+Images

## How to prepare runtime environment
- Install python3 with pip
- Run `pip install -r requirements.txt` to install needed libs

## How to use Git
- Clone git to your machine
    - Start a terminal (powershell on Windows)
    - Switch to the directory you'd like to clone to
    - Run command `git clone https://github.com/LiuTed/FontRecognition.git`

- Pull code from git
    - Run command `git pull`

- Push your change to git
    - Run command `git add <files you changed>`
    - Run command `git commit -m 'your description to the changes'`
    - Run command `git push`
    - If push failed and git gives the warning that you need to pull first, run `git pull`

- How to merge
    - Merge may happen when you pull from git
    - Find the files it lists
    - You would find part of the file looks like  
    <pre><code>
    >>> 1234567890abcdef  
    code1
    ----------------------------------
    code2
    <<< fedcba0987654321
    </code></pre>
    - There are some conflict between code1 and code2, you need to merge them manually.  
    Replace this part (the two hex number included) with the final code
    - Run `git commit` and save the description it automatically generated
    - Run `git push`

## Jobs
- Proposal
    - Done
- Preprocessing: Read from csv file and image reconstruction
    - Done
- Feature Extraction: Extract feature from image and (optional) encoding
    - SIFT: patent encumbered, replaced by Daisy Descriptor, Done
    - Fisher Vector: Liu
- **Model implementation & training**
    - Bayesian Inference: Li
    - Multiclass SVM: Li
    - Decision Tree (Random Forest):
    - Clustering:
    - ResNet-18 & LeNet: Liu