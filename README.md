# A Deep Learning Approach to Cat and Dog Breed Classification
The aim of this project is to show the usability of latest deep learning models on breed classification task. This project aims to create a pipeline that finetunes the models then test them on our scrapped, manually cleaned dataset.

## Installation
1. Clone this repository.
```
git clone https://github.com/oguzaliarslan/Group_7
```
2. Run the following code to install dependencies:
```
pip install -r requirements.txt
```
3. **Datasets** are too big to upload on Github and can be access from the drive link below,
```
https://drive.google.com/drive/u/0/folders/16mcVLxOPyd5upBfsj5jE56SEfsMUegjh
```
these are essential for the training part of these project. Unzip them and after extraction put them into same folder with the jupyter notebooks.

4. **Models** are also too big to upload on GitHub and can be accessed from the drive link below
```
https://drive.google.com/drive/u/0/folders/16mcVLxOPyd5upBfsj5jE56SEfsMUegjh
```
these are essential for testing part of this project. Unzip them and after unzipping put them into same folder level with jupyter notebooks. Or you can simply you can run the ipynb to train the models on your local. 

5. Run the respective jupyter noteboks for the specified tasks:
    - Cat vs Dog: **catvsdog-modelin.ipynb** for training, **catvsdog-testing.ipynb** for testing.
    - Breed classification: **breed_classification.ipynb** for both training and testing.


## Executing the Scraping Script
Scraping script can be runned on your local.
```
python scraper.py
```

