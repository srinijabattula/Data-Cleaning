# Data Preprocessing and Feature Engineering Pipeline

**Overview**
This project implements a data preprocessing and feature engineering pipeline using Python and popular libraries like pandas and scikit-learn. It handles tasks such as missing value treatment, column renaming, scaling, encoding categorical variables, and dimensionality reduction with PCA.

**Features**
- Load datasets and assign custom column names
- Detect and handle missing values
- Impute missing numeric data using mean strategy
- Scale numeric features using standardization
- One-hot encode categorical variables
- Create new features based on domain-specific transformations
- Perform PCA to reduce dimensionality
- Save the cleaned and processed data for further analysis

**Project Structure**
preprocess.py 
utils.py 
README.md
requirements.txt
test_preprocess.py 

**Prerequisites**
- Python 3.7+
- pandas
- numpy
- scikit-learn

Install dependencies with:
pip install -r requirements.txt

**Running the pipeline:**

python preprocess.py --input data/raw/dataset.csv --columns data/column_names.txt --output data/processed/cleaned_data.csv

**Usage**

Place your raw dataset CSV in data/raw/

Prepare a column names text file (one column name per line)

Run the pipeline script with appropriate file paths

Processed, cleaned dataset will be saved to the specified output file

**Testing**

Do the unit testing with the following command:

python -m unittest discover test

**Author**

Srinija Battula
(Cybersecurity and Threat Intelligence)
