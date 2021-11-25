# AC - To Loan or not to Loan

## Compilation

### Ubuntu

From the **src** folder of the repository:
Create mysql **database**:
1. mysql -u root -p 
    1. CREATE DATABASE bank_database;
    2. SET GLOBAL local_infile = true;
    4. quit;
    5. mysql -u root -p --local-infile=1 bank_database < database/database.sql

Create the virtual environment:
1. python3 -m venv env
2. source env/bin/activate
3. pip3 install -r ../requirements.txt

## Windows
1. TODO

***

## Run

1. **Clean**: Generate train and test csvs with clean data and save them to clean_data folder 
> `make clean PARAMS=<submission_name>` 
- outputs clean_data/<submission_name>.csv
- *e.g.* `make clean PARAMS=sub2` will generate the file sub2-train.csv and sub2-test.csv in the clean_data folder 

2. **Train**: Train the model with the clean data, using a specific classifier, compute the AUC and store the model in the models folder
> `make train PARAMS='<classifier> <submission_name>'` 
- outputs models/<classifier>-<submission_name>.sav
- *e.g.* `make train PARAMS='logistic_regression sub2'` will use as input the file sub2-train.csv from the clean_data folder and store in the models folder the model that results of applying the Logistic Regression Classifier to the data - `logistic_regression-sub2.sav`

3. **Test**: Test a model with the test data and store the result in the results folder
> `make test PARAMS='<classifier> <submission_name>'` 
- outputs results/<classifier>-<submission_name>.csv
- *e.g.* `make train PARAMS='logistic_regression sub2'` will apply the model models/logistic_regression-sub2.sav to the data from clean_data/sub2-test.csv and store in results/logistic_regression-sub2.csv