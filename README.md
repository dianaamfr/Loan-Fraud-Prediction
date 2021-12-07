# AC - To Loan or not to Loan

## Compilation

From the **src** folder of the repository:

Create MySQL database:

```properties
1. mysql -u root -p
```
```sql
    1. CREATE DATABASE bank_database;
    2. SET GLOBAL local_infile = true;
    3. quit;
```
```properties
2. mysql -u root -p --local-infile=1 bank_database < database/database.sql
```

### Ubuntu

Create the virtual environment:
```properties
1. python3 -m venv env
2. source env/bin/activate
3. pip3 install -r ../requirements.txt
```

## Windows

Create the virtual environment:
```properties
1. py -m venv env
2. .\env\Scripts\activate.bat
3. pip install -r ..\requirements.txt
```

***

## Run

1. **Clean**: Generate train and test csvs with clean data and save them to clean_data folder 
> `make clean <submission_name>` 
- outputs clean_data/<submission_name>.csv
- *e.g.* `make clean sub2` will generate the file sub2-train.csv and sub2-test.csv in the clean_data folder 

2. **Train**: Train the model with the clean data, using a specific classifier, compute the AUC and store the model in the models folder
> `make train <classifier> <submission_name>` 
- outputs models/&lt;classifier&gt;-&lt;submission_name&gt;.sav
- *e.g.* `make train logistic_regression sub2` will use as input the file sub2-train.csv from the clean_data folder and store in the models folder the model that results of applying the Logistic Regression Classifier to the data - `logistic_regression-sub2.sav`

3. **Test**: Test a model with the test data and store the result in the results folder
> `make test <classifier> <submission_name>` 
- outputs results/&lt;classifier &gt;-&lt;submission_name &gt;.csv
- *e.g.* `make test logistic_regression sub2` will apply the model models/logistic_regression-sub2.sav to the data from clean_data/sub2-test.csv and store in results/logistic_regression-sub2.csv