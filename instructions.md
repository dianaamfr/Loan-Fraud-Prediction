# Instructions

## Ubuntu

From the src folder of the repository:
Create mysql database:
1. mysql -u root -p 
2. > CREATE DATABASE bank_database;
3. > SET GLOBAL local_infile = true;
4. > quit;
5. mysql -u root -p --local-infile=1 bank_database < database/database.sql

Create the virtual environment:
1. python3 -m venv env
2. source env/bin/activate
3. pip3 install -r ../requirements.txt
4. TODO

Utils:
- pip install pipreqs

## Windows

