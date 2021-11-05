import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector

class Database:
    def __init__(self):
        try:
            self.db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root"
            )

            if self.db.is_connected():
                db_version = self.db.get_server_info()
                print('Connected to MySQL. Version: ', db_version)

        except Exception as e:
            print("Error connecting to MySQL database: ", e)

    def execute(self, query):
        cursor = self.db.cursor(buffered=True)
        cursor.execute(query)
        try:
            records = cursor.fetchall()
            return records
        except:
            return "Error executing query: " + query
    
    