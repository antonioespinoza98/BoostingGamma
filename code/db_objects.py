from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

connection_parameters = {
    "account": "lv85658.ca-central-1.aws",
    "user": "marcoespinoza",
    "password": "VZxzXQ9y7t",
    # "role": "<your snowflake role>",  # optional
    # "warehouse": "<your snowflake warehouse>",  # optional
    # "database": "<your snowflake database>",  # optional
    # "schema": "<your snowflake schema>",  # optional
    }  

new_session = Session.builder.configs(connection_parameters).create()

# Vamos a crear objetos para utilizar para ejecutar nuestro workflow

# -- WAREHOUSE

create_warehouse_sql = """
CREATE OR REPLACE WAREHOUSE snowpark_opt_wh 
    WITH WAREHOUSE_SIZE = 'MEDIUM' 
    WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED';
"""

new_session.sql(create_warehouse_sql).collect()

create_database = """
CREATE OR REPLACE DATABASE snowpark_db
    COMMENT = 'Base de datos para almacenar datos locales'
"""
new_session.sql(create_database).collect()

create_schema = """
CREATE OR REPLACE SCHEMA snowpark_db.BMI_SCH
"""
new_session.sql(create_schema).collect()

new_session.close()



