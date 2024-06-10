# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# -- Conexión inicial con Snowflake
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.connector.pandas_tools import write_pandas

connection_parameters = {
    "account": "lv85658.ca-central-1.aws",
    "user": "marcoespinoza",
    "password": "VZxzXQ9y7t",
    # "role": "<your snowflake role>",  # optional
    "warehouse": "snowpark_opt_wh",  # optional
    "database": "snowpark_db",  # optional
    "schema": "BMI_SCH",  # optional
    }  

new_session = Session.builder.configs(connection_parameters).create()


base = pd.read_csv("datos/ehresp_2014.csv")

snowflake_table_name = 'base'
new_session.write_pandas(df= base,table_name=snowflake_table_name, auto_create_table= True)

new_session.close()

# Inicialmente, leemos la base de datos. Los base de datos llamada Eating & Health Module Dataset
# El objetivo es predecir el índice de masa corporal utilizando un modelo Boosting
# Con el objetivo de: 
# 1. Demostrar la capacidad de los modelos de Machine Learning en ámbitos de no normalidad
# 2. La facilidad de aprovechar bases de datos cloud para la ejecución de archivos de análisis de datos

# -- USER DEFINED FUNCTION

udf_sql = """
CREATE OR REPLACE PROCEDURE SNOWPARK_DB.BMI_SCH.train()
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = 3.8
PACKAGES = ('snowflake-snowpark-python', 'scikit-learn', 'joblib')
HANDLER = 'main'
AS $$
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

def main(session):
    base = session.table('SNOWPARK_DB.BMI_SCH."base"').to_pandas()
    base = base[["erbmi","ertpreat","ertseat","euexfreq",
    "eufastfdfrq", "euhgt","euwgt"]]
    base = base[base["erbmi"] > 0]
    X, y = base.drop(columns=['erbmi']), base["erbmi"]
    # Split dataset into training and test
    X_train,X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state= 25)

    param_grid = [
    {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
     'subsample': [0.5, 1],
     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
     'ccp_alpha': [0, 0.05, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]}]


    # Create pipeline and train
    fit = GradientBoostingRegressor()

    clf = GridSearchCV(
        estimator= fit,
        param_grid= param_grid,
        scoring= 'neg_mean_squared_error',
        cv = None
    )

    clf.fit(X_train, y_train)

    reg = GradientBoostingRegressor(
    learning_rate= 0.1,
    subsample= 0.5,
    max_depth= 7,
    ccp_alpha= 0)

    modelo = reg.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    # Upload trained model to a stage
    # model_file = os.path.join('/tmp', 'model.joblib')
    # dump(modelo, model_file)
    # session.file.put(model_file, "@ml_models",overwrite=True)

  # Return model R2 score on train and test data
    return {"R2 score on Train": modelo.score(X_train, y_train),"R2 score on Test": modelo.score(X_test, y_test)}
$$;

"""

new_session.sql(udf_sql).collect()


new_session.call("train", 1)



base = base[["erbmi","ertpreat","ertseat","euexfreq",
             "eufastfdfrq", "euhgt","euwgt"]]
# Inspeccionamos la base de datos
base.columns
base.shape
print(base.dtypes)
pd.set_option('display.max_columns', None)
base.describe()
# La variable de interés es erbmi
base["erbmi"].describe() # Existen valores > 0 lo cual no es ideal porque es imposible tener una masa corporal de 0 o menos
base = base[base["erbmi"] > 0]
base.shape



# -- Base de entrenamiento y prueba
X, y = base.drop(columns=['erbmi']), base["erbmi"]

X_train,X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state= 25)

# dtrain = xgb.DMatrix(data=X_train, label=y_train)

# -- VALIDACIÓN CRUZADA
param_grid = [
    {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
     'subsample': [0.5, 1],
     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
     'ccp_alpha': [0, 0.05, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]}
]

fit = GradientBoostingRegressor()

clf = GridSearchCV(
    estimator= fit,
    param_grid= param_grid,
    scoring= 'neg_mean_squared_error',
    cv = None
)

clf.fit(X_train, y_train)

with open('model.pkl','wb') as f:
    pickle.dump(clf, f)

with open('model.pkl','rb') as f:
    clf2 = pickle.load(f)

clf.best_params_

# De acuerdo a la validación cruzada, el mejor es estimador es un alpha de 0, un max_depth de 7 y un subsample de 0.5

# -- AJUSTAMOS EL MODELO

reg = GradientBoostingRegressor(
    learning_rate= 0.1,
    subsample= 0.5,
    max_depth= 7,
    ccp_alpha= 0
)

modelo = reg.fit(X_train, y_train)

pred = modelo.predict(X_test)
print(pred)
# Coeficiente de determinación
modelo.score(X_test, y_test)

# Histograma de valores 


sns.histplot(y_test, color='blue', label='Valor real', kde=False, bins=10, alpha=0.5)

# Plotting the second histogram
sns.histplot(pred, color='red', label='Valor predicho', kde=False, bins=10, alpha=0.5)

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Ingreso and Pred')
plt.legend(title='Legend')
plt.show()