import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from scipy import stats
from IPython.display import Image
from langchain.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import OpenAI
from langchain.agents import create_sql_agent
import sqlite3
load_dotenv(dotenv_path='/workspace/EDA_bot/.env')

# sample_df = pd.read_csv('sample_df.csv')
def analyze_numerical_columns_with_visuals(df: pd.DataFrame, numerical_columns: list, output_dir: str = "./eda_visuals"):
    import os
    os.makedirs(output_dir, exist_ok=True)  # Create directory for visualizations if it doesn't exist

    summary = []

    for col in numerical_columns:
        data = df[col].dropna()  # Handle missing values
        stats = {
            'Column': col,
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Standard Deviation': np.std(data),
            'Variance': np.var(data),
            'Min': np.min(data),
            'Max': np.max(data),
            '25th Percentile': np.percentile(data, 25),
            '50th Percentile (Median)': np.percentile(data, 50),
            '75th Percentile': np.percentile(data, 75),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis(),
            'Count': len(data),
            'Missing Values': df[col].isnull().sum(),
        }

        # Identify outliers using IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        stats['Outliers Count'] = len(outliers)
        
        summary.append(stats)

        # Generate Visualizations
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(2, 2, 1)
        sns.histplot(data, kde=True, bins=30, color='blue')
        plt.title(f"Distribution of {col}")
        plt.show()
        
        # Boxplot
        plt.subplot(2, 2, 2)
        sns.boxplot(data, color='green')
        plt.title(f"Boxplot of {col}")
        plt.show()
        # KDE Plot
        plt.subplot(2, 2, 3)
        sns.kdeplot(data, shade=True, color='purple')
        plt.title(f"KDE Plot of {col}")
        plt.show()
        # Scatter Plot with Index
        plt.subplot(2, 2, 4)
        plt.scatter(range(len(data)), data, alpha=0.7, color='orange')
        plt.title(f"{col} Values Scatter")
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.show()
        # Save Visualizations
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_visuals.png")
        plt.close()

    summary_df = pd.DataFrame(summary)
    print("summary_df is :",summary_df)
    return summary_df

def perform_eda_on_categorical(df, categorical_columns, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    eda_summary = []

    for column in categorical_columns:
        value_counts = df[column].value_counts()
        proportions = df[column].value_counts(normalize=True)
        mode = df[column].mode()[0]

        eda_summary.append({
            'Column': column,
            'Value_counts': value_counts.to_dict(),
            'Proportions': proportions.to_dict(),
            'Mode': mode
        })

        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=df)
        plt.title(f"Countplot of {column}")
        
        sanitized_column_name = column.replace('/', '_').replace(' ', '_')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{sanitized_column_name}_eda_visualizations.png")
        plt.close()

    eda_report_df = pd.DataFrame(eda_summary)

    eda_report_df['Visualizations'] = eda_report_df['Column'].apply(
        lambda column: f"{output_folder}/{column.replace('/', '_').replace(' ', '_')}_eda_visualizations.png"
    )

    return eda_report_df

import pandas as pd

def combine_eda_reports(categorical_df, numerical_df):
    categorical_df_flat = categorical_df.copy()

    categorical_df_flat['Value_counts'] = categorical_df_flat['Value_counts'].apply(lambda x: str(x))
    categorical_df_flat['Proportions'] = categorical_df_flat['Proportions'].apply(lambda x: str(x))
    
    numerical_df_flat = numerical_df.copy()

    numerical_df_flat['Mean'] = numerical_df_flat['Mean']
    numerical_df_flat['Median'] = numerical_df_flat['Median']
    numerical_df_flat['Standard Deviation'] = numerical_df_flat['Standard Deviation']
    numerical_df_flat['Variance'] = numerical_df_flat['Variance']
    numerical_df_flat['Min'] = numerical_df_flat['Min']
    numerical_df_flat['Max'] = numerical_df_flat['Max']
    numerical_df_flat['25th Percentile'] = numerical_df_flat['25th Percentile']
    numerical_df_flat['50th Percentile (Median)'] = numerical_df_flat['50th Percentile (Median)']
    numerical_df_flat['75th Percentile'] = numerical_df_flat['75th Percentile']
    numerical_df_flat['Skewness'] = numerical_df_flat['Skewness']
    numerical_df_flat['Kurtosis'] = numerical_df_flat['Kurtosis']
    numerical_df_flat['Count'] = numerical_df_flat['Count']
    numerical_df_flat['Missing Values'] = numerical_df_flat['Missing Values']
    numerical_df_flat['Outliers Count'] = numerical_df_flat['Outliers Count']

    combined_df = pd.concat([categorical_df_flat, numerical_df_flat], ignore_index=True)

    return combined_df

def upload_dataframe_to_sql(df, db_path, table_name):
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Automatically generate a CREATE TABLE SQL command from the DataFrame schema
    create_table_query = pd.io.sql.get_schema(df, table_name)
    
    # Create the table if it doesn't exist
    cursor.execute(create_table_query)
    conn.commit()

    # Insert the data from the DataFrame into the table
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()
    print(f"Data successfully uploaded to the table '{table_name}' in {db_path}.")

def get_answer(prompt):
    # define the database we want to use for our test
    db = SQLDatabase.from_uri('sqlite:///sql_lite_database.db')

    # choose llm model, in this case the default OpenAI model
    llm = OpenAI(
                
                temperature=0,
                verbose=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                )
    # setup agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
  # Print the model used by accessing the model parameter directly
    print(f"Using model: {llm.model_name}")

    result =  agent_executor.invoke(prompt)
    return result