import pandas as pd
import numpy as np
from datetime import datetime
import os

def infer_and_convert_data_types():
    """
    Infer and convert data types for each column in the DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data to be processed.

    Returns:
    - DataFrame: Processed DataFrame with inferred and converted data types.

    This function iterates over each column in the DataFrame and attempts to infer the most appropriate
    data type for each column. It handles various data types, including numeric, datetime, and categorical,
    as well as handling null values such as 'Not available' or empty cells. After inferring the data types,
    it applies the inferred types to the DataFrame using the astype() method.

    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_directory, '../../sample_data.csv')

    df = pd.read_csv(csv_file_path, nrows=1000)
    print("Data types before inference:")
    print(df.dtypes)
    
    
    for col in df.columns:
        # Handle null elements
        df[col] = df[col].replace({'Not available': np.nan, 'null': np.nan})  # Replace 'Not available' and 'null' with NaN
        
        # Attempt to convert to numeric first
        df_converted = pd.to_numeric(df[col], errors='coerce')
        if not df_converted.isna().all():  # If at least one value is numeric
            # Check if the column has mixed numerical data types
            if df_converted.dtype.kind == 'f' and df_converted.apply(lambda x: x.is_integer()).all():
                df[col] = df_converted.astype(int)  # Convert float to integer
            else:
                df[col] = df_converted
            continue

        # Attempt to convert to datetime
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            df[col] = df[col].astype('datetime64[ns]')  # Specify datetime64 data type
            continue
        except (ValueError, TypeError):
            pass

        # Handle truncated years
        try:
            df[col] = df[col].apply(lambda x: datetime.strptime(str(x), '%y'))
            df[col] = df[col].astype('datetime64[ns]')
            continue
        except ValueError:
            pass

        # Attempt to parse ambiguous dates with DD-MM-YYYY format
        try:
            df[col] = df[col].apply(lambda x: datetime.strptime(str(x), '%d-%m-%Y'))
            df[col] = df[col].astype('datetime64[ns]')
            continue
        except ValueError:
            pass

        # Check if the column should be categorical
        if len(df[col].dropna().unique()) / len(df[col]) < 0.5:  # Example threshold for categorization
            df[col] = df[col].astype('category')


    print("\nData types after inference:")
    print(df.dtypes)
    csv_data = df.to_json(orient='records')
    return csv_data