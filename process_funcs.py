import pandas as pd
import os
import glob
import numpy as np
import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import io

def process_excel_files(file_names, pattern):
    all_df = []

    for file_name in file_names:
        course_id = extract_pattern(str(file_name), pattern) 
        df = pd.read_excel(file_name, sheet_name='UBYS', header=1)
        
        # Add the "Course" column to the DataFrame
        df.insert(0, "Ders", course_id)
        all_df.append(df)

    return all_df

def preprocess_data(df):

    # Remove the last two columns
    df = df.iloc[:, :-2]

    # Drop rows where the second column is NaN
    df = df.dropna(subset=[df.columns[2]])

    # Select only the columns
    cols = ['Ders','Numara', 'Ad', 'Soyad', 'PÇ1', 'PÇ2', 'PÇ3', 'PÇ4', 'PÇ5', 'PÇ6', 'PÇ7', 'PÇ8','PÇ9', 'PÇ10', 'PÇ11']
    new_df = df[cols]

    new_df = new_df.assign(**new_df[['Numara']].astype(np.int64))
    
    # Return the preprocessed data
    return new_df

def extract_rows_by_numbers(data_frames, target_numbers):
    result_dfs = []

    # Iterate through each target number
    for target_number in target_numbers:
        extracted_rows = []

        # Iterate through each data frame
        for df in data_frames:
            # Filter rows based on the "Number" column
            filtered_rows = df[df['Numara'] == target_number]

            # Append the filtered rows to the list
            extracted_rows.append(filtered_rows)

        # Concatenate the filtered rows into a new data frame for the current target number
        result_df = pd.concat(extracted_rows, ignore_index=True)

        # Append the result data frame to the list
        result_dfs.append(result_df)

    return result_dfs

def create_pattern(s):
    """Given a string that consists of letters followed by digits, creates a pattern to use in regex.

    Args:
    given_string: A string that consists of only letters followed by digits e.g. "XYZ123".

    Returns:
    A string that represents a compiled regular expression pattern.
    """

    # Check if the given string is a valid string that consists of letters followed by digits.
    letters = sum(c.isalpha() for c in s)
    numbers = sum(c.isdigit() for c in s)

    # Create a regular expression pattern that matches the given string.
    pattern = '([A-Za-z]{' + str(letters) + '}\d{' + str(numbers) + '})'

    # Return the compiled regular expression pattern.
    return pattern

def extract_pattern(input_string, pattern):
#     pattern = r'([A-Za-z]{3}\d{3})'
    matches = re.findall(pattern, input_string)
    result = ' '.join(matches)
    return result

def delete_empty_dfs(dataframes, target_list):
    deleted_ids = []
    filtered_df_list = []

    for i, df in enumerate(dataframes):
        if df.empty:
            deleted_ids.append(target_list[i])
        else:
            filtered_df_list.append(df)
    return filtered_df_list, deleted_ids
    
def process_student_df(df1, df2):
    """
    Process and merge two DataFrames based on the 'Dersler' column, fill NaN values, and perform data conversions.

    Args:
        df1 (pd.DataFrame): The first DataFrame containing the 'Dersler' column.
        df2 (pd.DataFrame): The second DataFrame to be merged with df1. Ex: result_dfs[0]

    Returns:
        pd.DataFrame: The processed DataFrame with NaN values filled, data type conversions, and 'Ders' column dropped.
    """
    # Merge df1 and df2 using a left join based on the 'Dersler' column in df1 and the 'Ders' column in df2
    df_merged = df1[['Dersler']].merge(df2, how='left', left_on='Dersler', right_on='Ders')

    # Drop the 'Ders' column from df_merged, as it's no longer needed
    df_merged.drop('Ders', axis=1, inplace=True)

    # Define a list of columns to fill NaN values with the most common value
    columns_to_fill = ["Numara", "Ad", "Soyad"]

    # Iterate through the specified columns and replace NaN values with the most common value
    for column in columns_to_fill:
        most_common_value = df_merged[column].mode().iloc[0]  # Get the most common value
        df_merged[column].fillna(most_common_value, inplace=True)

    # Fill any remaining NaN values in df_merged with 0
    df_merged = df_merged.fillna(0)

    # Convert the 'Numara' column to integer data type
    df_merged['Numara'] = df_merged['Numara'].astype(np.int64)

    return df_merged

def calculate_weighted_means(df_values, df_weights):
    """
    Calculate the weighted means of columns in a DataFrame based on values and weights.

    Args:
        df_values (pd.DataFrame): DataFrame containing values for which weighted means are calculated.
        df_weights (pd.DataFrame): DataFrame containing weights for the values.

    Returns:
        pd.DataFrame: DataFrame containing the calculated weighted means.
    """
    # Calculate the weighted means for each column
    weighted_means = (df_values * df_weights).sum(axis=0) / df_weights.sum(axis=0)

    # Convert the Series of weighted means to a DataFrame
    weighted_means_df = weighted_means.to_frame().T

    # Reset the index if needed
    weighted_means_df.reset_index(drop=True, inplace=True)

    return weighted_means_df

def process_result_dfs_v5(df_students, df_pc_dersler):
    
    # Get the weights 
    df_weights = df_pc_dersler.iloc[:, 1:]
    
    # Create an empty list
    df_list = []
    
    # Iterate over the student dataframes
    for df_student in df_students:

        if df_student.empty:
          print('There is an empty DataFrame')
          continue

        # Create a temporary list to store student info
        temp_list1 = [df_student['Numara'][0],df_student['Ad'][0],df_student['Soyad'][0]]
        
    
        # Values from df_students
        df_values = df_student.iloc[:, 4:]
        
        # Calculate the weighted means for each student
        weighted_means = (df_values * df_weights).sum(axis=0) / df_weights.sum(axis=0)
        
        # Store the weighted means in a list
        temp_list2 = list(weighted_means.values)
        
        # Combine lists for each student
        temp_list = temp_list1 + temp_list2
        
        # List of lists (each sublist has one students info)
        df_list.append(temp_list)

    # Concatenate the DataFrames
    df = pd.DataFrame(df_list)
    
    df.columns = ['Numara', 'Ad', 'Soyad', 'PÇ1', 'PÇ2', 'PÇ3', 'PÇ4', 'PÇ5', 'PÇ6', 'PÇ7', 'PÇ8', 'PÇ9', 'PÇ10', 'PÇ11']
    
    # Replace NaN values with zero
    df = df.fillna(0)

    return df

def hide_names(df, columns_to_hide=None):
    if columns_to_hide is None:
        columns_to_hide = df.columns

    def hide_name(name):
        return name.lstrip()[0] + '*' * (5) if len(name) > 1 else name
    
    df_copy = df.copy()
    for column in columns_to_hide:
        if column in df_copy.columns:
            if df_copy[column].dtype == 'object':
                df_copy[column] = df_copy[column].apply(hide_name)
            elif df_copy[column].dtype == 'int64':
                df_copy[column] = df_copy[column].astype(str).apply(hide_name)
    return df_copy

def append_average_row(df):
    # Calculate the average for the specified columns
    average_row = {
        'Numara': 'ORTALAMA',
        'Ad': 'ORTALAMA',
        'Soyad': 'ORTALAMA'
    }

    # Filter out the zero values from the DataFrame before calculating the mean
    non_zero_df = df.replace(0, np.nan)

    for col in ['PÇ1', 'PÇ2', 'PÇ3', 'PÇ4', 'PÇ5', 'PÇ6', 'PÇ7', 'PÇ8', 'PÇ9', 'PÇ10', 'PÇ11']:
        average_row[col] = non_zero_df[col].mean()

    # Create a DataFrame from the average_row dictionary
    df_average = pd.DataFrame([average_row])
    
    # Replace NaN values with zero
    df = df.fillna(0)
    
    # Concatenate the DataFrames
    df = pd.concat([df, df_average], ignore_index=True)
    
    return df
