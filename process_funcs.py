import pandas as pd
import os
import glob
import numpy as np
import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import io

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

def extract_rows_by_id(data_frames, target_numbers):
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

def process_excel_files(file_names):
    all_df = []

    for file_name in file_names:
        course_id = extract_pattern(str(file_name)) 
        df = pd.read_excel(file_name, sheet_name='UBYS', header=1)
        
        # Add the "Course" column to the DataFrame
        df.insert(0, "Ders", course_id)
        all_df.append(df)

    return all_df

def extract_pattern(input_string):
    pattern = r'([A-Za-z]{3}\d{3})'
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

def plot_bar_with_colors(data):
    # Generate a range of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

    # Create a bar plot
    plt.bar(range(len(data)), data, color=colors)

    # Show the plot
    plt.show()
    
def find_missing_courses(ref_df, result_dfs):
    missing_courses_count = {}
    unique_courses = ref_df['Dersler'].unique()

    for df in result_dfs:
        current_courses = df['Ders'].unique()
        missing_courses = [course for course in unique_courses if course not in current_courses]

        for course in missing_courses:
            if course in missing_courses_count:
                missing_courses_count[course] += 1
            else:
                missing_courses_count[course] = 1

    missing_courses_df = pd.DataFrame(list(missing_courses_count.items()), columns=['Dersler', 'MissingCount'])
    
    return missing_courses_df

def calculate_means_v3(df, df_pc_dersler):
    # Initialize a dictionary to store the mean values
    mean_values_dict = {}

    # Iterate over each performance outcome (e.g., 'PÇ1', 'PÇ2', ...)
    for column in df_pc_dersler.columns[1:]:

        # Calculate the weighted mean for the current outcome
        total_weight = 0
        weighted_outcomes_sum = 0

        for i in range(len(df)):
            course_number = df.iloc[i]['Ders']
            weight = df_pc_dersler[df_pc_dersler['Dersler']==course_number][column].values[0]
            total_weight += weight

            outcome_value = df.iloc[i][column]
            weighted_outcome = outcome_value * weight
            weighted_outcomes_sum += weighted_outcome

        # Handle the case where the total weight is zero to avoid division by zero error
        if total_weight == 0:
            weighted_mean = 0
        else:
            weighted_mean = weighted_outcomes_sum / total_weight

        # Store the mean value in the dictionary
        mean_values_dict[column] = weighted_mean

    # Return the mean values dictionary
    return mean_values_dict

def calculate_means_v4(df, df_pc_dersler):

    df_pc_dersler_index = df_pc_dersler.set_index('Dersler')

    # Initialize a dictionary to store the mean values
    mean_values_dict = {}

    # Iterate over each performance outcome (e.g., 'PÇ1', 'PÇ2', ...)
    for column in df_pc_dersler_index.columns:

        # Calculate the weighted mean for the current outcome
        total_weight = 0
        weighted_outcomes_sum = 0

        for i in range(len(df)):
            course_number = df.iloc[i]['Ders']
            weight = df_pc_dersler_index.loc[course_number, column]
            total_weight += weight

            outcome_value = df.iloc[i][column]
            weighted_outcome = outcome_value * weight
            weighted_outcomes_sum += weighted_outcome

        # Handle the case where the total weight is zero to avoid division by zero error
        if total_weight == 0:
            weighted_mean = 0
        else:
            weighted_mean = weighted_outcomes_sum / total_weight

        # Store the mean value in the dictionary
        mean_values_dict[column] = weighted_mean

    # Return the mean values dictionary
    return mean_values_dict

def process_result_dfs_v3(result_dfs, df_pc_dersler):
    # List of column names
    column_names = ['Numara', 'Ad', 'Soyad', 'PÇ1', 'PÇ2', 'PÇ3', 'PÇ4', 'PÇ5', 'PÇ6', 'PÇ7', 'PÇ8', 'PÇ9', 'PÇ10', 'PÇ11']

    # Create an empty list to store DataFrames
    df_list = []

    # Iterate over the result dataframes
    for i in range(len(result_dfs)):
        if result_dfs[i].empty:
            print('There is an empty DataFrame')
            continue

        # Create a temporary dictionary to store the values
        temp_dict = {'Numara': result_dfs[i]['Numara'][0],
                     'Ad': result_dfs[i]['Ad'][0],
                     'Soyad': result_dfs[i]['Soyad'][0]}

        # Calculate the means
        means_dict = calculate_means_v3(result_dfs[i], df_pc_dersler)

        # Combine the two dictionaries
        combine_dict = {**temp_dict, **means_dict}

        # Create a DataFrame from the combined dictionary
        df_dictionary = pd.DataFrame([combine_dict])

        # Append the DataFrame to the list
        df_list.append(df_dictionary)

    # Concatenate the list of DataFrames
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame(columns=column_names)  # Create an empty DataFrame

    # Replace NaN values with zero
    df = df.fillna(0)

    return df


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
