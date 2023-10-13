
import streamlit as st
import pandas as pd
import os
import glob
import numpy as np
import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from process_funcs import *

# Create a sidebar
st.sidebar.title("About the Project")

# Write a description of the project
st.sidebar.write("This is a sample project that demonstrates how to use Streamlit to create a simple app.")

header = st.container()
dataset = st.container()
results = st.container()

with header:
    st.title('Program Objectives Project')
    
    st.text_area("",
        "This app computes the Program Objectives for graduates.")
    
with dataset:
    st.title('Upload Course Evaulation Reports')

    eval_files = st.file_uploader("Upload the Excel files", accept_multiple_files=True, key='file_uploader1')
    
    st.write(f'{len(eval_files)} files are uploaded.')


    st.title('Upload the Graduation List')

    # Create a file uploader
    grad_list_file = st.file_uploader("Choose an Excel file", key='file_uploader2')

    # Read the uploaded file to a Pandas DataFrame
    if grad_list_file is not None:
        df_mezun_list = pd.read_excel(grad_list_file)
        disp_df = df_mezun_list.style.format(precision=0, thousands='')
        # Display the DataFrame
        st.write(disp_df)

    st.title('Upload the Course List File')
    # Create a file uploader
    course_list_file = st.file_uploader("Choose an Excel file", key='file_uploader3')

    # Read the uploaded file to a Pandas DataFrame
    if course_list_file is not None:
        df_pc_dersler = pd.read_excel(course_list_file)
        
        # Apply the extract_pattern function to each row in the 'Dersler' column
        df_pc_dersler['Dersler'] = df_pc_dersler['Dersler'].apply(extract_pattern)

        # Display the DataFrame
        st.write(df_pc_dersler)
    
with results:
    if eval_files is not None and grad_list_file is not None and course_list_file is not None:
        # Get all excel files in the folder as list of dataframes
        all_df = process_excel_files(eval_files)

        # Preprocess these dataframes
        processed_df = [preprocess_data(df) for df in all_df]

        # Store the id's in a list
        mezun_list = list(df_mezun_list['Öğrenci No'])

        # Get result_dfs
        result_dfs = extract_rows_by_id(processed_df, mezun_list)

        # Remove the empty dataframes of students and get which ones are deleted
        result_dfs, deleted_ids = delete_empty_dfs(result_dfs, mezun_list)

        # Create a dataframe of deleted ones
        deleted_df = df_mezun_list[df_mezun_list['Öğrenci No'].isin(deleted_ids)]
        deleted_students = deleted_df.copy()
        deleted_students = deleted_df.style.format(precision=0, thousands='')
        st.title('Deleted ones')
        st.dataframe(deleted_students)
        
        st.title('Results')
        df = process_result_dfs_v3(result_dfs, df_pc_dersler)
        df = append_average_row(df)
        df = df[(df != 0).all(1)].reset_index(drop=True)
        st.dataframe(df)
        
        # Create a download button
        def download_excel(df):
            df.to_excel('dataframe.xlsx', index=False)
            with open('dataframe.xlsx', 'rb') as f:
                data = f.read()
            st.download_button(
                label="Download Excel File",
                data=data,
                key='download_excel',
                file_name='dataframe.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

        # Call the download button function
        download_excel(df)
