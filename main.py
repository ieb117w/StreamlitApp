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
st.sidebar.title("About")

# Write a description of the project
st.sidebar.write("This application calculates the rates at which graduates meet their program outcomes.")

st.sidebar.subheader('Instructions', divider='orange')

# Add a list of instructions
instructions = [
    "Enter a sample course ID",
    "Upload course evaulation reports.",
    "Upload the graduation list.",
    "Upload the course list file.",
    "Run the code and display results.",
    "Download the resulting Excel file."
]
for i, instruction in enumerate(instructions,1):
    st.sidebar.markdown(f"{i}. {instruction}")

header = st.container()
dataset = st.container()
results = st.container()

with header:
    st.title('Program Objectives Project')
    
    st.write("This application calculates the rates at which graduates meet their program outcomes.")
        
    # Create course id pattern
    st.markdown("**IMPORTANT NOTICE: Before proceeding please provide a sample course id pattern, e.g., CSS210. **")
    with st.form(key='my_form'):
        sample_id = st.text_input(label='Enter a sample course ID')
        submit_id_button = st.form_submit_button(label='Submit_sample_id')
#     sample_id = st.text_input('Enter a sample course ID', 'XYZ101')
        course_id_pattern = create_pattern(sample_id)
    
with dataset:
    st.subheader('Step 1: Upload Course Evaulation Reports',divider='orange')
    
    # Sample report DataFrame
    ex_report = pd.read_excel('ex_report.xlsx')
    
    if st.checkbox("Display a Sample Course Evaulation Report", key = 'ex_report'):
        st.dataframe(ex_report)
   
    eval_files = st.file_uploader("Choose all the Excel files for course reports", accept_multiple_files=True, key='file_uploader1')
    
    st.write(f'{len(eval_files)} files are uploaded.')

    st.subheader('Step 2: Upload the Graduation List', divider = 'orange')
    
    # Sample Grad list DataFrame
    ex_grad_list = pd.read_excel('ex_grad_list.xlsx')

    if st.checkbox("Display a Sample Graduation List", key = 'ex_grads'):
        st.dataframe(ex_grad_list)
    
    # Create a file uploader
    grad_list_file = st.file_uploader("Choose an Excel file of graduation list", key='file_uploader2')
    
    # Read the uploaded file to a Pandas DataFrame
    if grad_list_file is not None:
        df_mezun_list = pd.read_excel(grad_list_file)
        disp_df = df_mezun_list.style.format(precision=0, thousands='')
        # Display the DataFrame
        st.write(disp_df)

    st.subheader('Step 3: Upload the Course List File', divider = 'orange')
    
    # Sample Weights DataFrame
    ex_weights = pd.read_excel('ex_weights.xlsx')
    
    if st.checkbox("Display a Sample Graduation List", key = 'ex_weights'):
        st.dataframe(ex_weights)
    
    # Create a file uploader
    course_list_file = st.file_uploader("Choose the Excel file of Course-Outcome relation", key='file_uploader3')

    # Read the uploaded file to a Pandas DataFrame
    if course_list_file is not None:
        df_pc_dersler = pd.read_excel(course_list_file)
        
        # Apply the extract_pattern function to each row in the 'Dersler' column
        df_pc_dersler['Dersler'] = df_pc_dersler['Dersler'].apply(lambda x: extract_pattern(x, course_id_pattern))

        # Display the DataFrame
        st.write(df_pc_dersler)
    
with results:
    if eval_files is not None and grad_list_file is not None and course_list_file is not None:
        if st.button("Get the Results", key = 'get_results'):
            with st.spinner("Loading the results"):
                # Get all excel files in the folder as list of dataframes
                all_df = process_excel_files(eval_files, course_id_pattern)

                # Preprocess these dataframes
                processed_df = [preprocess_data(df) for df in all_df]

                # Store the id's in a list
                mezun_list = list(df_mezun_list['Öğrenci No'])

                # Get result_dfs
                result_dfs = extract_rows_by_numbers(processed_df, mezun_list)

                # Remove the empty dataframes of students and get which ones are deleted
                result_dfs, deleted_ids = delete_empty_dfs(result_dfs, mezun_list)

                # Create a dataframe of deleted ones
                deleted_df = df_mezun_list[df_mezun_list['Öğrenci No'].isin(deleted_ids)]
                deleted_students = deleted_df.copy()
                deleted_students = deleted_df.style.format(precision=0, thousands='')
                
                st.subheader('Students not considered for evaluation', divider = "orange")
                st.write('These students are not included in the computations because all PC values are zero.')
                st.dataframe(deleted_students)

                st.subheader('Results', divider = 'orange')
                df_students = [process_student_df(df_pc_dersler, df2) for df2 in result_dfs]
                df = process_result_dfs_v5(df_students, df_pc_dersler)
                df = append_average_row(df)
                st.dataframe(df)

                st.subheader('Downloading the results as Excel file', divider = 'orange')
                # Create a download button
                def download_excel(df):
                    df.to_excel('dataframe.xlsx', index=False)
                    with open('dataframe.xlsx', 'rb') as f:
                        data = f.read()
                    st.download_button(
                        label="Download the results",
                        data=data,
                        key='download_excel',
                        file_name='dataframe.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )

                # Call the download button function
                download_excel(df)

                st.subheader('Data analysis', divider = 'orange')
                name = st.selectbox('Select a name:', df['Ad'])
                # Print the selected name
                st.write('You selected:', name)
