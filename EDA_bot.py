from process import *
import gradio as gr
import sqlite3

import gradio as gr
import pandas as pd
import sqlite3

def handle_csv_upload(file):
    try:
        sample_df = pd.read_csv(file.name)

        categorical_columns = sample_df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = sample_df.select_dtypes(include=['number']).columns.tolist()

        eda_report_df_categorical = perform_eda_on_categorical(
            sample_df, categorical_columns, output_folder='./eda_outputs')
        eda_report_df_numerical = analyze_numerical_columns_with_visuals(
            sample_df, numerical_columns, output_dir='./eda_visuals')

        combined_eda_df = combine_eda_reports(
            eda_report_df_categorical, eda_report_df_numerical)
                # Save the DataFrames to CSV
        if not os.path.exists('./processed_output'):
             os.makedirs('./processed_output')

        sample_df.to_csv('./processed_output/sample_df.csv', index=False)
        combined_eda_df.to_csv('./processed_output/combined_eda_df.csv', index=False)
        db_path = './sql_lite_database.db'
        upload_dataframe_to_sql(sample_df, db_path, 'input_dataframe1')
        upload_dataframe_to_sql(combined_eda_df, db_path, 'report_dataframe1')

        return "EDA completed, and data saved to the database!"
    except Exception as e:
        return f"Error occurred: {e}"

# Chatbot function
def chatbot_interface(query):
    global sample_df, combined_eda_df
    db_path = './sql_lite_database.db'
    try:
        result = get_answer(query)
        return  result.get('output')
    except Exception as e:
        return f"Error occurred: {e}"

# CSS for styling
css = """
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f5;
    margin: 0;
    padding: 0;
}
.title {
    text-align: center;
    font-size: 36px;
    color: #333;
    margin-bottom: 20px;
}
.gr-button {
    background-color: #4CAF50 !important; /* Green */
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    cursor: pointer !important;
    border-radius: 5px !important;
}
.gr-button:hover {
    background-color: #45a049 !important;
}
.ask-button {
    background-color: #007BFF !important; /* Blue */
    color: white !important;
}
.ask-button:hover {
    background-color: #0056b3 !important;
}
"""

# Gradio Interface
with gr.Blocks(css=css) as app:
    gr.Markdown("## 📊 EDA Chatbot ", elem_id="title")

    with gr.Tab("Upload and Process CSV"):
        with gr.Row():
            file_input = gr.File(label="📁 Upload CSV File")
        with gr.Row():
            process_button = gr.Button("📤 Process CSV")
        with gr.Row():
            process_output = gr.Textbox(label="📋 Output", interactive=False)
        process_button.click(handle_csv_upload, inputs=file_input, outputs=process_output)

    with gr.Tab("Chat with Database"):
        with gr.Row():
            query_input = gr.Textbox(label="📝 Ask your question:")
        with gr.Row():
            query_button = gr.Button("🗣️ Ask", elem_id="ask-button")
        with gr.Row():
            query_output = gr.Textbox(label="🤖 Chatbot Response", interactive=False)
        query_button.click(chatbot_interface, inputs=query_input, outputs=query_output)

if __name__ == "__main__":
    app.launch(share=True)
