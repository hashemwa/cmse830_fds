# Midterm Project: Student Performance Analyzer

## Project Overview

This project is a Streamlit web application designed to analyze student performance and well-being. It uses two distinct datasets to explore the various factors that influence academic success. This application is an interactive tool for data exploration, visualization, and analysis, and it fulfills the requirements for the CMSE 830 midterm project.

## Features

-   **Interactive UI**: A modern, dark-themed, and user-friendly interface built with Streamlit.
-   **Data Processing**: Demonstrates data cleaning and imputation techniques.
-   **Exploratory Data Analysis (EDA)**: Includes multiple interactive plot types (histograms, box plots, scatter plots, and correlation heatmaps) and statistical summaries.
-   **Advanced Analysis**: A section for more in-depth analysis, including merging datasets.
-   **Modular Code**: The code is organized into separate files for clarity and maintainability.

## How to Run the Application

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```

2.  **Install Dependencies**
    Make sure you have Python 3.7+ installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    To start the Streamlit application, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    The application will open in a new tab in your default web browser.

## How to Deploy on Streamlit Community Cloud

You can deploy this application for free on Streamlit Community Cloud.

1.  **Push your code to a GitHub repository.**
2.  **Go to [share.streamlit.io](https://share.streamlit.io) and sign up.**
3.  **Click on the "New app" button and connect your GitHub account.**
4.  **Select your repository and the `app.py` file.**
5.  **Click "Deploy!".**

Your application will then be deployed and accessible to anyone with the link.

## Datasets

The project uses two datasets:

-   `dataset1.csv`: Contains information about students' academic performance.
-   `dataset2.csv`: Contains information about students' well-being and mental health.

## Project Structure