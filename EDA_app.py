import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def display_data_info(data):
    st.subheader("Data Information")
    st.write("Data Information:")
    st.text(data.info())
    st.write("Descriptive Statistics:")
    st.text(data.describe())

def preprocess_text(text):
    translator = str.maketrans("", "", string.punctuation)
    stop_words = set(stopwords.words('english'))
    text = text.translate(translator)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def display_textual_column_info(data, column_name):
    st.subheader(f"Analysis of '{column_name}' column")

    unique_values = data[column_name].unique()
    st.write(f"Unique values: {unique_values}")

    st.subheader(f"{column_name} Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data[column_name].astype(str)))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader(f"Top 10 Words (without stop words) in {column_name}")
    words = " ".join(data[column_name].astype(str)).split()
    words = [preprocess_text(word) for word in words]
    word_counts = Counter(words)
    top_words = dict(word_counts.most_common(10))

    # Convert the dictionary to DataFrame
    top_words_df = pd.DataFrame(list(top_words.items()), columns=['Word', 'Count'])

    # Display the bar chart
    st.bar_chart(top_words_df.set_index('Word'))

def display_numerical_column_info(data, columns):
    st.subheader("Analysis of Selected Numerical Columns")

    for column_name in columns:
        st.subheader(f"Analysis of '{column_name}' column (Numerical)")
        st.write(f"Statistics for {column_name}:")
        st.write(data[column_name].describe())

        st.subheader(f"{column_name} Histogram")
        fig, ax = plt.subplots()
        ax.hist(data[column_name], bins='auto')
        st.pyplot(fig)

    st.subheader(f"Covariance Matrix for Selected Numerical Columns")
    covariance_matrix = data[columns].cov()
    st.write(covariance_matrix)

    st.subheader("Covariance Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Pair Plot for Selected Numerical Columns")
    pair_plot = sns.pairplot(data[columns])
    st.pyplot(pair_plot.fig)

    st.subheader("Correlation Matrix for Numerical Columns")
    correlation_matrix = data.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

st.header('Welcome to the Exploratory Data Analysis Webapp')

st.sidebar.subheader("App Information")
st.sidebar.markdown("This app performs Exploratory Data Analysis (EDA) on the uploaded CSV file.")
st.sidebar.markdown("This app is created by Abhijeet Anand")

st.write(
    "This application is designed to facilitate the analysis of your datasets through a simple and interactive interface. "
    "Whether you're a data scientist, analyst, or enthusiast, this tool empowers you to gain insights into your data effortlessly."
)

st.write(
    "To get started, upload a CSV file using the file uploader below. Once uploaded, you'll be able to explore the contents of "
    "the dataset and view the data types of each column. Dive into the world of EDA and uncover patterns, trends, and valuable "
    "information within your data!"
)

refresh_button = st.sidebar.button("Refresh")
if refresh_button:
    st.success("App refreshed to the beginning!")
    st.experimental_rerun()

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df)
    display_data_info(df)

    text_columns = df.select_dtypes(include='object').columns
    numerical_columns = df.select_dtypes(include=['number', 'datetime']).columns

    selected_columns_type = st.sidebar.selectbox("Select columns for analysis", ['text', 'numerical'])

    if selected_columns_type == 'text':
        selected_columns = st.sidebar.selectbox("Select textual columns", df.select_dtypes(include=['object']).columns)
        if selected_columns:
            display_textual_column_info(df, selected_columns)
        else:
            st.write("There are no text columns in your data")

    elif selected_columns_type == 'numerical':
        selected_columns = st.sidebar.multiselect("Select numerical columns", df.select_dtypes(include=['number']).columns)
        if selected_columns:
            display_numerical_column_info(df, selected_columns)
        else:
            st.write("There are no numerical columns in your data")
