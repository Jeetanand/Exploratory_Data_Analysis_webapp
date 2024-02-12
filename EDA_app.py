import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import streamlit_scrollable_textbox as stx
import string
from collections import Counter  # Add this import statement
import nltk
from nltk.corpus import stopwords



nltk.download('stopwords')

def display_data_info(data):
    st.subheader("Data Information")
    
    # Display basic information using df.info()
    st.write("Data Information:")
    st.text(data.info())

    # Display descriptive statistics using df.describe()
    st.write("Descriptive Statistics:")
    st.text(data.describe())

def preprocess_text(text):
    # Remove punctuation, convert to lowercase, and remove stop words
    translator = str.maketrans("", "", string.punctuation)
    stop_words = set(stopwords.words('english'))
    text = text.translate(translator)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def display_textual_column_info(data, column_name):
    st.subheader(f"Analysis of '{column_name}' column")

    # Display unique values
    unique_values = data[column_name].unique()
    st.write(f"Unique values: {unique_values}")

   

    # Display a word cloud for the column
    st.subheader(f"{column_name} Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data[column_name].astype(str)))
    
    # Explicitly pass the figure to st.pyplot()
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Display top 10 words without stop words
    st.subheader(f"Top 10 Words (without stop words) in {column_name}")
    words = " ".join(data[column_name].astype(str)).split()
    words = [preprocess_text(word) for word in words]
    word_counts = Counter(words)
    top_words = dict(word_counts.most_common(10))
    st.bar_chart(top_words)
   
def display_numerical_column_info(data, columns):
    st.subheader("Analysis of Selected Numerical Columns")

    for column_name in columns:
        st.subheader(f"Analysis of '{column_name}' column (Numerical)")

        # Display basic statistics
        st.write(f"Statistics for {column_name}:")
        st.write(data[column_name].describe())

        # Display a histogram for numerical columns
        st.subheader(f"{column_name} Histogram")
        fig, ax = plt.subplots()
        ax.hist(data[column_name], bins='auto')
        st.pyplot(fig)

        # Display covariance matrix
    st.subheader(f"Covariance Matrix for Selected Numerical Columns")
    covariance_matrix = data[columns].cov()
    st.write(covariance_matrix)

    # Display covariance heatmap
    st.subheader("Covariance Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)



     # Display pair plot for selected columns
    st.subheader("Pair Plot for Selected Numerical Columns")
    pair_plot = sns.pairplot(data[columns])
    st.pyplot(pair_plot.fig)

    # Display correlation matrix
    st.subheader("Correlation Matrix for Numerical Columns")
    correlation_matrix = data.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

    

st.header('Welcome to ths Exploratory Data Analysis Webapp')
# Sidebar options

st.sidebar.subheader("App Information")
st.sidebar.markdown("This app performs Exploratory Data Analysis (EDA) on the uploaded CSV file.")
st.sidebar.markdown("This app is created by Abhijeet Anand")


st.write(
        "This application is designed to facilitate"
        " the analysis of your datasets through a simple and interactive interface. Whether you're a data scientist,"
        " analyst, or enthusiast, this tool empowers you to gain insights into your data effortlessly."
    )
st.write(
        "To get started, upload a CSV file using the file uploader below. Once uploaded, you'll be able to explore the"
        " contents of the dataset and view the data types of each column. Dive into the world of EDA and uncover patterns,"
        " trends, and valuable information within your data!"
    )
refresh_button = st.sidebar.button("Refresh")
if refresh_button:
            st.success("App refreshed to the beginning!")
            st.experimental_rerun()
# File Upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
        


        st.success("File uploaded successfully!")

        # Displaying the contents of the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("You Sucessfully Uploaded CSV file:")
 # Displaying data types
         # Display raw data
        st.subheader("Raw Data")
        st.write(df)
        

        # Display data information
        display_data_info(df)
         

         # Displaying only text and numerical columns
        text_columns = df.select_dtypes(include='object').columns
        numerical_columns = df.select_dtypes(include=['number', 'datetime']).columns


      
     
         # Select columns for analysis
       # Select columns for analysis
        selected_columns_type = st.sidebar.selectbox("Select columns for analysis", ['text', 'numerical'])

        if selected_columns_type == 'text':
                selected_columns = st.sidebar.selectbox("Select textual columns", df.select_dtypes(include=['object']).columns)
                if selected_columns:
                     display_textual_column_info(df, selected_columns)
                else:
                     st.write("There is no text columns in your data")
        

        elif selected_columns_type == 'numerical':
                selected_columns = st.sidebar.multiselect("Select numerical columns", df.select_dtypes(include=['number']).columns)
                if selected_columns:
                      display_numerical_column_info(df, selected_columns)
                else:
                      st.write("There is no numerical columns in your data")
   
            


        


