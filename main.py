import streamlit as st
import re
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Predefined lists of skills and common education keywords
SKILLS_DB = [
    'Python', 'Java', 'SQL', 'Machine Learning', 'Data Science', 'Deep Learning', 
    'NLP', 'TensorFlow', 'Keras', 'Flask', 'Django', 'Pandas', 'NumPy', 'Matplotlib', 
    'Data Analysis', 'AI', 'AWS', 'GCP', 'Azure', 'Hadoop'
]

EDUCATION_DB = [
    'B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'PhD', 'MBA', 'Bachelor', 'Master', 'Diploma', 'Degree'
]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract and preprocess resume text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Function to analyze sentiment of the career objective
def analyze_sentiment(text):
    sentences = sent_tokenize(text)
    sentiment_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
    return sentiment_scores

# Function to cluster skills
def cluster_skills(skills):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(skills)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    clusters = kmeans.labels_
    return clusters, kmeans.cluster_centers_

# Function to create a word cloud for skills
def generate_wordcloud(skills):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(skills))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to visualize clustered skills
def visualize_clustered_skills(skills, clusters):
    df = pd.DataFrame({'Skill': skills, 'Cluster': clusters})
    cluster_counts = df['Cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Skills')
    plt.title('Skills Distribution by Cluster')
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Enhanced Smart Resume Analyzer")

    st.write("Upload your resume in PDF format to extract and analyze relevant details.")
    
    pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if pdf_file:
        resume_text = extract_text_from_pdf(pdf_file)
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        preprocessed_text = preprocess_text(resume_text)
        
        # Analyze Career Objectives
        career_objective = re.search(r'career objective(.*?)(education|experience|skills|certifications)', preprocessed_text, re.S)
        if career_objective:
            career_objective_text = career_objective.group(1)
            sentiment_scores = analyze_sentiment(career_objective_text)
            st.subheader("Career Objective Sentiment Analysis:")
            st.write(f"Sentiment Scores: {sentiment_scores}")
            st.write(f"Average Sentiment Score: {sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 'N/A'}")

        # Extract and Cluster Skills
        skills = [skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', preprocessed_text, re.IGNORECASE)]
        if skills:
            st.subheader("Extracted Skills:")
            st.write(", ".join(skills) if skills else "No skills found")
            st.subheader("Skills Word Cloud")
            generate_wordcloud(skills)
            clusters, cluster_centers = cluster_skills(skills)
            st.subheader("Clustered Skills Analysis")
            visualize_clustered_skills(skills, clusters)
        
        # Extract Education Details
        education = [edu for edu in EDUCATION_DB if re.search(r'\b' + re.escape(edu) + r'\b', preprocessed_text, re.IGNORECASE)]
        st.subheader("Education Details:")
        st.write(", ".join(education) if education else "No education details found")

        # Total Years of Experience
        experience_duration = re.findall(r'(19|20)\d{2}', resume_text)
        if experience_duration:
            experience_duration = sorted(set(int(year) for year in experience_duration))
            total_experience = experience_duration[-1] - experience_duration[0] if len(experience_duration) > 1 else 0
            st.subheader("Total Years of Experience:")
            st.write(f"{total_experience} years")

        # Contact Information
        email, phone = re.findall(r'\S+@\S+', resume_text), re.findall(r'\b\d{10}\b', resume_text)
        st.subheader("Contact Information:")
        st.write(f"Email: {', '.join(email) if email else 'No email found'}")
        st.write(f"Phone: {', '.join(phone) if phone else 'No phone number found'}")

if __name__ == "__main__":
    main()
