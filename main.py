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

# Function to extract education details
def extract_education(text):
    education_details = [edu for edu in EDUCATION_DB if re.search(r'\b' + re.escape(edu) + r'\b', text, re.IGNORECASE)]
    return education_details

# Function to calculate total years of experience based on dates found in the resume
def extract_experience_duration(text):
    dates = re.findall(r'(19|20)\d{2}', text)
    if dates:
        years = list(map(int, dates))
        years.sort()
        return years[-1] - years[0] if years else 0
    return 0

# Function to extract company names and job titles
def extract_companies_job_titles(text):
    company_pattern = r'\b(?:Inc|LLC|Ltd|Technologies|Corp|Company|Enterprises|Solutions)\b'
    title_pattern = r'\b(?:Manager|Engineer|Developer|Analyst|Consultant|Specialist|Lead|Director)\b'
    
    companies = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*(?: ' + company_pattern + r')', text)
    job_titles = re.findall(title_pattern, text)
    
    return companies, job_titles

# Function to visualize experience timeline
def visualize_experience_timeline(companies, dates):
    if not companies or not dates:
        st.write("Not enough data to display the experience timeline.")
        return

    # Ensure lengths are the same
    min_length = min(len(companies), len(dates))
    if min_length == 0:
        st.write("No data available for experience timeline.")
        return

    companies = companies[:min_length]
    dates = dates[:min_length]

    # Create DataFrame
    experience_df = pd.DataFrame({'Company': companies, 'Year': dates})

    # Convert 'Year' to numeric
    try:
        experience_df['Year'] = pd.to_numeric(experience_df['Year'])
    except ValueError:
        st.write("Error converting years to numeric values.")
        return

    # Plot
    fig, ax = plt.subplots()
    experience_df.set_index('Year').plot(kind='barh', ax=ax, legend=False)
    ax.set_xlabel('Year')
    ax.set_ylabel('Company')
    st.pyplot(fig)

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
        career_objective_match = re.search(r'career objective(.*?)(education|experience|skills|certifications)', preprocessed_text, re.S)
        if career_objective_match:
            career_objective_text = career_objective_match.group(1)
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
            clusters, _ = cluster_skills(skills)
            st.subheader("Clustered Skills Analysis")
            visualize_clustered_skills(skills, clusters)
        
        # Extract Education Details
        education = extract_education(preprocessed_text)
        st.subheader("Education Details:")
        st.write(", ".join(education) if education else "No education details found")

        # Total Years of Experience
        experience_duration = extract_experience_duration(resume_text)
        st.subheader("Total Years of Experience:")
        st.write(f"{experience_duration} years")

        # Companies and Job Titles
        companies, job_titles = extract_companies_job_titles(resume_text)
        st.subheader("Companies Worked For:")
        st.write(", ".join(companies) if companies else "No companies found")

        st.subheader("Job Titles:")
        st.write(", ".join(job_titles) if job_titles else "No job titles found")

        # Experience Timeline
        experience_dates = re.findall(r'(19|20)\d{2}', resume_text)
        if companies and experience_dates:
            visualize_experience_timeline(companies[:len(experience_dates)], experience_dates)

        # Job Description Matching (optional)
        st.subheader("Job Description Matching")
        job_desc = st.text_area("Paste Job Description", "")
        if job_desc:
            job_skills = [skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', preprocess_text(job_desc), re.IGNORECASE)]
            matched_skills = {cat: [skill for skill in skills if skill in job_skills] for cat, skills in SKILLS_DB.items()}
            st.write("Matched Skills by Category:")
            for category, skills in matched_skills.items():
                st.write(f"{category}: {', '.join(skills) if skills else 'None'}")

if __name__ == "__main__":
    main()
