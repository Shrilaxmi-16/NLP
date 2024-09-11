import streamlit as st
import re
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

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

# Function to extract contact details using regex
def extract_contact_details(text):
    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\b\d{10}\b', text)
    return email, phone

# Function to extract skills based on predefined list
def extract_skills(text):
    skills = [skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
    return skills

# Function to extract education details based on predefined list
def extract_education(text):
    education_details = [edu for edu in EDUCATION_DB if re.search(r'\b' + re.escape(edu) + r'\b', text, re.IGNORECASE)]
    return education_details

# Function to extract company names and experience
def extract_experience(text):
    companies = re.findall(r'\b\w+(?:\s\w+)*(?: Inc| LLC| Ltd| Technologies| Corp)\b', text)
    # Extract job titles and years (optional improvement)
    job_titles = re.findall(r'\b(?:Manager|Engineer|Developer|Consultant|Analyst|Lead|Director)\b', text)
    experience_dates = re.findall(r'\b(?:19|20)\d{2}\b', text)
    return companies, job_titles, experience_dates

# Function to match resume with a job description (simplified)
def match_job_description(text, job_description):
    skills = extract_skills(text)
    job_skills = extract_skills(job_description)
    matched_skills = [skill for skill in skills if skill in job_skills]
    return matched_skills, len(matched_skills) / len(job_skills) if job_skills else 0

# Function to create word cloud of extracted skills
def generate_wordcloud(skills):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(skills))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to visualize skill match comparison using a bar chart
def visualize_skill_match(resume_skills, job_skills):
    skill_data = {'Skills': resume_skills, 'Match': [1 if skill in job_skills else 0 for skill in resume_skills]}
    df = pd.DataFrame(skill_data)
    st.bar_chart(df.set_index('Skills'))

# Function to visualize education using a pie chart
def visualize_education(education_details):
    edu_count = pd.Series(education_details).value_counts()
    fig, ax = plt.subplots()
    ax.pie(edu_count, labels=edu_count.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Smart Resume Analyzer with Visualizations")
    
    st.write("Upload your resume in PDF format to extract and analyze relevant details.")
    
    pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if pdf_file:
        resume_text = extract_text_from_pdf(pdf_file)
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        email, phone = extract_contact_details(resume_text)
        st.subheader("Contact Information:")
        st.write(f"Email: {', '.join(email) if email else 'No email found'}")
        st.write(f"Phone: {', '.join(phone) if phone else 'No phone number found'}")

        skills = extract_skills(resume_text)
        st.subheader("Extracted Skills:")
        st.write(", ".join(skills) if skills else "No skills found")

        if skills:
            st.subheader("Skills Word Cloud")
            generate_wordcloud(skills)

        companies, job_titles, experience_dates = extract_experience(resume_text)
        st.subheader("Work Experience (Companies):")
        st.write(", ".join(companies) if companies else "No experience found")
        
        st.subheader("Job Titles:")
        st.write(", ".join(job_titles) if job_titles else "No job titles found")
        
        st.subheader("Experience Dates:")
        st.write(", ".join(experience_dates) if experience_dates else "No dates found")

        education = extract_education(resume_text)
        st.subheader("Education Details:")
        st.write(", ".join(education) if education else "No education details found")

        if education:
            st.subheader("Education Distribution")
            visualize_education(education)

        # Job Description Matching (optional)
        st.subheader("Job Description Matching")
        job_desc = st.text_area("Paste Job Description", "")
        if job_desc:
            matched_skills, match_score = match_job_description(resume_text, job_desc)
            st.write(f"Match Score: {match_score:.2%}")
            st.write("Matched Skills: ", ", ".join(matched_skills))

            st.subheader("Skill Match Visualization")
            visualize_skill_match(skills, extract_skills(job_desc))

if __name__ == "__main__":
    main()
