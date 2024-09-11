import streamlit as st
import re
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from datetime import datetime

# Predefined lists of skills and categories
SKILLS_DB = {
    'Programming': ['Python', 'Java', 'C++', 'JavaScript', 'SQL'],
    'Data Science': ['Machine Learning', 'Deep Learning', 'Data Science', 'NLP', 'TensorFlow', 'Keras', 'Pandas', 'NumPy'],
    'Cloud': ['AWS', 'GCP', 'Azure', 'Hadoop'],
    'Web Development': ['Flask', 'Django', 'HTML', 'CSS', 'React', 'Node.js']
}

EDUCATION_DB = ['B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'PhD', 'MBA', 'Bachelor', 'Master', 'Diploma', 'Degree']

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

# Function to extract skills based on predefined categories
def extract_skills_by_category(text):
    skill_categories = {category: [] for category in SKILLS_DB}
    for category, skills in SKILLS_DB.items():
        for skill in skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                skill_categories[category].append(skill)
    return skill_categories

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

# Function to extract company names, job titles, and locations using regex
def extract_companies_job_titles(text):
    # Pattern to capture potential company names and job titles
    company_pattern = r'\b(?:Inc|LLC|Ltd|Technologies|Corp|Company|Enterprises|Solutions)\b'
    title_pattern = r'\b(?:Manager|Engineer|Developer|Analyst|Consultant|Specialist|Lead|Director)\b'
    
    companies = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*(?: ' + company_pattern + r')', text)
    job_titles = re.findall(title_pattern, text)
    
    return companies, job_titles

# Function to create a word cloud of extracted skills
def generate_wordcloud(skills):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(skills))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to visualize skill categories as a bar chart
def visualize_skill_categories(skill_categories):
    categories = {category: len(skills) for category, skills in skill_categories.items()}
    df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
    st.bar_chart(df.set_index('Category'))

# Function to match job description skills with resume skills
def match_job_description(resume_skills, job_description):
    job_skills = extract_skills_by_category(job_description)
    matched_skills = {cat: [skill for skill in resume_skills[cat] if skill in job_skills[cat]] for cat in resume_skills}
    return matched_skills

# Function to visualize the experience timeline
def visualize_experience_timeline(companies, dates):
    experience_df = pd.DataFrame({'Company': companies, 'Year': dates})
    fig, ax = plt.subplots()
    experience_df['Year'] = pd.to_numeric(experience_df['Year'])
    experience_df.set_index('Year').plot(kind='barh', ax=ax, legend=False)
    ax.set_xlabel('Year')
    ax.set_ylabel('Company')
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Enhanced Smart Resume Analyzer (No spaCy)")

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

        skill_categories = extract_skills_by_category(resume_text)
        st.subheader("Extracted Skills by Category:")
        for category, skills in skill_categories.items():
            st.write(f"{category}: {', '.join(skills) if skills else 'None'}")

        if any(skill_categories.values()):
            st.subheader("Skills Word Cloud")
            all_skills = [skill for skills in skill_categories.values() for skill in skills]
            generate_wordcloud(all_skills)

            st.subheader("Skills Category Analysis")
            visualize_skill_categories(skill_categories)

        education = extract_education(resume_text)
        st.subheader("Education Details:")
        st.write(", ".join(education) if education else "No education details found")

        experience_duration = extract_experience_duration(resume_text)
        st.subheader("Total Years of Experience:")
        st.write(f"{experience_duration} years")

        companies, job_titles = extract_companies_job_titles(resume_text)
        st.subheader("Companies Worked For:")
        st.write(", ".join(companies) if companies else "No companies found")

        st.subheader("Job Titles:")
        st.write(", ".join(job_titles) if job_titles else "No job titles found")

        st.subheader("Experience Timeline (based on extracted dates)")
        experience_years = re.findall(r'(19|20)\d{2}', resume_text)
        if companies and experience_years:
            visualize_experience_timeline(companies[:len(experience_years)], experience_years)

        # Job Description Matching (optional)
        st.subheader("Job Description Matching")
        job_desc = st.text_area("Paste Job Description", "")
        if job_desc:
            matched_skills = match_job_description(skill_categories, job_desc)
            st.write("Matched Skills by Category:")
            for category, skills in matched_skills.items():
                st.write(f"{category}: {', '.join(skills) if skills else 'None'}")

if __name__ == "__main__":
    main()
