import streamlit as st
import re
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Predefined lists of skills and common education keywords
SKILLS_DB = [
    'Python', 'Java', 'SQL', 'Machine Learning', 'Data Science', 'Deep Learning', 
    'NLP', 'TensorFlow', 'Keras', 'Flask', 'Django', 'Pandas', 'NumPy', 'Matplotlib', 
    'Data Analysis', 'AI', 'AWS', 'GCP', 'Azure', 'Hadoop'
]

EDUCATION_DB = [
    'B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'PhD', 'MBA', 'Bachelor', 'Master', 'Diploma', 'Degree'
]

PROFICIENCY_KEYWORDS = {
    "expert": 3,
    "proficient": 2,
    "intermediate": 1
}

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

# Function to extract skills based on predefined list and estimate proficiency level
def extract_skills_with_proficiency(text):
    skills = []
    proficiency_levels = {}
    
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills.append(skill)
            
            # Check for proximity to proficiency keywords
            for prof_keyword, level in PROFICIENCY_KEYWORDS.items():
                pattern = r'\b' + re.escape(prof_keyword) + r'\b.*\b' + re.escape(skill) + r'\b|\b' + re.escape(skill) + r'\b.*\b' + re.escape(prof_keyword) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    proficiency_levels[skill] = prof_keyword.capitalize()
                    break
            else:
                proficiency_levels[skill] = 'Not Specified'
    
    return skills, proficiency_levels

# Function to extract education details based on predefined list and analyze the highest level
def extract_education(text):
    education_details = [edu for edu in EDUCATION_DB if re.search(r'\b' + re.escape(edu) + r'\b', text, re.IGNORECASE)]
    
    highest_level = 'Not Found'
    if 'PhD' in education_details:
        highest_level = 'PhD'
    elif 'M.Tech' in education_details or 'M.Sc' in education_details or 'Master' in education_details:
        highest_level = 'Master'
    elif 'B.Tech' in education_details or 'B.Sc' in education_details or 'Bachelor' in education_details:
        highest_level = 'Bachelor'
    elif 'Diploma' in education_details:
        highest_level = 'Diploma'

    return education_details, highest_level

# Function to extract company names and experience duration
def extract_experience(text):
    companies = re.findall(r'\b\w+(?:\s\w+)*(?: Inc| LLC| Ltd| Technologies| Corp)\b', text)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    
    if len(years) >= 2:
        start_year, end_year = min(years), max(years)
        experience_duration = int(end_year) - int(start_year)
    else:
        experience_duration = 0
    
    return companies, experience_duration

# Function to match resume with a job description (based on skills)
def match_job_description(resume_text, job_description):
    resume_skills, _ = extract_skills_with_proficiency(resume_text)
    job_skills, _ = extract_skills_with_proficiency(job_description)
    
    if not job_skills:
        return 0, [], job_skills  # No skills in job description
    
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    
    # Calculate percentage match
    match_percentage = (len(matched_skills) / len(job_skills)) * 100
    
    return match_percentage, matched_skills, job_skills

# Visualization Functions

# Plot skill matching
def plot_skill_match(matched_skills, job_skills):
    if len(job_skills) == 0:
        st.write("No skills found in the job description for matching.")
        return

    matched_count = len(matched_skills)
    unmatched_count = len(job_skills) - matched_count

    if matched_count == 0 and unmatched_count == 0:
        st.write("No skills available for comparison.")
        return

    labels = ['Matched Skills', 'Unmatched Skills']
    sizes = [matched_count, unmatched_count]
    colors = ['#4CAF50', '#FF6347']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Smart Resume Analyzer with Job Matching Percentage")

    st.write("Upload your resume in PDF format to extract and analyze relevant details, and match it with a job description.")

    pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if pdf_file:
        resume_text = extract_text_from_pdf(pdf_file)
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        email, phone = extract_contact_details(resume_text)
        st.subheader("Contact Information:")
        st.write(f"Email: {', '.join(email) if email else 'No email found'}")
        st.write(f"Phone: {', '.join(phone) if phone else 'No phone number found'}")

        skills, proficiency_levels = extract_skills_with_proficiency(resume_text)
        st.subheader("Extracted Skills and Proficiency Levels:")
        for skill in skills:
            st.write(f"{skill}: {proficiency_levels[skill]}")

        experience, experience_duration = extract_experience(resume_text)
        st.subheader("Work Experience (Companies and Duration):")
        st.write(f"Companies: {', '.join(experience) if experience else 'No experience found'}")
        st.write(f"Experience Duration: {experience_duration} years" if experience_duration else "Experience duration not found")

        education, highest_education = extract_education(resume_text)
        st.subheader("Education Details and Highest Level:")
        st.write(f"Education: {', '.join(education) if education else 'No education details found'}")
        st.write(f"Highest Level of Education: {highest_education}")

        # Job Description Matching (optional)
        st.subheader("Job Description Matching")
        job_desc = st.text_area("Paste Job Description", "")
        if job_desc:
            match_percentage, matched_skills, job_skills = match_job_description(resume_text, job_desc)
            st.write(f"Match Score: {match_percentage:.2f}%")

            # Skill match visualization
            st.subheader("Skill Match Visualization")
            plot_skill_match(matched_skills, job_skills)

if __name__ == "__main__":
    main()
