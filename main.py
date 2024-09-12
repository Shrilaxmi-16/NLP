import streamlit as st
import re
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract emails
def extract_emails(text):
    email_pattern = r'[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+'
    emails = re.findall(email_pattern, text)
    return emails

# Function to extract phone numbers
def extract_phone_numbers(text):
    phone_pattern = r'\b\d{10}\b|\+?\d[\d -]{8,12}\d'
    phone_numbers = re.findall(phone_pattern, text)
    return phone_numbers

# Function to extract skills (based on a predefined list of skills)
def extract_skills(text):
    skills = ["Python", "Machine Learning", "Data Science", "SQL", "Java", "Deep Learning", "NLP", "TensorFlow", "Keras", "Pandas", "NumPy"]
    found_skills = [skill for skill in skills if re.search(r'\b' + skill + r'\b', text, re.IGNORECASE)]
    return found_skills

# Function to plot skills distribution
def plot_skills_distribution(skills):
    skill_counts = Counter(skills)
    
    if skill_counts:
        labels, values = zip(*skill_counts.items())
        plt.bar(labels, values)
        plt.title("Skills Distribution")
        plt.xlabel("Skills")
        plt.ylabel("Count")
        st.pyplot(plt)
    else:
        st.write("No skills to display.")

# Function to extract sections like education, experience, etc.
def extract_sections(text):
    sections = {
        'Experience': r'(experience|work experience|employment history)',
        'Education': r'(education|academic background)',
        'Skills': r'(skills|technical skills)',
        'Projects': r'(projects|relevant projects)'
    }
    
    extracted_sections = {}
    
    for section, pattern in sections.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_sections[section] = match.group(0)
    
    return extracted_sections

# Streamlit App
def main():
    st.title("Smart Resume Analyzer Without Pre-trained NLP Model")

    st.subheader("Upload a resume for analysis")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt", "pdf"])

    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode('utf-8')
        st.text_area("Resume Content", resume_text, height=200)

        # Extracting Emails
        st.subheader("Extracted Emails")
        emails = extract_emails(resume_text)
        if emails:
            st.write(emails)
        else:
            st.write("No emails found.")

        # Extracting Phone Numbers
        st.subheader("Extracted Phone Numbers")
        phone_numbers = extract_phone_numbers(resume_text)
        if phone_numbers:
            st.write(phone_numbers)
        else:
            st.write("No phone numbers found.")

        # Extracting Skills
        st.subheader("Skills Extracted")
        skills = extract_skills(resume_text)
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No specific skills identified.")

        # Plot Skills Distribution
        st.subheader("Skills Distribution")
        plot_skills_distribution(skills)

        # Extracting Sections (like experience, education)
        st.subheader("Extracted Sections")
        sections = extract_sections(resume_text)
        for section, value in sections.items():
            st.write(f"{section}: {value}")

if __name__ == '__main__':
    main()
