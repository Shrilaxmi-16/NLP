import streamlit as st
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load the spacy NLP model
nlp = spacy.load('en_core_web_sm')

# Function to extract named entities and visualize them
def analyze_resume(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_skills(text):
    # A very basic regex pattern for identifying "skills"
    skills_pattern = r"(\bPython\b|\bMachine Learning\b|\bData Science\b|\bSQL\b|\bJava\b|\bNLP\b)"
    return re.findall(skills_pattern, text, re.IGNORECASE)

def plot_entity_distribution(entities):
    entity_labels = [label for text, label in entities]
    entity_counter = Counter(entity_labels)
    
    labels, values = zip(*entity_counter.items())
    plt.bar(labels, values)
    plt.title("Entity Distribution")
    plt.xlabel("Entity Types")
    plt.ylabel("Count")
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("Smart Resume Analyzer")

    st.subheader("Upload a resume for analysis")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode('utf-8')
        st.text_area("Resume Content", resume_text, height=200)

        # Extract named entities from the resume
        st.subheader("Entity Analysis")
        entities = analyze_resume(resume_text)

        # Display extracted entities
        st.write("Extracted Entities:")
        for entity in entities:
            st.write(f"{entity[0]}: {entity[1]}")

        # Extract and display skills
        st.subheader("Skills Extracted")
        skills = extract_skills(resume_text)
        if skills:
            st.write(f"Skills identified: {', '.join(set(skills))}")
        else:
            st.write("No specific skills identified.")

        # Visualize the entity distribution
        st.subheader("Entity Distribution")
        plot_entity_distribution(entities)

if __name__ == '__main__':
    main()
