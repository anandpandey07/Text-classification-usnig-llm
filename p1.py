import streamlit as st
from transformers import pipeline
import PyPDF2


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


labels = ["positive", "negative", "risk", "opportunity"]

st.title('Text Classifier')
st.write('This app classifies the extracted text from a PDF')


uploaded_pdf = st.file_uploader("Upload your PDF file", type="pdf")


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


if uploaded_pdf:
    with st.spinner('Extracting text from PDF and classifying...'):
        pdf_text = extract_text_from_pdf(uploaded_pdf)

        if pdf_text.strip():   
            
            result = classifier(pdf_text, candidate_labels=labels)
            
            
            st.subheader('Extracted Text:')
            st.text_area('PDF Content:', pdf_text, height=300)

            st.subheader('Classification Result:')
            st.write(f"Predicted Labels: {result['labels']}")
            st.write(f"Scores: {result['scores']}")
            
            for label, score in zip(result['labels'], result['scores']):
                st.write(f"Label: {label} - Score: {score:.4f}")
        else:
            st.warning("The uploaded PDF does not contain any readable text.")
