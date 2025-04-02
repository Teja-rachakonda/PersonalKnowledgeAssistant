import streamlit as st
from utils import process_document, answer_question

# Streamlit UI
st.title("Personal Knowledge Assistant")
st.write("Upload a PDF or text file and ask questions for free!")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    file_path = f"uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded!")
    
    # Process the document (runs locally)
    process_document(file_path)
    st.write("Document ready. Ask away!")

# Question input
query = st.text_input("Ask a question:")
if query:
    response = answer_question(query)
    st.write("**Answer:**", response)

# Reset button
if st.button("Reset"):
    st.session_state.clear()
    st.write("Memory cleared!")