import streamlit as st
import requests

# Set the URL for your FastAPI endpoint
API_URL = "http://127.0.0.1:8000/search_document/"  

# Set background color to white and text color to black using custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: white;  /* Set background color to white */
    }
    .stApp {
        background-color: white;  /* Ensure the main app background is white */
    }
    .stMarkdown, .stText, .stSubheader, .stHeader, .stButton, .stExpander {
        color: black;  /* Set all text elements to black */
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the Manchester United logo
logo_url = "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg"
st.image(logo_url, width=150)

# Set the title with Manchester United theme
st.markdown(
    """
    <h1 style='text-align: center; color: #e10c2e; font-family: sans-serif;'>Manchester United: Document Retrieval System</h1>
    """, unsafe_allow_html=True
)

# Input for the search query
query_text = st.text_input("Enter your search query:")

# Button for triggering the search
if st.button("Search"):
    if query_text:
        payload = {"query_text": query_text}
        
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            results = response.json().get("documents", [])
            if results:
                st.subheader("Search Results")

                for result in results:
                    with st.expander(f"{result['rank']}. {result['document_name']}", expanded=True):
                        st.write(f"**Document Index:** {result['document_index']}")
                        st.write(f"**Similarity Score:** {result['similarity_score']:.4f}")
                        
            else:
                st.warning("No documents found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a query.")
