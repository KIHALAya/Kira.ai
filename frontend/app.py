import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Adjust if needed

st.set_page_config(page_title="AI Research Paper Agent", layout="wide")

st.title("🧠 Agentic AI System for Academic Paper Generation")

st.markdown("Upload your research documents and provide a prompt to generate an AI-driven academic paper.")

# --- User Input Section ---
with st.form("research_form"):
    prompt = st.text_area("📝 Research Prompt", placeholder="Describe the topic or objective of your research...")
    uploaded_files = st.file_uploader(
        "📄 Upload supporting documents (PDF, DOCX, TXT, JPG, PNG)",
        type=["pdf", "docx", "txt", "jpg", "png"],
        accept_multiple_files=True
    )
    submitted = st.form_submit_button("🚀 Start Ingestion")

if submitted:
    if not prompt and not uploaded_files:
        st.warning("Please provide a prompt or upload at least one file.")
    else:
        with st.spinner("Uploading and processing your data..."):
            files = []
            for file in uploaded_files:
                files.append(("files", (file.name, file.getvalue(), file.type)))
            data = {"prompt": prompt}
            
            try:
                response = requests.post(f"{BACKEND_URL}/ingest/", data=data, files=files)
                if response.status_code == 200:
                    st.success("✅ Data successfully ingested and embeddings created!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"Backend not reachable: {e}")
