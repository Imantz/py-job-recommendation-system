import pdfplumber
import chromadb
import json
import gradio as gr
from sentence_transformers import SentenceTransformer

# Configurations
collection_name = "jobs"
model_name = "all-MiniLM-L6-v2"
jobs_json = "job_postings.json"

# Create a new ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name=collection_name)

# Load sentence transformer model
model = SentenceTransformer(model_name)

# Load job postings data
with open(jobs_json, "r") as f:
    job_postings = json.load(f)


def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def generate_embeddings(texts):
    """Generate embeddings using SentenceTransformer."""
    return model.encode(texts).tolist()  # Ensure output is a list


def store_embeddings_in_chroma_db():
    """Store job descriptions as embeddings in ChromaDB with correct job IDs."""
    ids = [str(job["job_id"]) for job in job_postings]
    metadatas = [{"job_title": job["job_title"]} for job in job_postings]

    job_texts = [
        f"{job['job_description']}\n\nResponsibilities:\n- " +
        "\n- ".join(job["job_responsibilities"]) +
        "\n\nPreferred Qualifications:\n- " +
        "\n- ".join(job["preferred_qualifications"])
        for job in job_postings
    ]

    job_embeddings = generate_embeddings(job_texts)

    collection.add(
        ids=ids,
        documents=job_texts,
        embeddings=job_embeddings,
        metadatas=metadatas,
    )


# Ensure job postings are stored in ChromaDB
store_embeddings_in_chroma_db()


def find_matching_jobs(resume_pdf, threshold=1):
    """Process resume, generate embedding, and find matching jobs in ChromaDB."""
    # Extract text from uploaded resume
    resume_text = extract_text_from_pdf(resume_pdf)
    if not resume_text:
        return "No text found in the resume. Please upload a valid PDF."

    # Generate embedding for resume
    resume_embedding = generate_embeddings([resume_text])[0]  # Extract first (and only) vector

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[resume_embedding],
        n_results=3,  # Get top 3 matches
    )

    # Process results
    output = ""
    if results["ids"]:
        for i in range(len(results["ids"][0])):
            job_id = results["ids"][0][i]
            job_title = results["metadatas"][0][i]["job_title"]
            job_description = results["documents"][0][i]
            similarity_score = results["distances"][0][i]  # Lower is better

            if similarity_score < threshold:  # Only include jobs within threshold
                output += (
                    f"\n**Job ID: {job_id}**\n"
                    f"**Title: {job_title}**\n"
                    f"**Similarity Score: {similarity_score:.4f}**\n"
                    f"{job_description}\n\n"
                )

    return output if output else "No matching job found with sufficient similarity."

    return output


# Gradio UI
iface = gr.Interface(
    fn=find_matching_jobs,
    inputs=gr.File(label="Upload Resume (PDF)"),
    outputs="markdown",
    title="AI Job Matcher",
    description="Upload your resume, and the AI will match it with the best job postings."
)

# Launch Gradio UI
iface.launch()
