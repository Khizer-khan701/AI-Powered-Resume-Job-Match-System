import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_resume(file_path, job_description):
    """Load, split, embed and retrieve relevant context from resume vs job description"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Keep only source metadata
    filtered_docs = [
        d.__class__(page_content=d.page_content, metadata={"source": d.metadata.get("source")})
        for d in documents
    ]

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(filtered_docs)

    # Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # QA Chain
    model = ChatOpenAI()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )

    # 1. Compare Resume to Job
    compare_query = f"Job Description: {job_description}. Compare the candidate’s resume with this job."
    compare_result = qa_chain.invoke({"question": compare_query})

    # 2. Resume Summary
    summary_query = "Summarize this resume in 4–6 sentences, focusing on education, skills, and experience."
    summary_result = qa_chain.invoke({"question": summary_query})

    # 3. Job Fit Points
    fit_query = f"Based on this resume and the job description: {job_description}, list bullet points of strengths and weaknesses for this candidate."
    fit_result = qa_chain.invoke({"question": fit_query})

    return {
        "analysis": compare_result["answer"],
        "sources": compare_result.get("sources", ""),
        "summary": summary_result["answer"],
        "fit_points": fit_result["answer"]
    }

