import os
import json
import time
import gc
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import streamlit as st
import pymupdf
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configuration
@dataclass
class Config:
    """Centralized configuration for the application."""
    # Model configurations
    MODEL_OPTIONS = {
        "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "gemma_2b": "google/gemma-1.1-2b-it",
        "gemma_7b": "google/gemma-7b-it",
        "gemma2_9b": "google/gemma-2-9b-it"
    }
    
    EMBEDDING_MODELS = {
        "multilingual": "intfloat/multilingual-e5-small",
        "miniLM": "all-MiniLM-L6-v2",
        "bge_en": "BAAI/bge-base-en-v1.5",
        "bge_multi": "BAAI/bge-m3"
    }
    
    # Application flags
    use_llama3: bool = False
    use_gemma: bool = True
    use_rag: bool = True
    use_faiss: bool = False
    use_chroma: bool = False
    
    # Model parameters
    chunk_size: int = 800
    chunk_overlap: int = 50
    max_tokens: int = 512
    
    @property
    def selected_model(self) -> str:
        """Determine which model to use based on configuration."""
        if self.use_llama3:
            return self.MODEL_OPTIONS["llama3_8b"]
        elif self.use_gemma:
            return self.MODEL_OPTIONS["gemma_2b"]
        return self.MODEL_OPTIONS["gemma_2b"]
    
    @property
    def selected_embedding(self) -> str:
        """Get the selected embedding model."""
        return self.EMBEDDING_MODELS["bge_en"]

class DocumentProcessor:
    """Handles PDF document processing and text extraction."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF file using pymupdf."""
        try:
            doc = pymupdf.open(pdf_path)
            return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)

class VectorStoreManager:
    """Manages vector store creation and operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.selected_embedding,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    
    def create_vector_store(self, chunks: List[str]) -> FAISS:
        """Create and save a FAISS vector store."""
        return FAISS.from_texts(chunks, embedding=self.embeddings)
    
    def load_vector_store(self, path: str) -> FAISS:
        """Load an existing FAISS vector store."""
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

class LLMService:
    """Handles LLM initialization and text generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = self._initialize_model()
        
    def _initialize_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize the language model with appropriate configuration."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if torch.cuda.is_available() else None
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.selected_model,
            device_map="auto",
            quantization_config=bnb_config,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.selected_model,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        return model, tokenizer
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response from the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class RAGSystem:
    """Implements the Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_service = LLMService(config)
        self.vector_store_manager = VectorStoreManager(config)
        
    def answer_question(self, question: str, context: str) -> str:
        """Generate an answer using RAG."""
        prompt_template = """
        Answer the question based on the context below. If you don't know the answer, 
        say "I don't know".

        Context: {context}

        Question: {question}
        Answer:"""
        
        prompt = prompt_template.format(context=context, question=question)
        return self.llm_service.generate_response(prompt)

def main():
    config = Config()
    load_dotenv()
    
    # Streamlit UI
    st.set_page_config("Document QA System", layout="wide")
    st.title("Document Question Answering System")
    
    with st.sidebar:
        st.header("Configuration")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        # Process documents and create vector store
                        processor = DocumentProcessor()
                        all_text = ""
                        
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            all_text += processor.extract_text_from_pdf(tmp_path)
                            os.unlink(tmp_path)
                        
                        chunks = processor.chunk_text(
                            all_text,
                            config.chunk_size,
                            config.chunk_overlap
                        )
                        
                        vector_store = VectorStoreManager(config).create_vector_store(chunks)
                        vector_store.save_local("faiss_index")
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file")
    
    question = st.text_input("Ask a question about your documents")
    if question and st.button("Get Answer"):
        try:
            # Load vector store and perform RAG
            vector_store = VectorStoreManager(config).load_vector_store("faiss_index")
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            # Generate answer
            rag_system = RAGSystem(config)
            answer = rag_system.answer_question(question, context)
            
            st.subheader("Answer")
            st.write(answer)
            
            st.subheader("Relevant Context")
            st.write(context)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
