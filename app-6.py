# --- IMPORTS (all existing imports plus new ones for STT) ---
import os
import requests
import urllib3
import pandas as pd
from tqdm import tqdm
import gradio as gr
from dotenv import load_dotenv
import json


import sys
import asyncio

from pathlib import Path
import hashlib
import time as _time

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Speech-to-Text Libraries
import openai 
import time
import nest_asyncio
nest_asyncio.apply()

# --- 1. CONFIGURATION AND INITIALIZATION ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
api_key_google = os.environ.get("GOOGLE_API_KEY")
api_key_orator = os.environ.get("ORATOR_API_KEY")
api_key_openai = os.environ.get("OPENAI_API_KEY")

if not api_key_google:
    raise ValueError("GOOGLE_API_KEY is not set.")
if not api_key_orator:
    raise ValueError("ORATOR_API_KEY is not set.")
if not api_key_openai:
    print("WARNING: OPENAI_API_KEY not found. Speech-to-Text feature will not work.")

# --- 2. DATA PATHS AND PREPARATION ---
# Note: For your local environment, you would use:
# PDF_FILE_PATH = "D:\\RAG-projects\\ppcrag\\data\\Pakistan_Penal _Code.pdf"
# FAISS_INDEX_PATH = "data/ppc_faiss_index"
# For Colab, you would need to adjust these paths.
# PDF_FILE_PATH = "data\Pakistan_Penal_Code.pdf"
# FAISS_INDEX_PATH = "data\ppc_faiss_index"
# SUMMARIES_CACHE_PATH = "data\summaries_cache.json"

PDF_INPUT = Path("data-1")       # directory containing many PDFs
FAISS_INDEX_PATH = "data-1/healthcare_faiss_index"
SUMMARIES_CACHE_PATH = "data-1/summaries_cache.json"



# --- 3. HELPER FUNCTIONS ---
# All your existing helper functions remain the same:
# - load_and_chunk_pdfs
# - summarize_text
# - build_vector_store_from_documents
# - text_to_speech

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _discover_pdf_paths(pdf_input) -> list[Path]:
    """
    Accepts either a directory (Path/str) or an iterable of files.
    Returns a list[Path] of PDFs.
    """
    if isinstance(pdf_input, (str, Path)):
        p = Path(pdf_input)
        if p.is_dir():
            return sorted([x for x in p.glob("**/*.pdf") if x.is_file()])
        elif p.is_file() and p.suffix.lower() == ".pdf":
            return [p]
        else:
            raise FileNotFoundError(f"No PDF(s) found at: {p}")
    # Iterable of paths/strings
    paths = [Path(x) for x in pdf_input]
    pdfs = [p for p in paths if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise FileNotFoundError("No valid .pdf files found in provided list.")
    return sorted(pdfs)

def _doc_fingerprint(path: Path) -> str:
    """
    Create a stable ID per source file (path + modified time + size).
    Useful for caching summaries across many PDFs.
    """
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(paths):
    """
    Accept a directory path, a single PDF path (str/Path), or a list of PDF paths.
    Return a flat list of chunked LangChain Documents with source metadata.
    Encrypted PDFs (without a password) are skipped.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    pdf_paths: list[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if p.is_dir():
            pdf_paths.extend(sorted(p.glob("**/*.pdf")))
        else:
            if p.suffix.lower() != ".pdf":
                raise ValueError(f"File is not a PDF: {p}")
            pdf_paths.append(p)

    if not pdf_paths:
        raise ValueError("No PDF files found.")

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()  # will raise if encrypted and crypto backend missing or password required
        except Exception as e:
            # Common encrypted/unsupported cases → skip
            print(f"⚠️ Skipping {file_path.name}: {e}")
            continue

        chunks = text_splitter.split_documents(docs)
        for d in chunks:
            d.metadata = d.metadata or {}
            d.metadata["source_file"] = file_path.name
            d.metadata["source_path"] = str(file_path.resolve())
        all_chunks.extend(chunks)

    if not all_chunks:
        #raise ValueError("No readable (non-encrypted) PDFs produced any chunks.")
        return[]
    return all_chunks






def summarize_text(llm, text, lang="en"):
    prompt = f"Summarize the following healthcare  text in {lang.capitalize()}. Keep it concise (15-150 words)."
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=text[:2000])
        ])
        return response.content.strip()
    except Exception as e:
        return f"Summary failed due to API error: {str(e)}"
    

def build_vector_store_from_documents(llm, faiss_path, embeddings_api_key):
    """
    Build (or load) a FAISS store from N PDFs with per-file summary caching.
    - If FAISS exists, load it.
    - Else, read or build `summaries_cache.json`, only summarizing new/changed files.
    """
    # 1) If FAISS already exists: just load it
    if os.path.exists(faiss_path):
        print("FAISS index already exists. Loading it...")
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", api_key=embeddings_api_key)
        return FAISS.load_local(faiss_path, embeddings=embeddings, allow_dangerous_deserialization=True)

    # 2) Discover inputs & prepare cache
    pdf_paths = _discover_pdf_paths(PDF_INPUT)
    if not pdf_paths:
        raise ValueError("No PDFs to build the index from.")

    cache = {}
    if os.path.exists(SUMMARIES_CACHE_PATH):
        try:
            with open(SUMMARIES_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # cache keyed by source_fp -> list of chunk dicts
                cache = {k: v for k, v in data.items()}
        except Exception:
            cache = {}

    summarized_texts = []
    metadatas = []
    updated_cache = dict(cache)  # we’ll mutate and write back

    # 3) Summarize (only missing/changed files)
    for path in tqdm(pdf_paths, desc="Preparing PDFs"):
        fp = _doc_fingerprint(path)
        if fp in cache:
            # use cached chunk summaries for this file
            for item in cache[fp]:
                summarized_texts.append(item["text"])
                metadatas.append(item["metadata"])
            continue

        # fresh summarize for this file

        print(f"Summarizing: {path.name}")

        #file_chunks = [c for c in load_and_chunk_pdfs([path])]
        #file_chunks = load_and_chunk_pdfs(path)
        try:
            file_chunks = load_and_chunk_pdfs(path)
        except Exception as e:
            print(f"⚠️ Skipping {path.name} due to loader error: {e}")
            continue

        if not file_chunks:
            print(f"⚠️ Skipping {path.name}: no extractable text (likely scanned/NO TEXT).")
            continue
    


        file_items = []
        for doc in tqdm(file_chunks, desc=f"Summarizing chunks of {path.name}", leave=False):
            summary_en = summarize_text(llm, doc.page_content, "en")
            summary_ur = summarize_text(llm, doc.page_content, "ur")
            combined_summary = (summary_en or "").strip() + " " + (summary_ur or "").strip()

            # enrich metadata
            doc.metadata = doc.metadata or {}
            doc.metadata["summary_en"] = summary_en
            doc.metadata["summary_ur"] = summary_ur

            summarized_texts.append(combined_summary)
            metadatas.append(doc.metadata)
            file_items.append({"text": combined_summary, "metadata": doc.metadata})

        updated_cache[fp] = file_items

    # 4) Persist updated cache (mapping: source_fp -> list[chunk_summaries])
    with open(SUMMARIES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_cache, f, indent=2)
    print(f"Summaries cache saved to {SUMMARIES_CACHE_PATH}.")

    # 5) Build & save FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", api_key=embeddings_api_key)
    vectorstore = FAISS.from_texts(texts=summarized_texts, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(faiss_path)
    print(f"FAISS index built with {vectorstore.index.ntotal} entries and saved to {faiss_path}.")
    return vectorstore



def speech_to_text(audio_path):
    """Transcribe audio with OpenAI Whisper."""
    if not api_key_openai:
        return "", "❌ OpenAI API Key not set for transcription."
    if not audio_path:
        return "", "❌ No audio input received."

    from openai import OpenAI
    client = OpenAI(api_key=api_key_openai)
    try:
        with open(audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        # tr is a plain text string when response_format="text"
        return (tr if isinstance(tr, str) else tr.text), "✅ Transcription successful!"
    except Exception as e:
        return "", f"❌ Transcription failed: {e}"


def text_to_speech(text):
    if not text:
        return None, "❌ No text to convert."
    if not api_key_orator:
        return None, "❌ Orator API Key not set."
    
    url = "https://api.upliftai.org/v1/synthesis/text-to-speech"
    headers = {
        "Authorization": f"Bearer {api_key_orator}",
        "Content-Type": "application/json"
    }
    data = {
        "voiceId": "v_8eelc901",
        "text": text,
        "outputFormat": "MP3_22050_128"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        output_path = "output.mp3"
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return output_path, "✅ Audio generated!"
    except requests.exceptions.RequestException as e:
        return None, f"❌ Error from Orator API: {e}"


async def voice_rag_pipeline(audio_file, text_query):
    # Initialize the LLM and retriever
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key_google)

    # Cache FAISS vectorstore for speed
    global _vectorstore
    if '_vectorstore' not in globals() or _vectorstore is None:
        _vectorstore = build_vector_store_from_documents(llm, FAISS_INDEX_PATH, api_key_google)

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

    # 1. Choose between text and audio input
    if text_query and text_query.strip():
        transcription_text = text_query.strip()
        stt_status = "✅ Using text input."
    elif audio_file:
        transcription_text, stt_status = speech_to_text(audio_file)
    else:
        return "", "", None, "❌ No input provided. Please enter text or speak.", ""

    if "❌" in stt_status:
        return transcription_text, "", None, stt_status, ""
        
    # --- MODIFIED RAG CHAIN ---
    # Create the prompt template
    prompt = PromptTemplate.from_template("""
    You are a bilingual chatbot for healthcare. Answer the question based on the following context from the attached documents.
    {context}
    Question: {question}
    Answer concisely.
    """)

    # Build the RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 2. RAG pipeline - Invoke the chain to get the answer
    try:
        rag_response_text = await rag_chain.ainvoke(transcription_text)
        rag_response_status = "✅ RAG search successful!"
    except Exception as e:
        return transcription_text, "", None, stt_status, f"❌ RAG chain failed: {e}"
    
    # --- Retrieve sources separately ---
    # We need to get the source documents to cite them.
    # The simplest way is to run the retriever again.
    source_docs = await retriever.ainvoke(transcription_text)

    # 3. Format the response with source attribution
    sources_text = ""
    if source_docs:
        # Use a set to store unique citations to avoid duplicates
        seen_citations = set()
        for doc in source_docs:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            page_number = int(doc.metadata.get("page", 0)) + 1 # Page numbers are 0-indexed, so add 1
            citation = f"Source: {source_file}, Page: {page_number}"
            if citation not in seen_citations:
                sources_text += f"\n- {citation}"
                seen_citations.add(citation)

    final_text_response = f"{rag_response_text}\n\n**Sources:**{sources_text}"

    # 4. TTS pipeline
    if not rag_response_text or len(rag_response_text.strip()) < 5:
        tts_status = "⚠️ No meaningful text to convert."
        audio_path = None
    else:
        # We'll convert only the generated answer, not the sources list
        audio_path, tts_status = text_to_speech(rag_response_text)

    # 5. Return all outputs to the UI
    return transcription_text, final_text_response, audio_path, rag_response_status, tts_status

    # 4. TTS pipeline (existing code)
    if not rag_response_text or len(rag_response_text.strip()) < 5:
        tts_status = "⚠️ No meaningful text to convert."
        audio_path = None
    else:
        # We'll convert only the generated answer, not the sources list
        audio_path, tts_status = text_to_speech(rag_response_text)

    # 5. Return all outputs to the UI
    return transcription_text, final_text_response, audio_path, rag_response_status, tts_status

# ... (rest of the code remains the same





    # """
    # Combines STT, RAG, and TTS into a single orchestrated pipeline.
    # Supports both audio and text input.
    # """
    # # Initialize the LLM and retriever
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key_google)

    # # Cache FAISS vectorstore for speed
    # global _vectorstore
    # if '_vectorstore' not in globals() or _vectorstore is None:
    #     _vectorstore = build_vector_store_from_documents(llm, FAISS_INDEX_PATH, api_key_google)

    # retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

    # prompt = PromptTemplate.from_template("""
    # Answer the question based on the following context from the attached documents:
    # {context}
    # Question: {question}
    # Answer concisely:
    # """)
    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # # 1. Choose between text and audio input
    # if text_query and text_query.strip():
    #     transcription_text = text_query.strip()
    #     stt_status = "✅ Using text input."
    # elif audio_file:
    #     transcription_text, stt_status = speech_to_text(audio_file)
    # else:
    #     return "", "", None, "❌ No input provided. Please enter text or speak.", ""

    # if "❌" in stt_status:
    #     return transcription_text, "", None, stt_status, ""

    # # 2. RAG pipeline
    # try:
    #     rag_response_text = await rag_chain.ainvoke(transcription_text)
    #     rag_response_status = "✅ RAG search successful!"
    # except Exception as e:
    #     return transcription_text, "", None, stt_status, f"❌ RAG chain failed: {e}"

    # # 3. TTS pipeline
    # if not rag_response_text or len(rag_response_text.strip()) < 5:
    #     return transcription_text, rag_response_text, None, rag_response_status, "⚠️ No meaningful text to convert."

    # audio_path, tts_status = text_to_speech(rag_response_text)

    # # 4. Return all outputs to the UI
    # return transcription_text, rag_response_text, audio_path, rag_response_status, tts_status



def get_vectorstore(llm):
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vector_store_from_documents(llm, FAISS_INDEX_PATH, api_key_google)
    return _vectorstore




# --- REVISED GRADIO UI ---
# The UI needs to be able to accept a null audio file path if the user doesn't speak.
# --- REVISED GRADIO UI ---
def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚖️ Voice and Text-Enabled Diabetes Care chatbot.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    label="Speak Your Question", 
                    type="filepath",
                    interactive=True
                )
                text_input = gr.Textbox(
                    label="Or Type Your Question Here",
                    placeholder="E.g., What is 'culpable homicide'?",
                    lines=2
                )
                submit_button = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                transcription_output = gr.Textbox(label="Your Query", interactive=False)
                rag_response_text = gr.Textbox(label="Text Answer", interactive=False)
                audio_output = gr.Audio(label="Spoken Answer", type="filepath", interactive=False, autoplay=True)
                status_stt = gr.Textbox(label="Status: Transcription/Text", interactive=False)
                status_tts = gr.Textbox(label="Status: Audio Generation", interactive=False)
        
        submit_button.click(
            fn=voice_rag_pipeline,
            inputs=[audio_input, text_input],
            outputs=[transcription_output, rag_response_text, audio_output, status_stt, status_tts]
        )

    demo.launch()

if __name__ == "__main__":
    main()
