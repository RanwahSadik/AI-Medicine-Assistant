import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

import os
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import pyttsx3
import easyocr
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv

from openai import OpenAI  # ✅ New OpenAI SDK

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# TTS
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# FDA Search
def search_openfda(medicine_name):
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medicine_name}&limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'results' in data:
            result = data['results'][0]
            brand_name = result['openfda'].get('brand_name', ['Unknown'])[0]
            manufacturer = result['openfda'].get('manufacturer_name', ['Unknown'])[0]
            usage = result.get('indications_and_usage', ['No usage'])[0][:200] + "..."
            dosage = result.get('dosage_and_administration', ['No dosage'])[0][:200] + "..."
            warnings = result.get('warnings', ['No warnings'])[0][:200] + "..."
            return f"""
**Drug Info:** {brand_name} by {manufacturer}  
**Usage:** {usage}  
**Dosage:** {dosage}  
**Warnings:** {warnings}
"""
        return "No information found in OpenFDA."
    except Exception as e:
        return f"Error: {e}"

# OCR for image
def extract_drug_name(image):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))
    if not results:
        return None
    texts = [(text, abs(bbox[3][1] - bbox[0][1])) for bbox, text, _ in results]
    largest_text = max(texts, key=lambda x: x[1])[0]
    return largest_text.lower().strip()

# Speech recognition
def recognize_speech():
    fs = 16000
    seconds = 5
    st.info("🎙️ Recording for 5 seconds...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, fs, (audio * 32767).astype(np.int16))
        with open(f.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text

# Chat using OpenAI 1.0+
def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful medicine assistant. Keep responses brief and to the point."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# LangChain setup
llm = ChatOpenAI(temperature=0, api_key=api_key, model_name="gpt-3.5-turbo-1106")
openfda_tool = Tool(
    name="search_openfda",
    func=search_openfda,
    description="Search for information about a medicine by its name."
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)
agent = initialize_agent(
    tools=[openfda_tool],
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    max_iterations=8,
    verbose=False
)

# PDF processing
def process_pdf(file):
    reader = PdfReader(file)
    max_pages = min(5, len(reader.pages))
    raw_text = "".join(reader.pages[i].extract_text() or "" for i in range(max_pages))
    texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(raw_text)
    texts = texts[:10]
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(texts, embeddings)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        chain_type="stuff"
    )

# Streamlit UI
st.set_page_config(page_title="Medicine Chatbot", layout="centered")
st.title("💊 AI Medicine Assistant Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about medicine..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if "what is" in prompt.lower() or "tell me about" in prompt.lower():
            med_name = prompt.lower().replace("what is", "").replace("tell me about", "").strip()
            response = agent.run(med_name)
        elif st.session_state.qa_chain:
            response = st.session_state.qa_chain.run(prompt)
        else:
            response = chat_with_gpt(prompt)
    except Exception as e:
        response = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Reset
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.qa_chain = None
    st.rerun()

# Voice input
if st.button("🎤 Speak your question"):
    try:
        text = recognize_speech()
        st.success(f"Recognized: {text}")
        st.session_state.messages.append({"role": "user", "content": text})
        response = agent.run(text)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Image upload
image_file = st.file_uploader("📷 Upload a medicine image", type=["jpg", "jpeg", "png"])
if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    try:
        med_name = extract_drug_name(image)
        if med_name:
            st.success(f"Detected Drug Name: {med_name}")
            response = agent.run(med_name)
            st.session_state.messages.append({"role": "user", "content": med_name})
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("user"):
                st.markdown(med_name)
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.warning("Could not detect a drug name in the image.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# PDF upload
pdf_file = st.file_uploader("📄 Upload a medicine-related PDF", type=["pdf"])
if pdf_file:
    try:
        st.session_state.qa_chain = process_pdf(pdf_file)
        st.success("PDF processed! You can now ask questions based on its content.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
