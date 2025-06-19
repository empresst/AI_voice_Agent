# AI Voice Sales Agent – Model Selection Justification

This project implements an AI-powered Voice Sales Agent designed to run in a **CPU-only environment** using **lightweight, open-source models**. Below is a detailed breakdown of each model selected, including trade-offs, alternatives, and rationale.

---

## 1. LLM: DistilGPT-2 (Hugging Face)

- **Why Chosen**:  
  DistilGPT-2 (~300MB) is a distilled version of GPT-2. It offers reduced memory and computational demands, making it ideal for CPU-based environments. It supports dynamic text generation needed for sales dialogue.

- **Alternatives Considered**:
  - **LLaMA / Mixtral** – Too large for CPU (~4GB+)
  - **GPT-3.5 (OpenAI)** – Requires a paid API and is not open-source

- **Trade-offs**:
  - Fast inference, low memory footprint, open-source  
  -  Simpler outputs compared to larger models

- **Justification**:  
  Combined with **LangChain** for prompt engineering, DistilGPT-2 delivers sufficient conversational quality while respecting system limitations.

---

## 2. TTS: gTTS (Google Text-to-Speech)

- **Why Chosen**:  
  gTTS is a free, easy-to-use, cloud-based TTS solution that produces clear audio. It requires no local model hosting, which is ideal for CPU-only deployment.

- **Alternatives Considered**:
  - **SpeechT5 (Hugging Face)** – Higher memory usage (~500MB)
  - **ElevenLabs** – Commercial, not open-source

- **Trade-offs**:
  - Free, simple integration, functional output  
  - Limited customization and voice variety

- **Justification**:  
  Given the focus on **logic over voice quality**, gTTS was the best balance of simplicity and clarity for speech synthesis.

---

## 3. STT: Whisper-Tiny (Hugging Face)

- **Why Chosen**:  
  Whisper-Tiny (~100MB) is a lightweight version of OpenAI’s Whisper, delivering reasonable transcription quality with minimal CPU load.

- **Alternatives Considered**:
  - **Whisper-Base** – Larger and slower (~200MB)
  - **Google Cloud STT** – Paid API, not open-source

- **Trade-offs**:
  - Compact, open-source, acceptable transcription quality  
  - Slightly lower accuracy than larger versions

- **Justification**:  
  Ideal for short audio clips and user interactions, Whisper-Tiny keeps the system responsive and resource-efficient.

---

## 4. Sentiment Analysis: DistilBERT (Hugging Face)

- **Why Chosen**:  
  DistilBERT (~250MB) is a distilled version of BERT, fine-tuned for sentiment classification. It runs well on CPU and offers good accuracy.

- **Alternatives Considered**:
  - **BERT Base** – Higher accuracy but ~400MB
  - **VADER** – Lightweight but not nuanced enough

- **Trade-offs**:
  - Fast, balanced accuracy, supports real-time prediction  
  - Binary or ternary classification only

- **Justification**:  
  Enables real-time emotion tracking to enhance **sales intelligence** and support **Streamlit-based analytics**.

---

## 5. Vector Database: ChromaDB

- **Why Chosen**:  
  ChromaDB is a lightweight vector store that runs in memory or local file system, supporting efficient semantic search using embeddings.

- **Alternatives Considered**:
  - **FAISS** – Blazing fast but lacks persistence
  - **Pinecone** – Commercial, SaaS-based

- **Trade-offs**:
  - Simple to set up, open-source, embedding support  
  - Slightly more memory use than flat file storage

- **Justification**:  
  Offers **semantic memory** for conversation history retrieval while being easy to integrate into a low-resource pipeline.

---

##  Conclusion

The model stack — **DistilGPT-2, gTTS, Whisper-Tiny, DistilBERT, and ChromaDB** — was carefully chosen to:

- Run efficiently on CPU-only systems  
- Use free and open-source tools  
- Satisfy both functional and bonus assessment criteria  

This resource-efficient architecture demonstrates practical trade-offs and thoughtful system design under real-world constraints.