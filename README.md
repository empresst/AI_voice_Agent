# AI Voice Sales Agent

A CPU-efficient AI Voice Sales Agent that simulates real-time sales calls for the **AI Mastery Bootcamp**. Built with open-source, lightweight models to work without a GPU, the system integrates speech processing, dynamic conversation flows, and sentiment analysis using FastAPI and LangChain, with a Streamlit UI for interactive testing.

---

##  Project Context

This project simulates a voice-driven sales agent capable of:
- Conversational pitching
- Objection handling
- Lead qualification

> **Constraint:** Must run entirely on CPU using open-source models with FastAPI and LangChain.

---

##  Architecture Overview

###  Components

- **FastAPI Server**: Handles `/start-call`, `/respond/{call_id}`, and `/conversation/{call_id}` endpoints.
- **Conversation Engine**: State-machine-based flow (Intro → Qualification → Pitch → Objection → Closing).
- **LLM Pipeline**: DistilGPT-2 via LangChain for generating dynamic, structured responses.
- **Voice Interface**:
  - **STT**: Whisper-Tiny for speech recognition.
  - **TTS**: gTTS for converting text to speech.
- **Storage**: ChromaDB for embedding-based conversation history.
- **Frontend**: Streamlit UI with real-time visualization of:
  - Sentiment analysis
  - Purchase intent
  - Conversation flow

---

##  Tech Stack

| Purpose             | Tool                          |
|---------------------|-------------------------------|
| Backend Framework   | FastAPI                        |
| AI Framework        | LangChain                      |
| Language Model      | DistilGPT-2 (Hugging Face)     |
| Text-to-Speech      | gTTS                           |
| Speech-to-Text      | Whisper-Tiny                   |
| Sentiment Analysis  | DistilBERT                     |
| Storage             | ChromaDB                       |
| Frontend UI         | Streamlit                      |
| Caching             | Cachetools (TTLCache)          |

---

##  Data Flow

1. Call is initiated via Streamlit or API.
2. FastAPI assigns a unique `call_id` and sends an initial greeting.
3. Customer input is received via audio or text.
4. STT (Whisper-Tiny) transcribes audio if applicable.
5. Input is passed to LangChain with prompt engineering.
6. Sentiment analysis runs on input to gauge interest.
7. LLM generates response using DistilGPT-2.
8. TTS (gTTS) converts response to audio.
9. All interactions stored in ChromaDB.
10. Streamlit updates charts and logs conversation flow.

---

##  Key Architecture Decisions & Trade-offs

- **LLM Choice**: DistilGPT-2 chosen over larger models like LLaMA to meet CPU constraint.
- **Voice Quality**: gTTS used despite robotic tone to ensure low resource usage.
- **Storage**: ChromaDB preferred over FAISS for embedding support and ease of use.
- **Telephony**: Real-time calling was mocked using the Streamlit interface for feasibility.

---

##  Implementation Timeline

| Day | Task |
|-----|------|
| 1–2 | Researched models, designed architecture, conversation stages |
| 3–4 | Built FastAPI, LangChain integration, gTTS/Whisper pipeline |
| 5   | Added DistilBERT for sentiment, polished UI, added analytics & documentation |

---

##  Creative Features

- **Streamlit Dashboard**: Visualizes conversation, sentiment, and sales intent.
- **Sentiment Charts**: Gauge customer interest live using DistilBERT.
- **A/B Pitch Testing**: Compares variants of the sales pitch.
- **Caching**: Optimized LLM calls using `TTLCache`.
- **Fallback Handling**: Default responses on STT/LLM failures.

---

##  Performance Optimizations

- Used ultra-light models (DistilGPT-2, Whisper-Tiny).
- Cached common LLM prompts to reduce inference time.
- Limited audio file size (<5MB) and duration (<10s).
- Async endpoints in FastAPI for scalability.

---

##  Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Low-quality LLM replies | Improved prompt design in LangChain |
| CPU-only voice processing | Used Whisper-Tiny & short audio limits |
| Lack of persistent DB | Used ChromaDB for fast, vector-based memory |
| No real telephony support | Mocked interaction with UI & audio playback |

## To run this

- pip install -r requirements.txt
- uvicorn Latestmain:app --host 0.0.0.0 --port 8000
- streamlit run Latest_Intrface.py

##  Future Improvements

- **Upgrade LLM**: Use more powerful models like Mistral if GPU becomes available.
- **Better TTS**: Try SpeechT5 or Bark for natural voice quality.
- **Live Analytics**: Add real-time feedback on customer sentiment.
- **Language Expansion**: Add multilingual support for broader reach.


##  Conclusion

This project delivers a well-architected, CPU-friendly AI Voice Agent using accessible tools and open-source models. Through creative use of LangChain, audio interfaces, and Streamlit, it showcases a thoughtful balance of technical depth and practical limitations — making it a standout solution for AI interaction under resource constraints.

![Alt Text](https://github.com/empresst/AI_Voice_Sales_Agent/blob/main/Screenshot%20from%202025-06-19%2020-52-11.png)


