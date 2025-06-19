
import logging
import uuid
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.llms.base import LLM
from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from gtts import gTTS
import re
from cachetools import TTLCache
from pydub import AudioSegment
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, filename='sales_agent.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Voice Sales Agent", description="API for an AI-powered voice sales agent pitching AI Mastery Bootcamp.")

# ChromaDB for conversation storage
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="conversations",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {str(e)}")
    raise

# Cache for LLM responses (TTL: 1 hour)
response_cache = TTLCache(maxsize=100, ttl=3600)

# Custom LLM wrapper for distilgpt2
class DistilGPT2LLM(LLM):
    pipeline: Optional[Any] = Field(default=None)

    def __init__(self, model_name: str = "distilgpt2"):
        super().__init__()
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=60,
                truncation=True
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            raise

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        cache_key = f"{prompt[:100]}"
        if cache_key in response_cache:
            logger.info("Returning cached LLM response")
            return response_cache[cache_key]

        try:
            response = self.pipeline(
                prompt,
                num_return_sequences=1,
                max_new_tokens=60,
                truncation=True
            )[0]["generated_text"]
            result = response[len(prompt):].strip()
            if not result:
                result = "I'm sorry, I didn't understand. Could you clarify?"
            response_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return "I'm sorry, I'm having trouble responding. Can you repeat that?"

    @property
    def _llm_type(self) -> str:
        return "distilgpt2"

# Initialize models
try:
    llm = DistilGPT2LLM()
    stt = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

# Mock TTS
def mock_tts(text: str) -> bytes:
    try:
        tts = gTTS(text=text, lang="en")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        return BytesIO().getvalue()

# Mock STT
def mock_stt(audio: bytes) -> str:
    try:
        temp_file = f"temp_audio_{uuid.uuid4()}.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio)
        logger.info(f"Saved audio to {temp_file}, size: {len(audio)} bytes")
        audio_segment = AudioSegment.from_file(temp_file)
        wav_file = f"temp_audio_{uuid.uuid4()}.wav"
        audio_segment.export(wav_file, format="wav")
        result = stt(wav_file)["text"]
        os.remove(temp_file)
        os.remove(wav_file)
        logger.info(f"STT processed audio, result: {result}")
        return result if result else "I'm sorry, I didn't catch that."
    except Exception as e:
        logger.error(f"STT processing failed: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)
        return "I'm sorry, I didn't catch that."

# LangChain prompt
prompt_template = PromptTemplate(
    input_variables=["history", "customer_input", "stage", "sentiment"],
    template="""
    You are an AI Voice Sales Agent for AI Academy, pitching the AI Mastery Bootcamp (12 weeks, $299, covers Large Language Models, Computer Vision, MLOps, hands-on projects, job placement assistance, and a certificate). 
    Current stage: {stage}. Customer sentiment: {sentiment}.
    Conversation history (last 5 turns): {history}
    Customer said: {customer_input}
    Respond as if on a phone call, using a friendly, persuasive, and professional tone. Address their input directly, handle objections (e.g., price, time, relevance), and advance the conversation toward enrollment. Keep responses concise (50-60 words), natural, and engaging. If unsure, ask a clarifying question to keep the conversation flowing.
    """
)

# Create RunnableSequence
llm_chain = prompt_template | llm

# A/B testing pitches
PITCH_VARIATIONS = {
    "A": "Our AI Mastery Bootcamp is a 12-week program for only $299, teaching LLMs, Computer Vision, and MLOps with hands-on projects and job placement support. Perfect for your goals?",
    "B": "Join our 12-week AI Mastery Bootcamp for just $299! Master AI skills like LLMs and Computer Vision with projects and career support. Ready to jump in?"
}

# Conversation stages
CONVERSATION_STAGES = {
    "introduction": "Hello {name}! I'm from AI Academy, calling about our AI Mastery Bootcamp to boost your AI skills. Interested?",
    "qualification": [
        "What’s your experience level with AI or machine learning?",
        "Are you looking to switch careers or enhance your current role with AI?",
        "What specific AI skills are you hoping to gain?"
    ],
    "objection_handling": {
        "price": "I hear you on cost. Our bootcamp is $299, down from $499, with flexible payment plans. Does that work for you?",
        "time": "It’s self-paced, so you can study evenings or weekends. What’s a good learning schedule for you?",
        "relevance": "The bootcamp covers practical AI for many fields. What are your goals? I’ll show how it aligns.",
        "not_interested": "No problem! Can I ask what’s holding you back? I’d love to address any concerns."
    },
    "closing": "Awesome! Can we schedule a follow-up to enroll, or are you ready to join for $299 today?"
}

# Objection detection patterns
OBJECTION_PATTERNS = {
    "price": re.compile(r"\b(cost|price|expensive|costly|budget|money)\b", re.IGNORECASE),
    "time": re.compile(r"\b(time|busy|schedule|availability)\b", re.IGNORECASE),
    "relevance": re.compile(r"\b(relevant|fit|use|useful|need)\b", re.IGNORECASE),
    "not_interested": re.compile(r"\b(not interested|no thanks|pass)\b", re.IGNORECASE)
}

# Pydantic models
class StartCallRequest(BaseModel):
    phone_number: str
    customer_name: str
    pitch_variant: str = "A"

class StartCallResponse(BaseModel):
    call_id: str
    message: str
    first_message: str
    audio: str

class RespondRequest(BaseModel):
    message: Optional[str] = None
    audio: Optional[str] = None

class RespondResponse(BaseModel):
    reply: str
    should_end_call: bool
    sentiment: str
    audio: str
    transcription: Optional[str] = None  # Added to return transcribed text

class DebugAudioRequest(BaseModel):
    audio: str

class DebugAudioResponse(BaseModel):
    decoded_length: int
    message: str

class ConversationResponse(BaseModel):
    call_id: str
    history: List[Dict[str, str]]
    sentiment_trend: List[float]
    turn_count: int
    objection_count: int

def sanitize_input(text: str) -> str:
    return re.sub(r'[<>;]', '', text[:500])

@app.post("/start-call", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    try:
        call_id = str(uuid.uuid4())
        customer_name = sanitize_input(request.customer_name)
        if not customer_name:
            raise HTTPException(status_code=400, detail="Customer name cannot be empty")
        first_message = CONVERSATION_STAGES["introduction"].format(name=customer_name)
        
        tts_output = mock_tts(first_message)
        audio_base64 = base64.b64encode(tts_output).decode("ascii")
        
        conversation = {
            "phone_number": request.phone_number,
            "customer_name": customer_name,
            "history": [{"role": "agent", "message": first_message}],
            "stage": "qualification",
            "qualification_step": 0,
            "pitch_variant": request.pitch_variant,
            "sentiment_scores": [0.0],
            "turn_count": 1,
            "objection_count": 0
        }
        
        collection.add(
            documents=[str({"history": conversation["history"], "stage": conversation["stage"], "qualification_step": conversation["qualification_step"]})],
            metadatas=[{
                "call_id": call_id,
                "phone_number": request.phone_number,
                "customer_name": customer_name,
                "pitch_variant": request.pitch_variant,
                "turn_count": 1,
                "objection_count": 0
            }],
            ids=[call_id]
        )
        
        logger.info(f"Started call {call_id} for {customer_name}")
        return StartCallResponse(
            call_id=call_id,
            message="Call initiated successfully",
            first_message=first_message,
            audio=audio_base64
        )
    except Exception as e:
        logger.error(f"Start call failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start call: {str(e)}")

@app.post("/respond/{call_id}", response_model=RespondResponse)
async def respond(call_id: str, request: Request):
    try:
        raw_data = await request.json()
        logger.info(f"Raw request for call {call_id}:\n{json.dumps(raw_data, indent=2)}")
        validated_request = RespondRequest(**raw_data)
        message = validated_request.message
        audio = validated_request.audio
        logger.info(f"Processing request for call {call_id}: message='{message}', audio_length={len(audio or '')}")
        
        if not message and not audio:
            logger.warning(f"No input provided for call {call_id}")
            raise HTTPException(status_code=422, detail="At least one of 'message' or 'audio' must be provided")
        
        results = collection.get(ids=[call_id])
        if not results["ids"]:
            logger.warning(f"Call {call_id} not found")
            raise HTTPException(status_code=404, detail="Call not found")
        
        conversation = eval(results["documents"][0])
        metadata = results["metadatas"][0]
        customer_input = message
        
        transcription = None
        if audio and not customer_input:
            try:
                logger.info(f"Attempting to decode audio, length: {len(audio)}, first_50: {audio[:50]}, last_50: {audio[-50:]}")
                audio_bytes = base64.b64decode(audio, validate=True)
                logger.info(f"Audio decoded successfully, bytes: {len(audio_bytes)}")
                customer_input = mock_stt(audio_bytes)
                transcription = customer_input  # Store transcription for response
            except base64.binascii.Error as e:
                logger.error(f"Failed to decode audio for call {call_id}: {str(e)}")
                customer_input = message or "Audio received, but decoding failed."
                transcription = customer_input
        
        if not customer_input:
            logger.warning(f"Empty input after processing for call {call_id}")
            return RespondResponse(
                reply="I'm sorry, I didn't catch that. Can you repeat?",
                should_end_call=False,
                sentiment="NEUTRAL",
                audio=base64.b64encode(mock_tts("Can you repeat?")).decode("ascii"),
                transcription=transcription
            )
        
        sentiment_result = sentiment_analyzer(customer_input)[0]
        sentiment = sentiment_result["label"]
        sentiment_score = 1.0 if sentiment == "POSITIVE" else -1.0 if sentiment == "NEGATIVE" else 0.0
        conversation["sentiment_scores"] = conversation.get("sentiment_scores", []) + [sentiment_score]
        metadata["turn_count"] = metadata.get("turn_count", 1) + 1
        
        conversation["history"].append({"role": "customer", "message": customer_input})
        history_str = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversation["history"][-5:]])
        
        objection_type = None
        for key, pattern in OBJECTION_PATTERNS.items():
            if pattern.search(customer_input):
                objection_type = key
                metadata["objection_count"] = metadata.get("objection_count", 0) + 1
                break
        
        response = llm_chain.invoke({
            "history": history_str,
            "customer_input": customer_input,
            "stage": conversation["stage"],
            "sentiment": sentiment
        })
        
        if conversation["stage"] == "qualification":
            response = handle_qualification(conversation, customer_input, response)
        elif conversation["stage"] == "pitch":
            response = handle_pitch(conversation, customer_input, response, objection_type)
        elif conversation["stage"] == "objection_handling":
            response = handle_objection(conversation, customer_input, response, objection_type)
        else:
            response = handle_closing(conversation, customer_input, response)
        
        tts_output = mock_tts(response)
        audio_base64 = base64.b64encode(tts_output).decode("ascii")
        
        conversation["history"].append({"role": "agent", "message": response})
        collection.update(
            documents=[str({"history": conversation["history"], "stage": conversation["stage"], "qualification_step": conversation.get("qualification_step", 0), "sentiment_scores": conversation["sentiment_scores"]})],
            metadatas=[metadata],
            ids=[call_id]
        )
        
        should_end = conversation["stage"] == "closing" and any(kw in customer_input.lower() for kw in ["bye", "not interested", "enroll"])
        
        logger.info(f"Responded to call {call_id}, stage: {conversation['stage']}, qualification_step: {conversation.get('qualification_step', 0)}, sentiment: {sentiment}, objection: {objection_type}")
        return RespondResponse(
            reply=response,
            should_end_call=should_end,
            sentiment=sentiment,
            audio=audio_base64,
            transcription=transcription
        )
    except ValueError as e:
        logger.error(f"Validation error for call {call_id}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Respond failed for call {call_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")

@app.post("/debug-audio", response_model=DebugAudioResponse)
async def debug_audio(request: DebugAudioRequest):
    try:
        audio_str = request.audio
        logger.info(f"Debug audio received, length: {len(audio_str)}, first_50: {audio_str[:50]}, last_50: {audio_str[-50:]}")
        decoded = base64.b64decode(audio_str, validate=True)
        logger.info(f"Debug audio decoded, length: {len(decoded)} bytes")
        return DebugAudioResponse(
            decoded_length=len(decoded),
            message="Audio decoded successfully"
        )
    except base64.binascii.Error as e:
        logger.error(f"Debug audio failed: {str(e)}, input_first_50: {audio_str[:50]}")
        raise HTTPException(status_code=422, detail=f"Invalid base64 audio: {str(e)}")
    except Exception as e:
        logger.error(f"Debug audio failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/conversation/{call_id}", response_model=ConversationResponse)
async def get_conversation(call_id: str):
    try:
        results = collection.get(ids=[call_id])
        if not results["ids"]:
            logger.warning(f"Call {call_id} not found")
            raise HTTPException(status_code=404, detail="Call not found")
        
        conversation = eval(results["documents"][0])
        metadata = results["metadatas"][0]
        return ConversationResponse(
            call_id=call_id,
            history=conversation["history"],
            sentiment_trend=conversation.get("sentiment_scores", [0.0]),
            turn_count=metadata.get("turn_count", 1),
            objection_count=metadata.get("objection_count", 0)
        )
    except Exception as e:
        logger.error(f"Get conversation failed for call {call_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

def handle_qualification(conversation: Dict, customer_input: str, llm_response: str) -> str:
    step = conversation.get("qualification_step", 0)
    logger.info(f"Handling qualification, current step: {step}, input: {customer_input}")
    if step < len(CONVERSATION_STAGES["qualification"]):
        conversation["qualification_step"] = step + 1
        response = CONVERSATION_STAGES["qualification"][step]
        logger.info(f"Advancing to qualification step {step + 1}, response: {response}")
        return response
    else:
        conversation["stage"] = "pitch"
        conversation["qualification_step"] = 0
        response = PITCH_VARIATIONS[conversation.get("pitch_variant", "A")]
        logger.info(f"Transitioning to pitch stage, response: {response}")
        return response

def handle_pitch(conversation: Dict, customer_input: str, llm_response: str, objection_type: str) -> str:
    customer_input_lower = customer_input.lower()
    logger.info(f"Handling pitch, input: {customer_input}, objection: {objection_type}")
    if objection_type:
        conversation["stage"] = "objection_handling"
        conversation["objection_type"] = objection_type
        response = CONVERSATION_STAGES["objection_handling"].get(objection_type, llm_response)
        logger.info(f"Detected objection, transitioning to objection_handling, response: {response}")
        return response
    elif any(keyword in customer_input_lower for keyword in ["interested", "sounds good"]):
        conversation["stage"] = "closing"
        response = CONVERSATION_STAGES["closing"]
        logger.info(f"Positive response, transitioning to closing, response: {response}")
        return response
    logger.info(f"Continuing pitch, response: {llm_response}")
    return llm_response

def handle_objection(conversation: Dict, customer_input: str, llm_response: str, objection_type: str) -> str:
    logger.info(f"Handling objection, input: {customer_input}, objection: {objection_type}")
    if any(keyword in customer_input.lower() for keyword in ["okay", "interested", "sign up"]):
        conversation["stage"] = "closing"
        response = CONVERSATION_STAGES["closing"]
        logger.info(f"Objection resolved, transitioning to closing, response: {response}")
        return response
    response = CONVERSATION_STAGES["objection_handling"].get(objection_type, llm_response)
    logger.info(f"Continuing objection handling, response: {response}")
    return response

def handle_closing(conversation: Dict, customer_input: str, llm_response: str) -> str:
    logger.info(f"Handling closing, input: {customer_input}")
    response = CONVERSATION_STAGES["closing"]
    logger.info(f"Closing response: {response}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
