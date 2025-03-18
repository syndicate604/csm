from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ValidationError
import torch
import torchaudio
import io
import sys
import os
import logging
import traceback
import subprocess
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import load_csm_1b

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU mode
torch.set_grad_enabled(False)
device = "cpu"
logger.info(f"Using device: {device}")

app = FastAPI()

# Configure CORS
origins = ["http://localhost:5174"]

@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = origins[0]
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = origins[0]
    return response

class ChatRequest(BaseModel):
    message: str
    speaker: int = 0
    temperature: float = 0.9
    max_audio_length_ms: int = 10000

# Global generator instance
generator = None

def get_generator():
    global generator
    if generator is None:
        logger.info(f"Loading model in CPU mode")
        
        model_path = os.path.join(os.path.dirname(__file__), "ckpt.pt")
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            generator = load_csm_1b(
                ckpt_path=model_path,
                device=device
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return generator

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        logger.info(f"Incoming {request.method} request to {request.url}")
        response = await call_next(request)
        logger.info(f"Request completed with status {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def convert_to_mp3(wav_data: bytes, sample_rate: int) -> bytes:
    """Convert WAV data to MP3 using ffmpeg"""
    try:
        # ffmpeg command to convert wav to mp3
        cmd = [
            'ffmpeg',
            '-f', 'wav',  # Input format
            '-i', 'pipe:0',  # Read from stdin
            '-f', 'mp3',  # Output format
            '-acodec', 'libmp3lame',  # Use LAME encoder
            '-ab', '128k',  # Bitrate
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono audio
            'pipe:1'  # Output to stdout
        ]
        
        # Run ffmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Feed wav data and get mp3 data
        mp3_data, stderr = process.communicate(input=wav_data)
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            raise RuntimeError("FFmpeg conversion failed")
            
        return mp3_data
        
    except Exception as e:
        logger.error(f"Error converting to MP3: {str(e)}")
        raise

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        model = get_generator()
        logger.info(f"Request: message='{request.message}', speaker={request.speaker}, temp={request.temperature}")
        
        # Generate audio from text
        logger.info("Generating audio...")
        try:
            audio = model.generate(
                text=request.message,
                speaker=request.speaker,
                context=[],  # No context for now
                temperature=request.temperature,
                max_audio_length_ms=request.max_audio_length_ms,
            )
            # Clear model caches after generation
            model._model.reset_caches()
            logger.info("Model caches cleared")
        except Exception as e:
            logger.error(f"Error during audio generation: {str(e)}")
            # Ensure caches are cleared even on error
            model._model.reset_caches()
            raise
        
        # Log audio properties
        logger.info(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
        
        # Convert to WAV format first
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0), sample_rate=model.sample_rate, format="wav")
        buffer.seek(0)
        wav_data = buffer.read()
        
        # Convert WAV to MP3
        mp3_data = convert_to_mp3(wav_data, model.sample_rate)
        
        # Create response with audio data
        return StreamingResponse(
            io.BytesIO(mp3_data),
            media_type="audio/mpeg",
            headers={
                "Sample-Rate": str(model.sample_rate)
            }
        )
    except ValidationError as e:
        logger.error(f"Request validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
