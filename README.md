# CSM CHAT - WEB INTERFACE - by Bron Fieldwalker
A web interface that works on CPU - this is a work in progress at the moment it only plays the Voices to test them out.
To upgrade this to a full voice system we need to hook the voice up to a LLM for responses, CSM is just the text-voice model.

![image](https://github.com/user-attachments/assets/86e272b6-e203-4c6a-bc96-af2885fba937)

## Web Interface

This repository includes a modern web interface built with:
- FastAPI backend for efficient model serving
- React frontend with Mantine UI components
- Optimized audio processing using ffmpeg

### Features
- Real-time text-to-speech generation
- Multiple speaker selection (0-2)
- Adjustable temperature (0.1-1.0)
- Configurable max audio length (1-30s)
- Automatic audio playback
- Progress tracking during generation
- Audio format optimization:
  - MP3 format with 128kbps bitrate
  - Mono audio for efficiency
  - LAME encoder for high-quality compression
  - Original sample rate preservation

### Performance Optimizations
- Local tokenizer caching
- CUDA acceleration when available
- Efficient audio transmission
- Minimized state updates
- Proper error handling
- Browser-compatible audio formats

## Audio Format Optimization

The web interface automatically optimizes generated audio using ffmpeg with the following settings:
- MP3 format for better compression and wider compatibility
- 128kbps bitrate for optimal quality/size balance
- Mono audio to reduce file size
- LAME encoder for high-quality compression
- Original sample rate preservation for audio fidelity

These optimizations result in:
- Smaller file sizes for faster transmission
- Wider browser and device compatibility
- Maintained audio quality
- Efficient storage and bandwidth usage

## Added Dependencies

```
# FastAPI backend dependencies
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
pydantic==2.6.0

# Audio processing
ffmpeg-python==0.2.0
pydub==0.25.1
```

## Not Included

* Not included in this repo are the CSM and llama model
* Model checkpoint file: server/ckpt.pt (PyTorch weights file)
* Model definition: models.py (Contains ModelArgs and Model class definitions)

# Quick Start Guide

1. Clone the repository and enter the directory:
```bash
git clone https://github.com/syndicate604/csm/
cd csm
```

2. Set up Python environment and install backend dependencies:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Login to access model weights
huggingface-cli login
```

3. Install system dependencies:
```bash
# For Ubuntu/Debian
sudo apt-get install ffmpeg

# For macOS
brew install ffmpeg

# For Windows
# Install ffmpeg from https://ffmpeg.org/download.html
# The triton package cannot be installed in Windows
# Use: pip install triton-windows
```

4. Set up and start the FastAPI backend:
```bash
cd server
uvicorn main:app --reload  # The server will run on http://localhost:8000
```

5. In a new terminal, set up and start the React frontend:
```bash
cd client
npm install
npm start  # The frontend will run on http://localhost:3000
```

The web interface will be available at http://localhost:3000


-------------------------------------------------------------------------------------------------
# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Usage

Generate a sentence

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
