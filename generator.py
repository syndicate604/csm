from dataclasses import dataclass
from typing import List, Tuple
import os
import torch
import torchaudio
import logging
import traceback
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global tokenizer cache
_CACHED_TOKENIZER = None

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """Load the Llama 3.2 tokenizer with caching"""
    global _CACHED_TOKENIZER
    
    if _CACHED_TOKENIZER is not None:
        logger.info("Using cached tokenizer")
        return _CACHED_TOKENIZER
        
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    cache_dir = os.path.join(os.path.dirname(__file__), "tokenizer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    try:
        # First try loading locally
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, local_files_only=True)
        logger.info("Loaded tokenizer from cache")
    except Exception as e:
        logger.info("Tokenizer not found locally, downloading...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, local_files_only=False)
        logger.info("Downloaded tokenizer successfully")
    
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    logger.info("Tokenizer configured successfully!")
    
    _CACHED_TOKENIZER = tokenizer
    return tokenizer


class Generator:
    _cached_mimi = None
    _cached_watermarker = None
    
    def __init__(
        self,
        model: Model,
    ):
        try:
            self._model = model
            self._model.setup_caches(1)
            logger.info("Model caches set up")

            self._text_tokenizer = load_llama3_tokenizer()
            logger.info("Text tokenizer loaded")

            device = next(model.parameters()).device
            logger.info(f"Using device: {device}")
            
            # Cache MIMI model
            if Generator._cached_mimi is None:
                logger.info("Loading MIMI model...")
                cache_dir = os.path.join(os.path.dirname(__file__), "mimi_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                try:
                    # First try loading locally
                    mimi_weight = hf_hub_download(
                        loaders.DEFAULT_REPO, 
                        loaders.MIMI_NAME, 
                        local_files_only=True,
                        cache_dir=cache_dir
                    )
                    logger.info("Found MIMI model in cache")
                except Exception as e:
                    logger.info("MIMI model not found locally, downloading...")
                    # If not found, download it
                    mimi_weight = hf_hub_download(
                        loaders.DEFAULT_REPO, 
                        loaders.MIMI_NAME, 
                        local_files_only=False,
                        cache_dir=cache_dir
                    )
                    logger.info("MIMI model downloaded successfully")
                
                Generator._cached_mimi = loaders.get_mimi(mimi_weight, device=device)
                Generator._cached_mimi.set_num_codebooks(32)
                logger.info("MIMI model initialized")
            
            self._audio_tokenizer = Generator._cached_mimi
            logger.info("Audio tokenizer set")

            # Cache watermarker
            if Generator._cached_watermarker is None:
                logger.info("Loading watermarker...")
                try:
                    Generator._cached_watermarker = load_watermarker(device=device)
                    logger.info("Watermarker loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading watermarker: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
            self._watermarker = Generator._cached_watermarker

            self.sample_rate = self._audio_tokenizer.sample_rate
            self.device = device
            logger.info("Generator initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing Generator: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # Ensure speaker is valid
        speaker = max(0, min(2, speaker))
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        try:
            # Reset model caches at start
            self._model.reset_caches()
            logger.info("Model caches reset before generation")

            max_audio_frames = int(max_audio_length_ms / 80)
            tokens, tokens_mask = [], []
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

            prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

            samples = []
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

            max_seq_len = 2048 - max_audio_frames
            if curr_tokens.size(1) >= max_seq_len:
                raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

            try:
                for _ in range(max_audio_frames):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    if torch.all(sample == 0):
                        break  # eos

                    samples.append(sample)

                    curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

                audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
            finally:
                # Clear any temporary tensors
                del curr_tokens
                del curr_tokens_mask
                del curr_pos
                del samples
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Initial normalization
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            audio = audio * 0.95  # Add small headroom

            # Watermarking
            # audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
            # audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

            # Final normalization and cleanup
            audio = audio.to(torch.float32)  # Ensure float32 format
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            audio = audio * 0.95  # Maintain headroom

            # Add a small fade out at the end
            if len(audio) > 100:
                fade_len = min(1000, len(audio) // 8)  # 1000 samples or 1/8 of audio
                fade = torch.linspace(1.0, 0.0, fade_len, device=self.device)
                audio[-fade_len:] *= fade

            # Clear any remaining temporary tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return audio
        
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            raise

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cpu") -> Generator:
    try:
        model_args = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )
        model = Model(model_args).to(device=device, dtype=torch.float32)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        generator = Generator(model)
        return generator
    
    except Exception as e:
        logger.error(f"Error loading CSM 1B: {str(e)}")
        logger.error(traceback.format_exc())
        raise
