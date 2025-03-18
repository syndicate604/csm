import argparse
import os
import silentcipher
import torch
import torchaudio
import logging
import traceback
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This watermark key is public, it is not secure.
# If using CSM 1B in another application, use a new private key and keep it secret.
CSM_1B_GH_WATERMARK = [212, 211, 146, 56, 201]

# Global watermarker cache
_CACHED_WATERMARKER = None

def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


def load_watermarker(device: str = "cpu"):
    """Load the watermarker model"""
    global _CACHED_WATERMARKER
    
    if _CACHED_WATERMARKER is not None:
        logger.info("Using cached watermarker")
        return _CACHED_WATERMARKER
    
    try:
        logger.info(f"Loading watermarker on device: {device}")
        model = silentcipher.get_model(
            model_type="44.1k",
            device=device,
        )
    except Exception as e:
        logger.error(f"Model not found locally, downloading...")
        # If not found, download it
        model = silentcipher.get_model(
            model_type="44.1k",
            device=device,
        )
        logger.info("Download complete!")
    
    _CACHED_WATERMARKER = model
    logger.info("Watermarker loaded successfully")
    return model


@torch.inference_mode()
def watermark(
    watermarker: silentcipher.server.Model,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    try:
        # Ensure audio is on the correct device
        audio_array = audio_array.to(watermarker.device)
        
        audio_array_44khz = torchaudio.functional.resample(audio_array, orig_freq=sample_rate, new_freq=44100)
        with torch.no_grad():
            encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36)

        output_sample_rate = min(44100, sample_rate)
        encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=output_sample_rate)
        logger.info("Audio watermarked successfully")
        return encoded, output_sample_rate
    except Exception as e:
        logger.error(f"Error watermarking audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@torch.inference_mode()
def verify(
    watermarker: silentcipher.server.Model,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    try:
        # Ensure audio is on the correct device
        watermarked_audio = watermarked_audio.to(watermarker.device)
        
        watermarked_audio_44khz = torchaudio.functional.resample(watermarked_audio, orig_freq=sample_rate, new_freq=44100)
        with torch.no_grad():
            result = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

        is_watermarked = result["status"]
        if is_watermarked:
            is_csm_watermarked = result["messages"][0] == watermark_key
        else:
            is_csm_watermarked = False

        logger.info(f"Watermark verification result: {is_watermarked and is_csm_watermarked}")
        return is_watermarked and is_csm_watermarked
    except Exception as e:
        logger.error(f"Error verifying watermark: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def check_audio_from_file(audio_path: str) -> None:
    watermarker = load_watermarker(device="cpu")

    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, CSM_1B_GH_WATERMARK)

    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    logger.info(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()
