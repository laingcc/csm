from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch
import os

os.environ["HF_TOKEN"]="TOKEN_HERE"

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
device="cpu"
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)
index = 0
while True:
    prompt = input('Prompt? > ')
    audio = generator.generate(
        text= prompt,
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )
    index+=1

    torchaudio.save("${0}.wav".format(index), audio.unsqueeze(0).cpu(), generator.sample_rate)

