import os  # noqa

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from transformers import (  # noqa
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset  # noqa
from helpers import get_identity  # noqa
from io import BytesIO  # noqa
from threading import Event  # noqa
import concurrent.futures  # noqa
import soundfile as sf  # noqa
import torch  # noqa
import pygame  # noqa
import enum  # noqa
import time  # noqa

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# load the model
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
# load the vocoder, that is the voice encoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# we load this dataset to get the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")


class Speakers(enum.IntEnum):
    MALE = 1138
    FEMALE = 6799
    MALE_2 = 5667
    FEMALE_2 = 2271


# -------------- Threading methods --------------
def generate_speech(input_ids, speaker_embeddings, vocoder):
    speech = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
    return speech


def print_processing(event, speaker):
    counter = 1
    emoji, name = get_identity(speaker.name.replace("_2", "").lower())
    while not event.is_set():
        if counter > 3:
            print(f"{emoji} {name}: {' ' * counter}", end="\r")
            counter = 0
        print(f"{emoji} {name}: {'.' * counter}", end="\r")
        counter += 1
        time.sleep(0.2)


# -------------- Threading methods --------------


def text_to_speech(text, speaker=None):
    # preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)

    speaker_embeddings = (
        torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
        if speaker is not None
        else torch.randn((1, 512)).to(device)
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        stop_event = Event()

        # Submit threads
        speech_gen_thread = executor.submit(
            generate_speech, inputs["input_ids"], speaker_embeddings, vocoder
        )

        executor.submit(print_processing, stop_event, speaker)

        # Get result
        speech = speech_gen_thread.result()

        # Stop the delay print
        stop_event.set()

    # # TODO CHECK INIT MIXER IF NOT THEN SKIP
    # if pygame.mixer != None:
    #     # Wait until voice is done with going to the next sentence
    #     while pygame.mixer.music.get_busy():
    #         pygame.time.wait(100)


    # Create memory file from audio
    mp3_fp = BytesIO()
    sf.write(mp3_fp, speech.cpu().numpy(), samplerate=16000, format="mp3")

    # Rewind to beginning of the audio bytes
    mp3_fp.seek(0)

    # Play audio
    pygame.mixer.init(frequency=16000)
    pygame.mixer.music.load(mp3_fp, "mp3")
    pygame.mixer.music.play()


if __name__ == "__main__":
    text_to_speech(
        "This is a test example using an american female voice", Speakers.FEMALE
    )
