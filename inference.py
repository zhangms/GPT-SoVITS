import datetime
import os
import time

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)


class Inference(object):
    def __init__(self):
        self.speakers = [
            "Binary",
            "Dara",
            "Fantasm",
            "MaXine",
            "Neon",
            "Pyro",
            "Vigor",
            "Vio",
            "Ziggy"
        ]

        self.ref_prompt_text = {
            "Binary": "Welcome to NovaJoy Island! Doesn't the name just sparkle with excitement,zip-zap?",
            "Dara": "Welcome! Before I introduce myself, please look this way, Pioneer--",
            "Fantasm": "Alright, everyone, please go see my adorable assistant, Katie, to pick up your very own tent.",
            "MaXine": "Welcome to NovaJoy Island! Doesn't the name just sparkle with excitement,zip-zap?",
            "Neon": "Welcome! Before I introduce myself, please look this way, Pioneer--",
            "Pyro": "Good morning, Drylander. The recent samples you brought have been a bit repetitive, but that's "
                    "okay. I trust you'll soon discover the patterns of rare insect appearances, glub glub.",
            "Vigor": "Good morning, Drylander. The recent samples you brought have been a bit repetitive, but that's "
                     "okay. I trust you'll soon discover the patterns of rare insect appearances, glub glub.",
            "Vio": "Alright, everyone, please go see my adorable assistant, Katie, to pick up your very own tent.",
            "Ziggy": "Good morning, Drylander. The recent samples you brought have been a bit repetitive, but that's "
                     "okay. I trust you'll soon discover the patterns of rare insect appearances, glub glub.",
        }

        self.tts_models = {}
        self.load_tts_model()

    def load_tts_model(self):
        for speaker in self.speakers:
            cfg_path = f"{root_path}/GPT_SoVITS/mycfg/tts_infer_{speaker.lower()}.yaml"
            ref_audio = f"{root_path}/GPT_SoVITS/gptsovits-930/{speaker}/{speaker}.wav"
            print(f"init tts_model|{speaker}|{cfg_path}|{ref_audio}")
            config = TTS_Config(cfg_path)
            tts_pipeline = TTS(config)
            tts_pipeline.set_ref_audio(ref_audio)
            self.tts_models[speaker.lower()] = tts_pipeline
            self.inference(speaker, "warmup", speaker)

    def get_speakers(self):
        return self.speakers

    def generator(self, text, speaker):
        req = {
            "text": text,
            "text_lang": "en",
            "text_split_method": "cut5",
            "ref_audio_path": None,
            "prompt_text": self.ref_prompt_text[speaker],
            "prompt_lang": "en",
            "return_fragment": True,
        }
        return self.tts_models[speaker.lower()].run(req)

    def inference(self, trace_id, text, speaker):
        req = {
            "text": text,
            "text_lang": "en",
            "text_split_method": "cut5",
            "ref_audio_path": None,
            "prompt_text": self.ref_prompt_text[speaker],
            "prompt_lang": "en"
        }

        start = time.time()
        tts_generator = self.tts_models[speaker.lower()].run(req)
        sr, audio_data = next(tts_generator)
        end = time.time()

        print(f"{datetime.datetime.now()}|TTS_INFERENCE|{trace_id}|{speaker}|{text}|rt:{end - start}")
        return sr, audio_data
