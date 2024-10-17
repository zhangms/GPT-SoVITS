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

        self.tts_models = {}
        self.load_tts_model()

    def load_tts_model(self):
        for speaker in self.speakers:
            cfg_path = f"{root_path}/GPT_SoVITS/mycfg/tts_infer_{speaker.lower()}.yaml"
            config = TTS_Config(cfg_path)
            ref_audio = f"{root_path}/GPT_SoVITS/gptsovits-930/{speaker}/{speaker}.wav"
            print(f"init tts_model|{speaker}|{cfg_path}|{ref_audio}")
            tts_pipeline = TTS(config)
            tts_pipeline.set_ref_audio(ref_audio)
            self.tts_models[speaker.lower()] = tts_pipeline
            self.inference("warmup", speaker)

    def get_speakers(self):
        return self.speakers

    def inference(self, text, speaker):
        req = {
            "text": text,
            "text_lang": "en",
            "text_split_method": "cut5",
            "ref_audio_path": None
        }

        start = time.time()
        tts_generator = self.tts_models[speaker.lower()].run(req)
        sr, audio_data = next(tts_generator)
        end = time.time()

        print(f"TTS_INFERENCE|{speaker}|{text}|rt{end - start}")
        return sr, audio_data
