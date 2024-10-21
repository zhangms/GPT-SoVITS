import datetime
import json
import os
import time

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)


class Inference(object):
    def __init__(self):
        self.speakers = {}
        self.speaker_names = []
        self.ref_prompt_text = []
        self.tts_models = {}
        self.load_speakers("/workspace/GPT-SoVITS/GPT_SoVITS/gptsovits-930/")
        self.load_speakers("/workspace/GPT-SoVITS/GPT_SoVITS/NPC10.21/")
        print(self.speakers)
        self.load_tts_model()

    def load_speakers(self, model_path):
        with open(os.path.join(model_path, "model_list.json"), 'r') as f:
            speaker_cfgs = json.load(f)
            for key in speaker_cfgs:
                self.speakers[key.lower()] = speaker_cfgs[key]
                self.speaker_names.append(key)

    def load_tts_model(self):
        for speaker_id in self.speakers:
            speaker_cfg = self.speakers[speaker_id]
            cfg_path = speaker_cfg["tts_infer_file"]
            ref_audio = speaker_cfg["ref_audio"]
            print(f"init tts_model|{speaker_id}|{cfg_path}|{ref_audio}")
            config = TTS_Config(cfg_path)
            tts_pipeline = TTS(config)
            tts_pipeline.set_ref_audio(ref_audio)
            self.tts_models[speaker_id] = tts_pipeline
            self.inference(speaker_id, speaker_id, "warmup")

    def get_speakers(self):
        return list(self.speaker_names)

    def generator(self, speaker, text):
        speaker_id = speaker.lower()
        req = {
            "text": text,
            "text_lang": "en",
            "text_split_method": "cut5",
            "ref_audio_path": None,
            "prompt_text": self.speakers[speaker_id]["ref_prompt_text"],
            "prompt_lang": "en",
            "return_fragment": True,
        }
        return self.tts_models[speaker_id].run(req)

    def inference(self, trace_id, speaker, text):
        speaker_id = speaker.lower()
        req = {
            "text": text,
            "text_lang": "en",
            "text_split_method": "cut5",
            "ref_audio_path": None,
            "prompt_text": self.speakers[speaker_id]["ref_prompt_text"],
            "prompt_lang": "en"
        }

        start = time.time()
        tts_generator = self.tts_models[speaker_id].run(req)
        sr, audio_data = next(tts_generator)
        end = time.time()

        print(f"{datetime.datetime.now()}|TTS_INFERENCE|{trace_id}|{speaker}|{text}|rt:{end - start}")
        return sr, audio_data


if __name__ == "__main__":
    pipline = Inference()
