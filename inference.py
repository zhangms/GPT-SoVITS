import datetime
import json
import os
import time

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)


class Inference(object):
    def __init__(self):
        self.speakers = {}
        self.ref_prompt_text = []
        self.tts_models = {}
        self.load_speakers("/workspace/GPT-SoVITS/GPT_SoVITS/gptsovits-930/")
        self.load_tts_model()

    def load_speakers(self, model_path):
        with open(os.path.join(model_path, "model_list.json"), 'r') as f:
            speaker_cfgs = json.load(f)
            for key in speaker_cfgs:
                self.speakers[key] = speaker_cfgs[key]

    def load_tts_model(self):
        pass
        # for speaker in self.speakers:
        #     cfg_path = f"{root_path}/GPT_SoVITS/mycfg/tts_infer_{speaker.lower()}.yaml"
        #     ref_audio = f"{root_path}/GPT_SoVITS/gptsovits-930/{speaker}/{speaker}.wav"
        #     print(f"init tts_model|{speaker}|{cfg_path}|{ref_audio}")
        #     config = TTS_Config(cfg_path)
        #     tts_pipeline = TTS(config)
        #     tts_pipeline.set_ref_audio(ref_audio)
        #     self.tts_models[speaker.lower()] = tts_pipeline
        #     self.inference(speaker, "warmup", speaker)

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


if __name__ == "__main__":
    pipline = Inference()
