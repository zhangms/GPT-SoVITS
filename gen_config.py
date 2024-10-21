import argparse
import os

tts_infer = """
custom:
  bert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  version: v2
  t2s_weights_path: {}
  vits_weights_path: {}
default:
  bert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  version: v1
  vits_weights_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2G488k.pth
default_v2:
  bert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  version: v2
  vits_weights_path: /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
"""


class ModelConfig(object):
    speaker_id: str = ""
    ckpt_path: str = ""
    pth_path: str = ""
    ref_audio: str = ""
    ref_prompt_text: str = ""


def process_model(speaker_id, dir_path):
    print(speaker_id, dir_path)
    cfg = ModelConfig()
    cfg.speaker_id = speaker_id
    for f in os.listdir(dir_path):
        if f.endswith(".ckpt"):
            cfg.ckpt_path = os.path.join(dir_path, f)
        if f.endswith(".pth"):
            cfg.pth_path = os.path.join(dir_path, f)
        if f.endswith(".wav"):
            cfg.ref_audio = os.path.join(dir_path, f)
        if f.endswith(".txt"):
            with open(os.path.join(dir_path, f), 'r') as file:
                # Read the contents of the file
                content = file.read()
                cfg.ref_prompt_text = content

    # print(cfg.speaker_id, cfg.ckpt_path, cfg.pth_path, cfg.ref_audio, cfg.ref_prompt_text)
    speaker_ttf_infer = tts_infer.format(cfg.ckpt_path, cfg.pth_path)
    print(speaker_ttf_infer)


def gen_config(model_dir: str):
    print("Generating config...")
    for d in os.listdir(model_dir):
        dir_path = os.path.join(model_dir, d)
        if os.path.isdir(dir_path):
            speaker_id = d.lower()
            process_model(speaker_id, dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Custom Model Config")
    parser.add_argument('--model_dir', required=True, help="Model directory")
    args = parser.parse_args()
    gen_config(args.model_dir)
