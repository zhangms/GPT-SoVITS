import argparse
import json
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


def process_model_config(speaker_id, dir_path):
    print(speaker_id, dir_path)
    cfg = {
        "speaker_id": speaker_id
    }
    for f in os.listdir(dir_path):
        if f.endswith(".ckpt"):
            cfg["ckpt_path"] = os.path.abspath(os.path.join(dir_path, f))
        if f.endswith(".pth"):
            cfg["pth_path"] = os.path.abspath(os.path.join(dir_path, f))
        if f.endswith(".wav"):
            cfg["ref_audio"] = os.path.abspath(os.path.join(dir_path, f))
        if f.endswith(".txt"):
            with open(os.path.join(dir_path, f), 'r') as file:
                # Read the contents of the file
                content = file.read()
                cfg["ref_prompt_text"] = content

    return cfg


def write_tts_infer(cfg):
    # print(cfg.speaker_id, cfg.ckpt_path, cfg.pth_path, cfg.ref_audio, cfg.ref_prompt_text)
    speaker_ttf_infer = tts_infer.format(cfg["ckpt_path"], cfg["pth_path"])
    os.makedirs("./GPT_SoVITS/mycfg/", exist_ok=True)
    speaker_id = cfg["speaker_id"]
    tts_infer_file = f"./GPT_SoVITS/mycfg/tts_infer_{speaker_id}.yaml"
    print(tts_infer_file)
    with open(tts_infer_file, 'w') as f:
        f.write(speaker_ttf_infer)
    cfg["tts_infer_file"] = os.path.abspath(tts_infer_file)


def gen_config(model_dir: str):
    print("Generating config...")
    cfg_map = {}
    for d in os.listdir(model_dir):
        dir_path = os.path.join(model_dir, d)
        if os.path.isdir(dir_path):
            speaker_id = d.lower()
            cfg = process_model_config(speaker_id, dir_path)
            write_tts_infer(cfg)
            cfg_map[speaker_id] = cfg

    with open(os.path.join(model_dir, "model_list.json"), 'w') as file:
        json.dump(cfg_map, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Custom Model Config")
    parser.add_argument('--model_dir', required=True, help="Model directory")
    args = parser.parse_args()
    gen_config(args.model_dir)
