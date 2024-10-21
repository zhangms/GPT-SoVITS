# coding=utf-8

'''
requires Python 3.6 or later
pip install requests
'''
import base64
import json
import time
import uuid

import requests


# 填写平台申请的appid, access_token以及cluster
# appid = "1581839837"
# access_token= "Aw6zEm1AqvqRtUhKzg538VJMxDWE-Olb"
# cluster = "volcano_icl"

# voice_type = "S_5Ond9Tq01"

def voclengine_generate(voice_id: str, text: str, save_file_path: str):
    appid = "1581839837"
    access_token = "Aw6zEm1AqvqRtUhKzg538VJMxDWE-Olb"
    cluster = "volcano_icl" if voice_id.startswith('S_') else "volcano_tts"
    host = "openspeech.bytedance.com"
    api_url = f"https://{host}/api/v1/tts"

    header = {"Authorization": f"Bearer;{access_token}"}

    request_json = {
        "app": {
            "appid": appid,
            "token": "access_token",
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice_type": voice_id,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"

        }
    }

    try:
        # save_file_path = "generate.mp3"
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        # print(f"resp body: \n{resp.json()}")
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open(save_file_path, "wb")
            file_to_save.write(base64.b64decode(data))
        return save_file_path
    except Exception as e:
        e.with_traceback()


if __name__ == "__main__":
    for i in range(1):
        start = time.time()
        voclengine_generate(voice_id='S_9G6DhzV11',
                            text="Hey there! Feel free to share your thoughts or any interesting stories. I'm all ears!",
                            save_file_path="generate1.mp3")
        print("Generation time: " + str(time.time() - start))

        start = time.time()
        voclengine_generate(voice_id='zh_female_shuangkuaisisi_moon_bigtts',
                            text="Hey there! Feel free to share your thoughts or any interesting stories. I'm all ears!",
                            save_file_path="generate2.mp3")
        print("Generation time: " + str(time.time() - start))
