
apt install -y ffmpeg cmake
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt


wget https://huggingface.co/lj1995/GPT-SoVITS/blob/main/s2G488k.pth -O ./GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth