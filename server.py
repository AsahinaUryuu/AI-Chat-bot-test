import asyncio
import re
import json
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import google.generativeai as genai

# === 🔑 配置区域 ===
GEMINI_KEY = "填写你的key" 
TTS_API_URL = "http://127.0.0.1:9880" 

# 模型和参考音频路径 (保持你修改后的 ref.wav)
# ⚠️ 注意：一定要确认 prompt_text 和你的 ref.wav 内容大致一致
# 模型采用**语音数据集**: [AI Hobbyist - Genshin_Datasets](https://github.com/AI-Hobbyist/Genshin_Datasets)
#  - 开源原神角色语音训练数据集
#  - 数据集整理者: [@红血球AE3803](https://github.com/AI-Hobbyist)
REF_AUDIO_PATH = r"填写你的文件夹地址\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\reference_audios\中文\emotions\ref.wav"
REF_TEXT = "白先生想采药，所以，七七来采。可是，想不起来了。" 
REF_LANG = "zh" 

# 模型权重路径 (你的路径)
GPT_MODEL_PATH = r"填写你的文件夹地址\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\七七_ZH-e10.ckpt"
SOVITS_MODEL_PATH = r"填写你的文件夹地址\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\七七_ZH_e10_s190_l32.pth"

# === 初始化 ===
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview') # 替换为你想使用的模型名称，目前采用gemini的模型
app = FastAPI()

# 标点符号切分正则
SPLIT_PATTERN = r"([，。！？；,.!?;])"

# === 🛠️ 初始化模型 ===
def init_model():
    print("🔄 正在初始化模型权重...")
    try:
        requests.get(f"{TTS_API_URL}/set_gpt_weights", params={"weights_path": GPT_MODEL_PATH})
        requests.get(f"{TTS_API_URL}/set_sovits_weights", params={"weights_path": SOVITS_MODEL_PATH})
        print("✅ 模型切换指令已发送")
    except Exception as e:
        print(f"⚠️ 无法连接 TTS 后台: {e}")

# === 核心函数：调用 TTS ===
def get_tts_audio(text):
    # 🚫 过滤无效文本：如果只有标点或空格，直接跳过，不调 API
    if not any(char.isalnum() or '\u4e00' <= char <= '\u9fff' for char in text):
        print(f"🚫 跳过无效文本: [{text}]")
        return None

    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": REF_AUDIO_PATH,
        "prompt_text": REF_TEXT,
        "prompt_lang": REF_LANG,
        "top_k": 5,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "streaming_mode": False,
        "media_type": "wav",
        "parallel_infer": True,
        "repetition_penalty": 1.35
    }
    try:
        response = requests.post(f"{TTS_API_URL}/tts", json=payload)
        if response.status_code == 200:
            return response.content
        else:
            print(f"❌ TTS Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"❌ TTS Connection Error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    init_model()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 客户端连接")
    
    chat_session = model.start_chat(history=[
        {"role": "user", "parts": ["你现在是七七。请用简短、呆萌、断断续续的口语与我对话。"]}
    ])

    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"👂 收到: {user_text}")

            response = await chat_session.send_message_async(user_text, stream=True)
            buffer = ""
            
            async for chunk in response:
                if chunk.text:
                    content = chunk.text
                    buffer += content
                    
                    # 发送字幕
                    await websocket.send_text(json.dumps({"type": "text", "content": content}))
                    
                    # 切分逻辑
                    parts = re.split(SPLIT_PATTERN, buffer)
                    if len(parts) > 1:
                        for i in range(0, len(parts)-1, 2):
                            sentence = parts[i] + parts[i+1]
                            # 只有当句子包含实际文字时才去合成
                            if sentence.strip():
                                print(f"🗣️ 尝试合成: {sentence}")
                                audio_data = await asyncio.to_thread(get_tts_audio, sentence)
                                if audio_data:
                                    print(f"✅ 发送音频数据: {len(audio_data)} bytes")
                                    await websocket.send_bytes(audio_data)
                        buffer = parts[-1]
            
            # 处理尾巴
            if buffer.strip():
                print(f"🗣️ 收尾合成: {buffer}")
                audio_data = await asyncio.to_thread(get_tts_audio, buffer)
                if audio_data:
                    await websocket.send_bytes(audio_data)
            
            await websocket.send_text(json.dumps({"type": "status", "content": "done"}))

    except WebSocketDisconnect:
        print("❌ 客户端断开连接")
    except Exception as e:
        print(f"❌ 系统错误: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)