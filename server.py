import asyncio
import re
import json
import requests
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import google.generativeai as genai

# === 🔑 配置区域 ===
GEMINI_KEY = "XXXX" 
TTS_API_URL = "http://127.0.0.1:9880" 

# 模型和参考音频路径 (保持你之前的配置)
REF_AUDIO_PATH = r"Z:\AiChatBot\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\reference_audios\中文\emotions\ref.wav"
REF_TEXT = "白先生想采药，所以，七七来采。可是，想不起来了。" 
REF_LANG = "zh" 

# 模型权重路径 (保持你之前的配置)
GPT_MODEL_PATH = r"Z:\AiChatBot\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\七七_ZH-e10.ckpt"
SOVITS_MODEL_PATH = r"Z:\AiChatBot\GPT-SoVITS-1007-cu124\models\v4\原神-中文-七七_ZH\七七_ZH_e10_s190_l32.pth"

# === 初始化 ===
genai.configure(api_key=GEMINI_KEY)
# 使用 Gemini 1.5 Flash，它支持音频输入
model = genai.GenerativeModel('gemini-3-flash-preview')
app = FastAPI()

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

# === TTS 函数 ===
def get_tts_audio(text):
    if not any(char.isalnum() or '\u4e00' <= char <= '\u9fff' for char in text):
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
        return None
    except Exception as e:
        print(f"TTS Connection Error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    init_model()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 客户端连接")
    
    # 初始化对话，设置系统提示词
    # 技巧：告诉 Gemini 如果收到音频，先输出用户说了什么，再输出回答，用 JSON 格式
    SYSTEM_PROMPT = """
    你现在是《原神》里的僵尸娘七七。
    你的记忆力不好，说话简短、呆萌、断断续续。
    
    【重要规则】
    我可能会发给你“文本”或者“音频”。
    如果是“音频”，请你听音频里的内容。
    
    无论我发什么，请务必返回一个 JSON 格式，包含两个字段：
    1. "user_transcription": 如果是音频，这里填你听到的用户说的话；如果是文本，填原文本。
    2. "qiqi_response": 七七的回答（纯文本，不要Markdown）。
    
    例如：
    {"user_transcription": "你想喝椰奶吗", "qiqi_response": "椰奶...好喝。七七...喜欢。"}
    """
    
    chat_session = model.start_chat(history=[
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["{\"user_transcription\": \"收到\", \"qiqi_response\": \"七七...知道了。\"}"]}
    ])

    try:
        while True:
            # 1. 接收数据 (bytes 或 text)
            # receive() 可以自动判断是文本还是二进制
            message = await websocket.receive()
            
            user_input_content = None
            
            if "text" in message:
                # === 收到文本 ===
                text_data = message["text"]
                print(f"👂 收到文本: {text_data}")
                user_input_content = text_data
                
            elif "bytes" in message:
                # === 收到音频 (Unity发来的WAV) ===
                audio_bytes = message["bytes"]
                print(f"🎤 收到音频: {len(audio_bytes)} bytes")
                
                # 构造 Gemini 需要的音频格式
                user_input_content = {
                    "mime_type": "audio/wav",
                    "data": audio_bytes
                }

            # 2. 发送给 Gemini
            if user_input_content:
                # 非流式调用 (因为我们要解析JSON，流式比较麻烦)
                response = await chat_session.send_message_async(user_input_content)
                gemini_reply = response.text
                print(f"🧠 Gemini 原始回复: {gemini_reply}")

                # 3. 解析 JSON (尝试提取 transcript 和 response)
                try:
                    # 清理一下可能存在的 Markdown 代码块标记 ```json ... ```
                    clean_json = gemini_reply.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_json)
                    
                    user_text = parsed.get("user_transcription", "")
                    qiqi_text = parsed.get("qiqi_response", "")
                    
                    # A. 告诉前端：用户刚才说了什么 (用于显示在右边的气泡)
                    if user_text:
                         await websocket.send_text(json.dumps({
                            "type": "transcription", 
                            "content": user_text
                        }))
                    
                    # B. 告诉前端：七七回答了什么 (显示在左边)
                    if qiqi_text:
                        await websocket.send_text(json.dumps({
                            "type": "text", 
                            "content": qiqi_text
                        }))
                        
                        # C. 合成语音并发送
                        # 简单起见，这里不再流式切分，直接整句合成 (七七说话本来就短)
                        print(f"🗣️ 合成语音: {qiqi_text}")
                        audio_data = await asyncio.to_thread(get_tts_audio, qiqi_text)
                        if audio_data:
                            await websocket.send_bytes(audio_data)
                            
                except Exception as e:
                    print(f"⚠️ JSON解析失败，直接回退到普通模式: {e}")
                    # 如果解析失败，就把整段话当做回答
                    await websocket.send_text(json.dumps({"type": "text", "content": gemini_reply}))
                    audio_data = await asyncio.to_thread(get_tts_audio, gemini_reply)
                    if audio_data:
                        await websocket.send_bytes(audio_data)

            await websocket.send_text(json.dumps({"type": "status", "content": "done"}))

    except WebSocketDisconnect:
        print("❌ 客户端断开连接")
    except Exception as e:
        print(f"❌ 系统错误: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
