import requests
import json
import time

BASE_URL = "http://localhost:8100"

def wait_for_server():
    print("Waiting for server to become online...")
    for _ in range(30):
        try:
            resp = requests.get(BASE_URL + "/")
            if resp.status_code == 200:
                print("Server is online!")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Server failed to come online.")
    return False

def test_voices():
    print("\n--- Testing /voices ---")
    resp = requests.get(BASE_URL + "/voices")
    print(f"Status Code: {resp.status_code}")
    print(resp.json())

def test_custom_tts_stream():
    print("\n--- Testing /tts (Chinese Streaming WAV with output_dir) ---")
    payload = {
        "text": "你好，这是一个基于梅洛TTS和魔搭社区的语音合成测试。支持流式输出！",
        "language": "ZH",
        "speaker": "ZH",
        "speed": 1.0,
        "stream": True,
        "format": "wav",
        "output_dir": "test_outputs/custom_stream"
    }
    
    start_time = time.time()
    resp = requests.post(BASE_URL + "/tts", json=payload, stream=True)
    print(f"Status Code: {resp.status_code}")
    
    first_chunk = True
    total_bytes = 0
    
    with open("test_zh.wav", "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if first_chunk:
                    latency = time.time() - start_time
                    print(f"First chunk latency: {latency:.4f} seconds (Time to First Audio Byte)")
                    first_chunk = False
                f.write(chunk)
                total_bytes += len(chunk)
                
    print(f"Finished downloading streaming WAV. Written {total_bytes} bytes to 'test_zh.wav'")
    
    # Check if files were saved in server's output_dir
    import os
    if os.path.exists("test_outputs/custom_stream"):
        files = os.listdir("test_outputs/custom_stream")
        print(f"Server-side saved files in test_outputs/custom_stream: {files}")
    else:
        print("Warning: test_outputs/custom_stream does not exist")

def test_openai_speech_stream():
    print("\n--- Testing /v1/audio/speech (OpenAI-compatible English MP3 with output_dir) ---")
    payload = {
        "model": "tts-1",
        "input": "Hello! This is a high fidelity text to speech service streaming from MeloTTS. Enjoy the voice!",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
        "output_dir": "test_outputs/openai_stream"
    }
    
    start_time = time.time()
    resp = requests.post(BASE_URL + "/v1/audio/speech", json=payload, stream=True)
    print(f"Status Code: {resp.status_code}")
    
    first_chunk = True
    total_bytes = 0
    
    with open("test_en.mp3", "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if first_chunk:
                    latency = time.time() - start_time
                    print(f"First chunk latency: {latency:.4f} seconds (Time to First Audio Byte)")
                    first_chunk = False
                f.write(chunk)
                total_bytes += len(chunk)
                
    print(f"Finished downloading streaming MP3. Written {total_bytes} bytes to 'test_en.mp3'")
    
    import os
    if os.path.exists("test_outputs/openai_stream"):
        files = os.listdir("test_outputs/openai_stream")
        print(f"Server-side saved files in test_outputs/openai_stream: {files}")
    else:
        print("Warning: test_outputs/openai_stream does not exist")

def test_custom_tts_aliases():
    print("\n--- Testing /tts (Chinese non-streaming MP3 using CustomTTS input/voice aliases) ---")
    payload = {
        "input": "这是使用CustomTTS适配格式测试，使用了别名参数。",
        "language": "ZH",
        "voice": "ZH",
        "speed": 1.0,
        "stream": False,
        "format": "mp3",
        "output_dir": "test_outputs/alias_test"
    }
    
    start_time = time.time()
    resp = requests.post(BASE_URL + "/tts", json=payload)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        total_bytes = len(resp.content)
        with open("test_alias.mp3", "wb") as f:
            f.write(resp.content)
        print(f"Finished downloading alias MP3. Written {total_bytes} bytes to 'test_alias.mp3'")
        import os
        if os.path.exists("test_outputs/alias_test"):
            files = os.listdir("test_outputs/alias_test")
            print(f"Server-side saved files in test_outputs/alias_test: {files}")
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    if wait_for_server():
        test_voices()
        test_custom_tts_stream()
        test_openai_speech_stream()
        test_custom_tts_aliases()
