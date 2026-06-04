# MeloTTS API & Streaming Service

A high-performance, streaming-first Text-to-Speech (TTS) service based on [MeloTTS](https://github.com/myshell-ai/MeloTTS). The service uses **ModelScope** for model weight downloads, utilizes **FastAPI** to support both custom endpoints and OpenAI-compatible streaming speech endpoints, and is structured according to open-source project best practices.

---

## 📂 Project Structure

```
.
├── melotts_service/              # Main package directory
│   ├── __init__.py               # Package metadata and version info
│   ├── config.py                 # Constant definitions and environment mappings
│   ├── schemas.py                # Request and response Pydantic models (with alias support)
│   ├── core.py                   # MeloTTS model loading and inference logic
│   ├── api.py                    # FastAPI route handlers and middleware
│   └── static/                   # Static front-end assets
│       └── index.html            # Animated voice portal dashboard
├── main.py                       # Root launcher script
├── test_client.py                # Test client script (exercises streams and aliases)
├── install_systemd.sh            # Automated systemd installer script
├── pyproject.toml                # Dependency definition (managed by uv)
└── uv.lock                       # Lockfile mapping exact dependency versions
```

---

## ⚡ API Endpoints

### 1. GET `/`
Serves the animated glassmorphism web portal dashboard, offering a playground to synthesize speech directly in the browser.

### 2. GET `/voices`
Lists all supported languages, their ModelScope repo IDs, and available speaker codes.

### 3. POST `/tts` (Custom & Compatibility Mode)
Synthesizes speech. Accepts standard request fields as well as aliases/fallbacks to easily interface with custom clients like `xiaozhi-esp32-server`.
*   **Request Schema**:
    ```json
    {
      "text": "要合成的文本",         // 或使用别名 "input"
      "language": "ZH",            // 选项: ZH, EN, JP, KR, ES, FR
      "speaker": "ZH",             // 或使用别名 "voice"。可选项如: ZH, EN-US, JP等
      "speed": 1.0,                // 语速范围 0.25 - 4.0
      "stream": false,             // 是否启用流式输出 (边合成边传输)
      "format": "mp3",             // 音频格式: mp3, wav, pcm
      "output_dir": "test_outputs" // 可选：指定在服务器端保存的音频文件目录
    }
    ```

### 4. POST `/v1/audio/speech` (OpenAI Compatibility Mode)
Accepts standard OpenAI format payloads, enabling dropping this server directly into clients like NextChat, OpenWebUI, or the `OpenAITTS` provider in the Xiaozhi server.

---

## ⚙️ Deployment & Systemd System Service Setup

To run this application as a daemon on production Linux servers (restarting on failures and starting automatically on system boot):

> [!NOTE]
> **在国内服务器访问 GitHub 遇到网络问题的解决方案**：
> 如果您的服务器在执行 `uv sync` 时遇到连接 `github.com` 超时或失败，建议在终端中配置 Git 全局代理代理替换：
> ```bash
> git config --global url."https://mirror.ghproxy.com/https://github.com/".insteadOf "https://github.com/"
> ```
> 配置后，Git 会在后台自动将 GitHub 依赖拉取转向国内的代理镜像，无需修改任何项目代码文件。

1.  Synchronize the environment dependencies first:
    ```bash
    uv sync
    ```
2.  Run the systemd setup script with root privileges:
    ```bash
    sudo ./install_systemd.sh
    ```
    *This installer automatically detects your local paths, current login user, and creates the systemd service file `/etc/systemd/system/melotts.service`.*

### Useful Service Commands
*   **Check status**: `systemctl status melotts.service`
*   **View live logs**: `journalctl -u melotts.service -f`
*   **Restart service**: `sudo systemctl restart melotts.service`

---

## 🤖 Xiaozhi-ESP32-Server Integration

To integrate this service with the [xiaozhi-esp32-server](https://github.com/diannaoc/xiaozhi-esp32-server) project:

Modify `main/xiaozhi-server/config.yaml` to specify either CustomTTS or OpenAITTS:

### Option A: CustomTTS Mode (Using Custom HTTP POST Adapter)
Configure the `CustomTTS` section under `TTS:`:
```yaml
TTS:
  CustomTTS:
    type: custom
    method: POST
    url: "http://127.0.0.1:8100/tts"
    headers:
      Content-Type: "application/json"
    params:
      input: "{prompt_text}"   # Supports our compatibility alias
      language: "ZH"           # Model language
      voice: "ZH"              # Speaker code alias (voice mapping to ZH)
      speed: 1.0
      stream: false            # Must be false as CustomTTS expects the complete file in response
      format: "mp3"            # Output format
    output_dir: "tmp/"
```

### Option B: OpenAI Mode
Configure the `OpenAITTS` section under `TTS:`:
```yaml
TTS:
  OpenAITTS:
    type: openai
    api_key: "any-dummy-key"
    api_url: "http://127.0.0.1:8100/v1/audio/speech"
    model: "tts-1"
    voice: "zh"               # Maps to MeloTTS Chinese Default voice
    speed: 1.0
    output_dir: "tmp/"
```
