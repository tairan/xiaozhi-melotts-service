import uvicorn

def main():
    print("Starting MeloTTS API and Streaming Service on http://localhost:8100")
    uvicorn.run("melotts_service.api:app", host="0.0.0.0", port=8100, reload=False)


if __name__ == "__main__":
    main()
