# Virtual-Chetas
Virtual-Chetas is a voice-based assistant that listens for speech, transcribes it with Whisper, uses camera input for visual context, and routes the request to an Ollama model.

## Folder Structure

```text
Virtual-Chetas/
├── main.py
├── README.md
├── requirements.txt
├── variables.py
├── data/
│   └── databaseModel.py
├── model/
│   ├── convo.py
│   ├── model_selector.py
│   └── vision.py
├── retirival/
│   ├── cosine.py
│   └── RAGcontext.py
└── vedio_capture/
	├── load.py
	└── model.py
```

## Features

- Voice activity detection and speech-to-text
- Camera-based person and object detection
- Face recognition for known people
- Simple memory storage in SQLite
- Ollama-based response generation and task routing

## Requirements

- Python 3.10 or newer
- Microphone access
- Camera access
- Ollama installed and running locally

## Install

```bash
pip install -r requirements.txt
```

## Run

Start Ollama, then run:

```bash
python main.py
```

## Notes

- The project uses local model files and NumPy embeddings stored in `npySaves/`.
- The memory database is created under `data/database/memory.db`.
