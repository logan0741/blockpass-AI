# GEMINI.md: Project Analysis for BlockPass AI

## Project Overview

This project, **BlockPass AI**, is a sophisticated OCR (Optical Character Recognition) service specialized for Korean contract documents. It leverages a powerful, fine-tuned AI model (`fasoo/Qwen2-VL-7B-Instruct-KoDocOCR`) to digitize documents from businesses like gyms and study rooms by extracting key information from images.

The application is built as a FastAPI web service, designed to be run on a GPU-accelerated environment (NVIDIA CUDA). It exposes a set of RESTful API endpoints for submitting images (via Base64 JSON payload or direct file upload) and receiving structured, digitized contract data. The architecture is modular, with clear separation between the API layer, the AI model wrapper, database communication, and utility functions. It also includes optional integration with ngrok for exposing the local server to the internet and connecting to a remote database.

**Key Technologies:**
- **Backend Framework:** FastAPI
- **Web Server:** Uvicorn
- **AI/ML:** PyTorch, Hugging Face (Transformers, Hub)
- **Core Model:** Qwen2-VL-7B (fine-tuned for Korean OCR)
- **Data Validation:** Pydantic
- **Environment:** Python 3.12+, Conda/venv

## Building and Running

The project setup and execution process is well-documented in the `README.md`.

### 1. Environment Setup

It is highly recommended to use the provided `ocr_env` Python virtual environment.

```bash
# Create a virtual environment (if not already present)
python3 -m venv ocr_env

# Activate the environment
source ocr_env/bin/activate
```

### 2. Installing Dependencies

The project uses `pip` and a `requirements.txt` file. Some packages, like PyTorch, have specific installation requirements based on the CUDA version.

```bash
# First, install PyTorch for your specific CUDA version (e.g., CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

### 3. Acquiring the AI Model

The core OCR model is hosted on Hugging Face Hub and must be downloaded into the `./models/` directory.

```bash
# Download the model files using huggingface-cli
huggingface-cli download fasoo/Qwen2-VL-7B-Instruct-KoDocOCR \
  --local-dir ./models/Qwen2-VL-7B-Instruct-KoDocOCR
```

### 4. Configuration

The application is configured via a `.env` file. Create one by copying the example and modifying it as needed.

```bash
cp .env.example .env
# Edit the .env file with your settings (e.g., NGROK_AUTH_TOKEN)
```

### 5. Running the Server

The main entry point is `main.py`, which runs the FastAPI application using Uvicorn.

```bash
# Ensure the virtual environment is active
source ocr_env/bin/activate

# Start the server
python main.py
```
- The API will be available at `http://localhost:8000`.
- Interactive API documentation (Swagger UI) is at `http://localhost:8000/docs`.

### 6. Testing

While there is no dedicated test suite (`tests/` directory) visible, the API can be tested via the `/api/v1/health` endpoint or by submitting requests to the OCR endpoints using tools like `curl` or the provided Python script examples in `README.md`.

## Development Conventions

- **Modularity:** The codebase is well-organized into distinct modules:
    - `main.py`: Application entry point, lifecycle management.
    - `config/`: Handles application settings via Pydantic.
    - `src/`: Core application logic.
        - `api.py`: Defines all FastAPI routes.
        - `model.py`: A wrapper for the Hugging Face OCR model, abstracting the loading and inference logic.
        - `database.py`: Manages communication with the external database.
        - `schemas.py`: Contains all Pydantic models for request/response validation and serialization.
        - `utils.py`: Holds helper functions for tasks like image processing.
- **Typing:** The code extensively uses Python's type hints, which are enforced by FastAPI and Pydantic for robust data validation.
- **Configuration:** Environment variables (`.env` file) are loaded into a Pydantic `Settings` object, providing a single, type-safe source of configuration.
- **Asynchronous Code:** The API endpoints are defined with `async def`, leveraging FastAPI's asynchronous capabilities for better performance.
- **Error Handling:** The API uses `HTTPException` to return clear, standard HTTP error responses. Resource management (e.g., cleaning up temporary image files) is handled reliably using `try...except...finally` blocks.
