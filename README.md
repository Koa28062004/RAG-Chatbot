# RAG-Chatbot

## Overview
RAG-Chatbot is a Retrieval-Augmented Generation (RAG) chatbot designed to process PDF documents, extract relevant data (text, images, and tables), and generate responses based on the extracted information. It leverages tools like ChromaDB for database management and Streamlit for deployment.

## Features
- Converts PDF files to Markdown.
- Extracts images, captions, and tables from PDFs.
- Loads extracted data into ChromaDB for efficient querying.
- Generates responses based on the processed data.

## Prerequisites
- Python 3.9 or higher
- A valid Gemini API key (add it to a `.env` file as shown in `.env.example`).

## Installation

1. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Gemini API key to the `.env` file.

## Usage

### 1. Process PDF Data
Run the `mainV2.py` script to process PDF files and extract data:
```bash
python mainV2.py
```
This script will:
- Convert PDFs in the `data/` folder to Markdown in the `data-md/` folder.
- Extract images to the `output-images/` folder.
- Extract tables to the `output-tables/` folder.
- Save metadata to the `temp_process/` folder.

### 2. Deploy the Chatbot
Run the Streamlit app:
```bash
python -m streamlit run app.py
```
Access the chatbot in your browser at `http://localhost:8501`.

## Project Structure
```
RAG/
├── app.py                 # Streamlit app for chatbot deployment
├── mainV2.py              # Main script for data processing and response generation
├── utils/                 # Utility functions for PDF processing
├── data/                  # Input PDF files
├── data-md/               # Converted Markdown files
├── output-images/         # Extracted images
├── output-tables/         # Extracted tables
├── temp_process/          # Temporary JSON files for metadata
├── database/              # ChromaDB database files
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables
└── README.md              # Project documentation
```

## Dependencies
The project uses the following Python libraries:
- `google-generativeai`
- `chromadb`
- `langchain`
- `tqdm`
- `PyMuPDF`
- `sentence_transformers`
- `streamlit`
- `python-dotenv`

Refer to [requirements.txt](requirements.txt) for the full list.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.
