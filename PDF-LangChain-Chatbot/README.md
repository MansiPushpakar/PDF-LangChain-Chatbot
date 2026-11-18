# PDF LangChain Chatbot

A Streamlit PDF question-answering app using LangChain, OpenAI embeddings/chat, and Astra DB vector storage.

## Features
- Upload one or more text-based PDF files.
- Extract PDF text and split it into overlapping chunks.
- Store embeddings in Astra DB.
- Retrieve relevant passages and answer questions with source/page context.
- Keeps chat history for the current browser session.

> Scanned/image-only PDFs need OCR. This starter does not include OCR.

## 1. Create credentials
Create a new OpenAI API key and an Astra DB application token. Configure an Astra DB database and copy its Data API endpoint.

**Important:** If credentials were previously pasted into a chat, notebook, or GitHub repository, revoke/rotate them before deployment. Do not reuse exposed keys.

## 2. Run locally
Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` as a template, then replace all placeholder values with your newly generated credentials.

Run:

```bash
streamlit run app.py
```

## 3. Push to GitHub
Create a new repository, for example `PDF-LangChain-Chatbot`.

```bash
git init
git add .
git status
git commit -m "Initial commit: PDF LangChain chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/PDF-LangChain-Chatbot.git
git push -u origin main
```

Check `git status` before committing. `.streamlit/secrets.toml`, PDFs, `.env`, and virtual environments are ignored by `.gitignore`.

## 4. Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app** and choose your `PDF-LangChain-Chatbot` repository.
3. Select branch `main` and main file path `app.py`.
4. Deploy.
5. In the deployed app, open **App settings → Secrets**.
6. Paste the TOML configuration below, replacing the placeholders with your new credentials.

```toml
OPENAI_API_KEY = "your-new-openai-api-key"
ASTRA_DB_APPLICATION_TOKEN = "your-new-astra-token"
ASTRA_DB_API_ENDPOINT = "https://your-database-id-your-region.apps.astra.datastax.com"
ASTRA_DB_COLLECTION = "pdf_chatbot_documents"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

Save secrets and reboot the app.

## Notes
- OpenAI API usage may incur charges; monitor your usage and set limits.
- Astra DB availability and pricing depend on your account and plan.
- The default collection name is `pdf_chatbot_documents`. Use a dedicated collection for this app.
- Re-uploading a PDF in a new browser session may create duplicate chunks in Astra DB. For a production app, add document-level deduplication and delete/update workflows.
- Do not upload confidential or personally identifiable documents unless you have permission and an appropriate data-protection basis.
