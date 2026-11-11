H# 🤖 Agentic RAG Document Assistant

An interactive **Agentic Retrieval-Augmented Generation (RAG)** web application built with **LangChain**, **LangGraph**, **OpenAI**, and **Streamlit**.

This application allows users to upload PDF documents, automatically index them into a vector database, and query an AI agent that dynamically decides when to retrieve context from the documents to provide grounded, accurate answers.

---

## ✨ Features

- **📄 Dynamic Document Processing**: Upload multiple PDF files simultaneously.
- **⚡ In-Memory Vector Store**: Fast document embedding using OpenAI's `text-embedding-3-small` and storing with LangChain's `InMemoryVectorStore`.
- **🛠️ Agentic Retrieval**: Uses a LangChain agent (`gpt-4o-mini`) equipped with a custom retrieval tool to fetch document context on-demand.
- **🧠 Conversational Memory**: Managed using LangGraph's `InMemorySaver` checkpointer for state retention across multi-turn interactions.
- **💬 Streamlit Chat Interface**: Simple, responsive UI providing a smooth chat experience.

---

## 🛠️ Project Architecture & Workflow

1. **Upload & Storage**: PDFs are uploaded via the Streamlit interface and saved locally to `./doc_files/`.
2. **Chunking & Embedding**: Documents are parsed with `PyPDFDirectoryLoader`, split into 2000-character chunks with 200-character overlap via `RecursiveCharacterTextSplitter`, and embedded.
3. **Agent Creation**: A LangGraph agent is initialized with system instructions to use the `retrieve_context` tool for fetching document context and avoiding hallucinations.
4. **Chat Execution**: User messages trigger the agent loop (`agent.invoke()`), maintaining thread context through a persistent `thread_id`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 2. Create and Activate a Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters langgraph python-dotenv pypdf

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here

```

### 5. Create Directory for Uploads

Make sure the local upload directory exists:

```bash
mkdir -p doc_files

```

---

## 💻 Usage

Run the Streamlit app:

```bash
streamlit run app.py

```

1. **Upload PDFs**: Drag and drop one or more PDF files.
2. **Process**: Wait for the app to embed and index the documents into memory.
3. **Chat**: Ask questions directly related to your uploaded documents in the chat interface!

---

## 📂 File Structure

```text
├── doc_files/             # Local directory where uploaded PDFs are temporarily saved
├── app.py                 # Main Streamlit application file
├── .env                   # Environment file containing secret keys
└── README.md              # Project documentation

```

---

## ⚙️ Configuration & Customization

- **Chunk Size / Overlap**: Modify `chunk_size` and `chunk_overlap` inside `RecursiveCharacterTextSplitter` in `app.py` to tune search granularity.
- **Models**: Change the embedding or chat completion models in `OpenAIEmbeddings` or `ChatOpenAI`.
- **System Prompt**: Customize `System_Prompt` to adjust agent behavior and domain boundaries.
