from dotenv import load_dotenv 
load_dotenv()  # Load environment variables from .env file

# from langchain_community import memory
from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.tools import tool 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver 
import streamlit as st

# Store data in streamlit session state
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):
    # Load the PDF document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    # creating a agent, tool, llm, prompt
    llm = ChatOpenAI(model="gpt-4o-mini")

    @tool
    def retrieve_context(query: str):
        """Retrieve relevant context from the vector store based on the query."""
        context = ""
        docs =vector_db.similarity_search(query, k=5)  # Retrieve top 3 similar documents
        for doc in docs:
            context = doc.page_content + "\n\n"

        print("-------------Tool Result", context)  # Debugging line to check the retrieved context
        return context

    System_Prompt = """
        You are a helpful assistant that can answer questions about Machine Learning. You have access to a tool called retriever_tool, which can retrieve relevant information from a PDF document about Machine Learning. 
        Use this tool to find information and provide accurate answers to user queries, do not make up any information. If the information is not available in the PDF, respond with 'I don't know'.
    """
    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=System_Prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

## Upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload a PDF document", type=["pdf"],accept_multiple_files=True)
    if uploaded:
        print("Uploaded files:", [file.name for file in uploaded])  # Debugging line to check uploaded files
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

## chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("ai").markdown(message["content"])


    query = st.chat_input("Ask anything related to the uploaded document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages": [{"role":"user","content":query}]},
            {"configurable": {"thread_id": 1}}
        )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})