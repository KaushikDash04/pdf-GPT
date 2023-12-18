import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import huggingface_hub
from htmlTemp import css, user_template, ai_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectore_stote(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    # embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = huggingface_hub.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory,
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(ai_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 


def main():
    load_dotenv()
    st.set_page_config(page_title="pdf-GPT", page_icon=":books:", layout="wide")
    with st.container():
        st.header("Chat with multiple PDFs :books:")

    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
        z-index: 1;
        padding: 20px;
        background-color: #AAAAAA;
        border-radius: 15px;
        }}
    </style>
    """

    st.markdown(styl, unsafe_allow_html=True)

    with st.container():
        user_question = st.text_input(label = "", label_visibility="collapsed", disabled=False, placeholder="Ask a question about your documents:")
        if user_question:
            handle_user_input(user_question)

    with st.container():
        st.write(css, unsafe_allow_html=True)
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
    # with st.container():
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(user_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)
    #     st.write(ai_template.replace("{{MSG}}", "Hiiii"), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload PDFs and click on Process", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing...."):
                # get pdf texts
                raw_texts = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_texts)
                
                # create vector store
                vectorstore = get_vectore_stote(text_chunks)
                st.write("Ask your assistant!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
