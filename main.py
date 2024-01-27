### This code is from https://medium.com/mlearning-ai/create-a-chatbot-in-python-with-langchain-and-rag-85bfba8c62d2

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from setup_FAISS import create_or_get_vector_store
import streamlit as st

from pprint import pprint

def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = ChatOpenAI(model="gpt-4")
    # llm = HuggingFaceHub(model="HuggingFaceH4/zephyr-7b-beta") # if you want to use open source LLMs
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit

    Args:
        user_question (str): User question
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    human_style = "background-color: #f2f2f2; color: #000000; border-radius: 10px; padding: 10px;"
    chatbot_style = "background-color: #f2f2f2; color: #000000; border-radius: 10px; padding: 10px;"

    for i, message in enumerate(st.session_state.chat_history):
        # If the message content is a list, join it into a single string
        if isinstance(message.content, list):
            message.content = '\n'.join(message.content)

        if i % 2 == 0:
            st.markdown(
                f"<p style='text-align: right;'><b>User</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )

def main():
    load_dotenv()
    # df = load_dataset()
    # chunks = create_chunks(df, 1000, 0)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a chatbot tasked with responding to questions about VA Loans.

        You should never answer a question with a question, and you should always respond with the most relevant documentation page.

        Do not answer questions that are not about the VA Loans

        Given a question, you should respond after reviewing the most relevant information below:\n
        {context}
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(
        page_title="VA Loan Chatbot",
        page_icon=":house:",
    )

    st.title("Documentation Chatbot")
    st.subheader("Chatbot with VA Loan specialization!")
    st.markdown(
        """
        This chatbot was created to answer questions about VA Loans.
        
        """
    )
    # st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e") # Image rights to Alex Knight on Unsplash

    user_question = st.text_input("Ask your question")
    with st.spinner("Processing..."):
        if user_question:
            handle_style_and_responses(user_question)

    st.session_state.conversation = get_conversation_chain(
        st.session_state.vector_store, system_message_prompt, human_message_prompt
    )


if __name__ == "__main__":
    main()