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

# Define system_message_prompt and human_message_prompt as global variables
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

def get_conversation_chain(vector_store: FAISS, system_message: str, human_message: str, num_docs: int = 5) -> ConversationalRetrievalChain:
    # Initialize the chatbot conversation chain here with the specified num_docs
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Fetch more nearest neighbors by adjusting the k parameter
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # Adjust the k value as needed
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
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
    # Retrieve and update the conversation state using Streamlit's session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store,
            system_message_prompt,
            human_message_prompt
        )

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    # Print the retrieved documents
    if "retriever" in response:
        st.write("Retrieved Documents:")
        for doc in response["retriever"]:
            st.write(doc)


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
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store()
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

    user_question = st.text_input("Ask your question")
    with st.spinner("Processing..."):
        if user_question:
            handle_style_and_responses(user_question)

if __name__ == "__main__":
    main()
