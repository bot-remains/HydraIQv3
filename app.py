from chatbot.chat_utils import fetchChatHistory, chatWithChain
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# app config
st.set_page_config(page_title="FinBot", page_icon="🤖")
st.title("FinBot")

chat_history = fetchChatHistory("123456", "abcde")

for message in chat_history.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(chatWithChain(user_query, chat_history))
