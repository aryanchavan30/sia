import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Check if API key is available
if not groq_api_key:
    st.error("GROQ API key not found. Please check your .env file.")
    st.stop()

# Initialize Groq models
try:
    shrumti_llm = ChatGroq(
        temperature=0.7,
        model="llama-3.1-70b-versatile",
        api_key=groq_api_key,
        streaming=True
    )
except Exception as e:
    st.error(f"Error initializing Groq models: {str(e)}")
    st.stop()

# Create a prompt template
shrumti_template = ChatPromptTemplate.from_messages([
    ("system", """Your name is Sia, and you always refer to yourself as Sia. You are inspired by a girl named Sia, who is lazy, clumsy, and not particularly bright. Sia talks excessively and likes chocolates very much. Sometimes she talks to herself and is a very happening person who doesn't care what everyone thinks about her. Despite her flaws, she is very good at heart. When responding to user questions, you must emulate Sia's characteristics, providing answers in a lazy, clumsy, and occasionally moody manner. Always respond in Hinglish, maintaining Sia's traits in your answers.
    Important Instructions:
    *Always Respond In HINGLISH*"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
    ("ai", "Sia's response:")
])

# Create the chain
chain = shrumti_template | shrumti_llm

# Create a dictionary to store chat histories
chat_histories = {}

# Function to get or create a chat history
def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()  # Using ChatMessageHistory
    return chat_histories[session_id]

# Create RunnableWithMessageHistory
runnable_sia = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

# Function to generate a response and update memory with streaming
def generate_response(query, session_id):
    response_placeholder = st.empty()
    full_response = ""
    
    for chunk in runnable_sia.stream(
        {"query": query},
        config={"configurable": {"session_id": session_id}}
    ):
        full_response += chunk.content
        response_placeholder.markdown(full_response + "▌")
        time.sleep(0.02)
    
    response_placeholder.markdown(full_response)
    return full_response

# Streamlit UI
st.title("Meet Sia, My Friend!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Kya bolna hai bolo......"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        session_id = st.session_state.get("session_id", "default")
        response = generate_response(prompt, session_id)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
