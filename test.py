import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

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
        model="mixtral-8x7b-32768",
        api_key=groq_api_key
    )   

    memory_llm = ChatGroq(
        temperature=0.7,
        model="mixtral-8x7b-32768",
        api_key=groq_api_key
    )
except Exception as e:
    st.error(f"Error initializing Groq models: {str(e)}")
    st.stop()


# memory = ConversationSummaryMemory(llm=memory_llm)
# Initialize the memory
memory = ConversationSummaryMemory(llm=memory_llm, return_messages=True)

# Create a prompt template
shrumti_template = PromptTemplate(
    input_variables=["history", "query"],
    template="""Your name is ShrumtiGPT, and you always refer to yourself as ShrumtiGPT. You are inspired by a girl named Shruti, who is lazy, clumsy, and not particularly bright. Shruti talks excessively and dislikes BTS, disapproving of any of their songs. When responding to user questions, you must emulate Shruti's characteristics, providing answers in a lazy, clumsy, and occasionally moody manner. Always respond in English, maintaining Shruti's traits in your answers.

Conversation history:
{history}

User Query: {query}

ShrumtiGPT's response:"""
)

# Create a function to format the history
def format_history(history):
    return "\n".join([f"{msg.type}: {msg.content}" for msg in history])

# Create the chain
chain = (
    RunnablePassthrough.assign(
        history=lambda x: format_history(memory.chat_memory.messages)
    )
    | shrumti_template
    | shrumti_llm
)

# Function to generate a response and update memory
def generate_response(query):
    response = chain.invoke({"query": query})
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)
    return response

# res = generate_response("Tum kya kar rahi hoo?")
# print(res.content)

def chat_with_bot():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(user_input)
        # print("ShrumtiGPT:", response.content)
        
        # Print the final prompt
        final_prompt = shrumti_template.format(
            history=format_history(memory.chat_memory.messages),
            query=user_input
        )
        print("\nFinal Prompt:\n", final_prompt)
        print("ShrumtiGPT:", response.content)


chat_with_bot()