import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

import json

def init_streamlit_page():
    st.set_page_config(
        page_title="Personal ChatGPT"
    )
    st.header("Personal ChatGPT")
    st.sidebar.title("Options")

#def init_messages():
#    clear_button = st.sidebar.button("Clear Conversation", key="clear")
#    if clear_button or "messages" not in st.session_state:
#        st.session_state.messages = [
#            SystemMessage(
#                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
#        ]
#        st.session_state.costs = []


# Memory
memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

def select_model():
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo", "gpt-4"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.9, step=0.1)

    llm = OpenAI(temperature=temperature, max_tokens=3000, model_name=model_name)
    #return ChatOpenAI(temperature=temperature, model_name=model_name)
    return llm

def build_chain(llm, title, raw_message):
    # prompt template
    prompt1 = PromptTemplate(
        input_variables=['title', 'raw_message'],
        template="""
        You are a professional Information Development expert working for an enterprise software in a semiconductor Company. 
        The software is a RMS (Recipe Management System). 
        The users are Process Engineers, and the tool is their primary work tool.
        Your task is to refine original error messages from my inputs for I will choose from your suggestions. \

        More information:
        - The tone of the voice should be humble and informative. Please do not be Joyful or funny.
        - The error message will appear on the web application.
        - Please go to the point. No need to add text apart from the minimum information necessary.
        - The modal will include two buttons. One to close the modal and one to upload again. Please provide labels.

        Guidelines for Writing Error Messages:
        - Be Clear: Ensure the message is straightforward and easy to understand.
        - Stay Concise: Only provide the necessary information.
        - Avoid Jargon: Use plain, easily understandable language.
        - Specify Severity: Differentiate between critical errors, warnings, or informational messages.
        - Offer Solutions: Provide steps or actions to resolve the issue when possible.
        - Use Neutral Language: Avoid negative or blame-oriented terms.
        - Consistent Terminology: Maintain consistency in phrasing across all messages.
        - Log Details for Support: Ensure detailed information is available behind the scenes for support, even if not shown to the user.
        - Maintain Professional Tone: Messages should be humble and informative without attempts at humor.

        The original error messages are as the following format:

        [Title] The one-liner title 
        [Message] The detailed messages.

        For each submission, you will output 3 suggested messages. 
        When output, format the output as JSON with the keys of title, and suggested outputs (an array with 3 items)
        
        title
        suggested_outputs: [output1, output2, output3]

        Now, based on above, write me the refined error based on the original input:

        [Title] {title}
        [Message] {raw_message}
        """)

    chain1 = LLMChain(llm=llm, prompt=prompt1, verbose=True, output_key='refined_error_message', memory=memory)

    return chain1.run({'title': title, 'raw_message': raw_message})


def main():
    _ = load_dotenv(find_dotenv())

    init_streamlit_page()
    llm = select_model()

    # Supervise user input
    title = st.text_input("Title")
    msg = st.text_area("Error Message")
    if st.button("Submit", type="primary") and title and msg:

        response = build_chain(llm, title, msg)
        st.json(json.loads(response))

    with st.expander('Message History'):
        st.info(memory.buffer)

if __name__ == "__main__":
    main()
