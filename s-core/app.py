import streamlit as st
from elasticsearch import Elasticsearch
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain, ConversationalRetrievalChain
from elasticsearch_chat_message_history import ElasticsearchChatMessageHistory
from pathlib import Path
from uuid import uuid4

import os

ES_URL = 'https://127.0.0.1:9200'
ES_USER = "elastic"
ES_USER_PASSWORD = "elastic"
CERT_PATH = 'D:\\es\\8.11.1\\kibana-8.11.1\\data\\ca_1701918227592.crt'

# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def create_llm():
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear that the LLM is typing the
    # responses in real-time.
    # callback_manager = CallbackManager([stream_handler])

    # (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    #                               "mistral-7b-instruct-v0.1.Q4_0.gguf")

    # initialize LlamaCpp LLM model
    # n_gpu_layers, n_batch, and n_ctx are for GPU support.
    # When not set, CPU will be used.
    # set 1 for Mac m2, and higher numbers based on your GPU support
    llm = LlamaCpp(
        model_path=str(Path("models/ggml-model-q4_0.gguf")),
        max_tokens=1024,

        # https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
        temperature=0,
        top_k=1,
        top_p=1,

        # callback_manager=callback_manager,
        # n_gpu_layers=1,
        # n_batch=512,
        n_ctx=1024,
        stop=["[INST]"],
        verbose=True,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        streaming=True,
    )

    return llm


# Set the webpage title
st.set_page_config(page_title="Your own aiChat!")

# Create a header element
st.header("Your own aiChat!")

# This sets the LLM's personality for each prompt.
# The initial personality provided is basic.
# Try something interesting and notice how the LLM responses are affected.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt",
)

client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_USER_PASSWORD),
    ca_certs=CERT_PATH,
    request_timeout=100,
    max_retries=10,
    retry_on_timeout=True
)

embeddings = HuggingFaceEmbeddings(
    model_name=str(Path("models/multilingual-e5-base")),
    model_kwargs = {'device': 'cpu'}
)

vector_store = ElasticsearchStore(
    embedding=embeddings,
    index_name="es-docs",
    es_connection=client
)

# Create LLM chain to use for our chatbot.
llm = create_llm()

template = (
    "<s>[INST]Combine the chat history and follow up question into "
    "a standalone question.[/INST]</s> [INST]Chat History: {chat_history}"
    "Follow up question: {question}[/INST]"
)
prompt = PromptTemplate.from_template(template)
question_generator_chain = LLMChain(llm=llm, prompt=prompt)
llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    condense_question_prompt=prompt,
    retriever=vector_store.as_retriever(),
    max_tokens_limit=1024,
    return_source_documents=True
)

session_id = str(uuid4())
chat_history = ElasticsearchChatMessageHistory(
    client=client,
    session_id=session_id,
    index="es-docs-chat-history"
)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "ELK에 대해 물어보세요."}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):
    # Add our input to the session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)
        chat_history.add_user_message(user_prompt)

    # Pass our input to the LLM chain and capture the final responses.
    # It is worth noting that the Stream Handler is already receiving the
    # streaming response as the llm is generating. We get our response
    # here once the LLM has finished generating the complete response.
    # response = llm_chain.invoke({"question": user_prompt})

    # Add the response to the session state
    # st.session_state.messages.append({"role": "assistant", "content": response})
    result = llm_chain({"question": user_prompt, "chat_history": chat_history.messages})
    print(chat_history.messages)
    response = result["answer"]
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)
        chat_history.add_ai_message(response)
