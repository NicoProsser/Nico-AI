from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, PromptHelper, StorageContext, load_index_from_storage
#from langchain.chat_models import ChatOpenAI # -> deprecated
from langchain_community.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'
#openai.api_key = os.environ["OPENAI_API_KEY"]

def init_index(directory_path):
    # model params
    # max_input_size: maximum size of input text for the model.
    # num_outputs: number of output tokens to generate.
    # max_chunk_overlap: maximum overlap allowed between text chunks.
    # chunk_size_limit: limit on the size of each text chunk.
    max_input_size = 4096
    num_outputs = 512
    #max_chunk_overlap = 20
    #chunk_size_limit = 600
    chunk_overlap_ratio=0.1

    # llm predictor with langchain ChatOpenAI
    # ChatOpenAI model is a part of the LangChain library and is used to interact with the GPT-3.5-turbo model provided by OpenAI
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio )#val_chunk_overlap_ratio) #chunk_size_limit=val_chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4-1106-preview", max_tokens=num_outputs))  #model_name="gpt-3.5-turbo

    # read documents from docs folder
    # DEBUG
    print('Path to docs:   ' + directory_path)
    documents = SimpleDirectoryReader(directory_path).load_data()

    # init index with documents data
    # This index is created using the LlamaIndex library. It processes the document content and constructs the index to facilitate efficient querying
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # save the created index
    #index.save_to_disk('index.json')
    index.storage_context.persist()

    return index


def chatbot(input_text):
    # load index
    #index = GPTVectorStoreIndex.load_from_disk('index.json')
    #vector_store = FaissVectorStore.from_persist_dir()
    storage_context = StorageContext.from_defaults(persist_dir='storage')

    # reload load index
    index = load_index_from_storage(storage_context)

    # get response for the question
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)#, response_mode="compact")
    #response = index.query(input_text, response_mode="compact")

    return response.response

# create index
#init_index(os.path.abspath("docs"))
init_index("docs")

# create ui interface to interact with gpt-3 model
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, placeholder="Enter your question here"),
                     outputs="text",
                     title="Nico AI: Know everything there is!",
                     description="Ask any question about Controlling")
                     #allow_screenshot=True)
iface.launch(share=True)