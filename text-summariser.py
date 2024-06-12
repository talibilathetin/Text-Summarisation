from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pypdf import PdfReader 



# this function is responsible for splitting the data into smaller chunks and convert the data in document format
def chunks_and_document(txt):
    
    text_splitter = CharacterTextSplitter() # text splitter method by default it has chunk_size = 200 and chunk_overlap = 200
    texts = text_splitter.split_text(txt) # split the text into smaller chunks
    docs = [Document(page_content=t) for t in texts] # convert the splitted chunks into document format
    
    return docs


# Loading the Llama 2's LLM
def load_llm():
    # We instantiate the callback with a streaming stdout handler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])   

    # loading the LLM model
    # This open source model can be downloaded from here
    # Their are multiple models available just replace it in place of model and try it.

    
    llm = CTransformers(model='/Users/talib/Documents/Projects/Summary/Text Summarisation/Test/llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens': 512,
                              'temperature': 0.01,
                              'context_length': 20000}
                      )
        
    return llm


# this functions is used for applying the llm model with our document 
def chains_and_response(docs):
    
    llm = load_llm()
    chain = load_summarize_chain(llm,chain_type='map_reduce')
    
    return chain.run(docs)


reader = PdfReader('/Users/talib/Documents/Projects/Summary/Text Summarisation/Data/Attention is all you need.pdf') 
print('Number of Pages:' ,len(reader.pages)) 


page = reader.pages[0] 
text = page.extract_text() 
print("Number of Tokens",len(text)) 


result = []
docs = chunks_and_document(text)
response = chains_and_response(docs)
result.append(response)

print("Length of Summary:", len(result))
print("Summary: ", result)



