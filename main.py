from ast import Assign
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from operator import itemgetter

from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components....")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model ="gpt-4o-mini")

vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"),embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only in the following context:
    {context}

    Question: {question}

    Provide a detail answer:""")

def format_docs (docs):
    "Format retrieved docs into a single string"
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """ Simple retrieval chain without LCEL
    Manually retrieves documents, format them and generates a response
    """

    #Step 1: Retrieve relevant documents.
    docs = retriever.invoke(query)

    #Step 2: Format documents
    context = format_docs(docs)

    #Step 3: Format the prompt passing the values needed.
    messages = prompt_template.format_messages(context = context, question = query)

    #Step 4: Invoke the ll with the messages.
    response = llm.invoke(messages)

    #Step 5: Return the content
    return response.content

def retrieval_chain_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")|retriever|format_docs
            )
        | prompt_template 
        | llm 
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print ("Retrieving...")

    #Query
    query = "What is pinecone in machine learning?"

    result = retrieval_chain_without_lcel(query)

    print ("RAG Implementation without LCEL")
    print ("-"*60)
    print (result)

    print ("RAG Implementation with LCEL")
    print ("-"*60)
    print (retrieval_chain_lcel().invoke({"question":query}))




