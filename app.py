import os
import tempfile
import streamlit as st
import asyncio

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from chainlit.types import AskFileResponse

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
docs = []  # Define docs globally to make it accessible in other functions

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def process_file(file: AskFileResponse):
    global docs  # Use the global docs variable
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file.read())
        temp_file.flush()
        loader = Loader(temp_file.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)  # Use local docs variable
    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch

async def main():
    st.title("TheAIMart(PATBot)")

    files = st.file_uploader("Upload a PDF or text file", type=["txt", "pdf"], accept_multiple_files=False, key="file")
    if files is not None:
        file = files
        st.info(f"Processing `{file.name}`...")

        docsearch = get_docsearch(file)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(temperature=0, streaming=True),
            chain_type="stuff",
            retriever=docsearch.as_retriever(max_tokens_limit=4097),
        )

        st.success(f"`{file.name}` processed. You can now ask questions!")

        question = st.text_input("Ask a question:")
        if st.button("Submit"):
            with st.spinner("Searching for the answer..."):
                res = await asyncio.gather(chain.acall(question))
                answer = res[0]["answer"] if "answer" in res[0] else "No answer found"
                sources = res[0]["sources"].strip() if "sources" in res[0] else ""

                if sources:
                    all_sources = [doc.metadata["source"] for doc in docs]
                    found_sources = []
                    for source in sources.split(","):
                        source_name = source.strip().replace(".", "")
                        try:
                            index = all_sources.index(source_name)
                        except ValueError:
                            continue
                        text = docs[index].page_content
                        found_sources.append(text)
                    if found_sources:
                        st.info("Sources:")
                        for source_text in found_sources:
                            st.write(source_text)
                    else:
                        st.warning("No sources found")

                st.write("Answer:")
                st.write(answer)

if __name__ == "__main__":
    asyncio.run(main())
