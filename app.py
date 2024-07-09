'''
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-03-09 18:00:15
 	@modify date 2024-03-17 18:55:26
 	@desc        Main file for PDF QnA Application
 '''

import os
import streamlit as st
from langchain_openai.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.set_page_config(page_title='ChatPDF')


def read_data(pdf_filepath):
    pdf_reader = PdfReader(pdf_filepath)
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text


def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def load_doc_search(texts, query):
    embeddings = OpenAIEmbeddings() 
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search.similarity_search(query)


def load_chain():
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.6)
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain


def get_response(chain, docs, question):
    response = chain.run(input_documents=docs, question=question)
    return response


st.subheader('**:green[Upload a file and Ask Questions]**')
uploaded_file = st.file_uploader(':blue[Choose your .pdf file]', type=['pdf'])
if uploaded_file is not None:
    question = st.text_input(':orange[What would you like to know about PDF?]', key='input')
    submit_btn = st.button('Ask the Question')
    if submit_btn:
        try:
            chain = load_chain()
            document = read_data(uploaded_file)
            texts = split_text(document)
            doc_search = load_doc_search(texts, question)
            response = get_response(chain, doc_search, question)
            st.write(response)
        except Exception as e:
            st.write(f'{e}: An error has occured!')
