import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os

st.set_page_config(page_title="Policy QA", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
body {background-color: #0f1117; color: white;}
section.main > div {background-color: #1e1e1e; padding: 2em; border-radius: 1em;}
</style>
""", unsafe_allow_html=True)

st.title("🧐 Health Policy Query Enhancer")
query = st.text_input("Enter your query:")

if query:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", k=4),
        return_source_documents=True
    )
    structured_prompt = f"""
    Based on the following insurance policy documents, answer this query:

    Query: {query}

    Return the result as JSON:
    {{
      "decision": "approved/rejected",
      "amount": "₹amount or NA",
      "justification": "reason",
      "clause_reference": "Clause number"
    }}
    """
    response = qa_chain.invoke(structured_prompt)
    st.json(response["result"])
