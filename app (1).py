import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os

st.set_page_config(page_title="Policy QA", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Set full background to dark gradient */
body {
    background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
    color: white;
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    padding: 0;
    height: 100vh;
}

/* Main section container */
section.main {
    background-color: transparent;
}

/* Centered main content box */
section.main > div {
    background: rgba(255, 255, 255, 0.05);
    padding: 2em;
    border-radius: 1em;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

/* Stylish gradient title */
h1 {
    background: linear-gradient(135deg, #ff6ec4, #7873f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 900;
    text-align: center;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Input box styling */
.stTextInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid #aaa;
    border-radius: 10px;
    padding: 0.75em;
    color: white;
    font-size: 1.1em;
}

.stTextInput label {
    font-weight: 600;
    color: #ddd;
    font-size: 1em;
    margin-bottom: 0.5em;
}
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
