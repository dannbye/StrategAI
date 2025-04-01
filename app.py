import os
import streamlit as st
import tempfile
import sqlite3
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from streamlit_oauth import OAuth2Component
import json

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = "http://localhost:8501"

# === AUTHENTICATION ===
oauth2 = OAuth2Component(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    revoke_endpoint="https://oauth2.googleapis.com/revoke",
)

# === DATABASE ===
def init_db():
    conn = sqlite3.connect("strategai.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY)")
    cursor.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT, owner_email TEXT, created_at TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS project_users (project_id INTEGER, email TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS project_notes (project_id INTEGER, note TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS project_strategies (project_id INTEGER, brand TEXT, objective TEXT, strategy TEXT, created_at TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS project_chat (project_id INTEGER, sender TEXT, message TEXT, timestamp TIMESTAMP)")
    conn.commit()
    return conn

conn = init_db()
cursor = conn.cursor()

# === PAGE SETUP ===
st.set_page_config(page_title="StrategAI - Strategy Assistant", layout="wide")

# === AUTH ===
authorization_url = oauth2.get_authorization_url(
    redirect_uri=REDIRECT_URI,
    scope=["https://www.googleapis.com/auth/userinfo.email"],
    state="random_state_string",
    access_type="offline",
    prompt="consent",
)

if "token" not in st.session_state:
    st.markdown(f"[Login with Google]({authorization_url})")
    token = oauth2.get_access_token(redirect_uri=REDIRECT_URI)
    if token:
        st.session_state.token = token
else:
    st.success("Logged in with Google")
    user_info = oauth2.get_user_info(st.session_state.token["access_token"])
    user_email = user_info["email"]
    cursor.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (user_email,))
    conn.commit()

    embedding_model = OpenAIEmbeddings()
    documents = []

    st.markdown("""
    <style>
    body { background-color: black; color: #00FF00; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { background-color: #000; color: #00FF00; border: 2px solid #00FF00; }
    .stTextInput>div>div>input, .stTextArea textarea { background-color: #111; color: #0f0; }
    </style>
    """, unsafe_allow_html=True)

    st.title("StrategAI - Your 1980s Retro Strategy Sidekick")

    # === PROJECT DASHBOARD ===
    st.subheader("My Projects")
    cursor.execute("SELECT id, name FROM projects WHERE owner_email=?", (user_email,))
    my_projects = cursor.fetchall()
    for proj_id, name in my_projects:
        st.markdown(f"**{name}** (ID: {proj_id})")

    st.subheader("Shared With Me")
    cursor.execute("SELECT p.id, p.name FROM projects p JOIN project_users pu ON p.id = pu.project_id WHERE pu.email=? AND p.owner_email!=?", (user_email, user_email))
    shared_projects = cursor.fetchall()
    for proj_id, name in shared_projects:
        st.markdown(f"**{name}** (ID: {proj_id})")

    st.subheader("Create New Project")
    new_project_name = st.text_input("Project Name")
    if st.button("Create Project") and new_project_name:
        now = datetime.now().isoformat()
        cursor.execute("INSERT INTO projects (name, owner_email, created_at) VALUES (?, ?, ?)", (new_project_name, user_email, now))
        conn.commit()
        st.experimental_rerun()

    selected_project_id = st.text_input("Enter Project ID to Open")
    if selected_project_id:
        st.session_state.project_id = int(selected_project_id)

    if "project_id" in st.session_state:
        project_id = st.session_state.project_id
        st.markdown(f"### Working on Project ID: {project_id}")

        brand = st.text_input("Brand Name")
        objective = st.text_area("Brand Objective or Challenge")
        uploaded_files = st.file_uploader("Upload previous strategy PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                loader = PyPDFLoader(file)
                raw_docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = splitter.split_documents(raw_docs)
                documents.extend(docs)

        if documents:
            vectorstore = FAISS.from_documents(documents, embedding_model)
            retriever = vectorstore.as_retriever()
        else:
            retriever = None

        if st.button("Generate Strategy"):
            if brand and objective:
                llm = ChatOpenAI(temperature=0.5, model="gpt-4")
                search = DuckDuckGoSearchRun()
                tools = [Tool(name="Web Search", func=search.run, description="Search brands and trends")]
                agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)
                with st.spinner("StrategAI is thinking..."):
                    context = ""
                    if retriever:
                        context_result = retriever.get_relevant_documents(objective)
                        context = "\n".join([doc.page_content for doc in context_result])
                    prompt = f"""
You're a senior advertising strategist. Help develop a strategy for the brand \"{brand}\".
Objective: {objective}
Context: {context}

1. Target audience & insights
2. Strategic idea
3. Creative territories
4. Cultural hooks
"""
                    response = agent.run(prompt)
                    now = datetime.now().isoformat()
                    cursor.execute("INSERT INTO project_strategies (project_id, brand, objective, strategy, created_at) VALUES (?, ?, ?, ?, ?)", (project_id, brand, objective, response, now))
                    conn.commit()
                    st.markdown(response)
                    st.download_button("Download Strategy", response, file_name=f"{brand}_strategy.txt")

        # === NOTES ===
        st.subheader("StrategAI Notes")
        note_input = st.text_area("Type your insight fragments")
        if st.button("Save Note") and note_input:
            cursor.execute("INSERT INTO project_notes (project_id, note) VALUES (?, ?)", (project_id, note_input))
            conn.commit()

        if st.button("Summarise Notes"):
            cursor.execute("SELECT note FROM project_notes WHERE project_id=?", (project_id,))
            all_notes = "\n".join([row[0] for row in cursor.fetchall()])
            llm = ChatOpenAI(temperature=0.5, model="gpt-4")
            summary = llm.predict(f"Summarise and cluster these notes into themes:\n{all_notes}")
            st.markdown(summary)

        # === STRATEGY HISTORY ===
        st.subheader("Project Strategy History")
        cursor.execute("SELECT brand, objective, strategy, created_at FROM project_strategies WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        for brand, objective, strategy, ts in rows:
            st.markdown(f"**{brand}** - {objective} *(Created {ts})*")
            st.code(strategy, language="markdown")

        # === COLLABORATORS ===
        st.subheader("Invite Collaborators")
        invite_email = st.text_input("Email to invite to this project")
        if st.button("Add Collaborator") and invite_email:
            cursor.execute("INSERT INTO project_users (project_id, email) VALUES (?, ?)", (project_id, invite_email))
            conn.commit()
            st.success(f"{invite_email} added to project.")

        owner_email = cursor.execute("SELECT owner_email FROM projects WHERE id=?", (project_id,)).fetchone()[0]
        if user_email == owner_email:
            remove_email = st.text_input("Remove Collaborator Email")
            if st.button("Remove Collaborator") and remove_email:
                cursor.execute("DELETE FROM project_users WHERE project_id=? AND email=?", (project_id, remove_email))
                conn.commit()
                st.success(f"{remove_email} removed from project.")

        # === PROJECT CHAT ===
        st.subheader("Team Comments")
        chat_input = st.text_input("Leave a comment")
        if st.button("Post Comment") and chat_input:
            now = datetime.now().isoformat()
            cursor.execute("INSERT INTO project_chat (project_id, sender, message, timestamp) VALUES (?, ?, ?, ?)", (project_id, user_email, chat_input, now))
            conn.commit()

        cursor.execute("SELECT sender, message, timestamp FROM project_chat WHERE project_id=? ORDER BY timestamp DESC", (project_id,))
        chat_rows = cursor.fetchall()
        for sender, msg, ts in chat_rows:
            st.markdown(f"**{sender}** ({ts}): {msg}")
