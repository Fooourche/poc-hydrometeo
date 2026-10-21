import os
import requests
import pandas as pd
import duckdb
import plotly.express as px
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_mistralai import ChatMistralAI

st.set_page_config(page_title="POC Hydrométéo DTG", layout="wide")
st.title("🌊 Assistant Hydrométéo DTG (POC)")

# Panneau latéral pour la clé API
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Clé API Mistral", type="password")
if not api_key:
    api_key = os.environ.get("MISTRAL_API_KEY", "")

# Initialisation et chargement des données Hub'Eau dans DuckDB
@st.cache_data(ttl=600)
def load_hubeau_data():
    url = "https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_entite=W3020010&size=100&grandeur_hydro=Q"
    res = requests.get(url).json()
    obs = [{'date_heure': x['date_obs'], 'debit_m3s': x['resultat_obs'], 'station': 'Isere_Grenoble'} for x in res.get('data', [])]
    df = pd.DataFrame(obs)
    
    conn = duckdb.connect("hydrometeo.duckdb")
    conn.execute("CREATE OR REPLACE TABLE debits_isere AS SELECT * FROM df")
    conn.close()
    return df

df_hydro = load_hubeau_data()

# Affichage du graphique Plotly
st.subheader("📊 Débits récents de l'Isère à Grenoble (Hub'Eau)")
if not df_hydro.empty:
    fig = px.line(df_hydro, x='date_heure', y='debit_m3s', labels={'date_heure': 'Date/Heure', 'debit_m3s': 'Débit (m³/s)'})
    st.plotly_chart(fig, use_container_width=True)

# Interface de Chat avec l'Agent
st.subheader("💬 Poser une question à l'Agent IA")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex: Quel est le débit maximum récent et quelle est la tendance ?"):
    if not api_key:
        st.error("Veuillez saisir votre clé API Mistral dans la barre latérale à gauche.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("L'agent interroge DuckDB..."):
            try:
                db = SQLDatabase.from_uri("duckdb:///hydrometeo.duckdb")
                llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=api_key)
                prefix_prompt = "Tu es un expert hydrométéorologue DTG. Analyse la table debits_isere et réponds de manière concise et synthétique."
                agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", prefix=prefix_prompt, verbose=False)
                
                res = agent.invoke({"input": prompt})
                output_text = res["output"]
            except Exception as e:
                output_text = f"Erreur lors de l'exécution : {e}"

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)
      
