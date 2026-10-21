import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

st.set_page_config(page_title="POC Hydrométéo", layout="wide")
st.title("🌊 Assistant Hydrométéo (POC)")

# Récupération automatique de la clé API depuis Secrets ou l'environnement
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

# Panneau latéral de secours si la clé n'est pas dans les Secrets
if not api_key:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Clé API Gemini", type="password")

# Initialisation de SQLite en mémoire
if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite:///:memory:", echo=False)
    with st.session_state.engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS debits (
                date_heure TEXT,
                debit_m3s REAL,
                station TEXT,
                code_station TEXT
            )
        """))

# Outil d'interrogation Hub'Eau
@tool
def charger_donnees_hubeau(recherche: str) -> str:
    """
    Télécharge les débits récents depuis Hub'Eau et les insère dans la table 'debits' SQLite.
    'recherche' peut être un nom de cours d'eau (ex: 'Isère', 'Loire') ou un département (ex: '2A', '2B', '38').
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    query = recherche.strip()
    
    if query.isdigit() or query.upper() in ['2A', '2B']:
        url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_departement={query.upper()}&size=200&grandeur_hydro=Q"
    else:
        url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?libelle_cours_eau={query}&size=200&grandeur_hydro=Q"

    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        obs = [{
            'date_heure': x['date_obs'], 
            'debit_m3s': x['resultat_obs'], 
            'code_station': x['code_entite'],
            'station': x.get('nom_station', x['code_entite'])
        } for x in res.get('data', [])]
        
        if obs:
            df = pd.DataFrame(obs)
            df.to_sql("debits", con=st.session_state.engine, if_exists="replace", index=False)
            return f"Succès : {len(df)} observations chargées dans SQLite pour '{query}'."
        else:
            return f"Aucune donnée trouvée sur Hub'Eau pour '{query}'."
    except Exception as e:
        return f"Erreur Hub'Eau : {e}"

# Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Quels sont les débits récents en Corse (2A) ?"):
    if not api_key:
        st.error("Aucune clé API Gemini détectée. Configure les Secrets dans Streamlit Cloud.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Traitement en cours..."):
            try:
                db = SQLDatabase(st.session_state.engine)
                # Passage au modèle gemini-3.8-flash
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                prefix_prompt = (
                    "Tu es un expert hydrométéorologue. "
                    "1. Utilise systématiquement l'outil `charger_donnees_hubeau` pour charger le cours d'eau ou département demandé. "
                    "2. Interroge la table `debits` en SQL pour calculer les débits min, max et la tendance. "
                    "3. Rédige une synthèse concise."
                )
                
                agent = create_sql_agent(
                    llm=llm, 
                    db=db, 
                    extra_tools=[charger_donnees_hubeau],
                    agent_type="tool-calling", 
                    prefix=prefix_prompt, 
                    verbose=False
                )
                
                res = agent.invoke({"input": prompt})
                output_text = res["output"]
            except Exception as e:
                output_text = f"Erreur : {e}"

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)

        # Graphique
        try:
            df_visu = pd.read_sql("SELECT * FROM debits", con=st.session_state.engine)
            if not df_visu.empty:
                st.subheader("📊 Graphique des débits analysés")
                fig = px.line(df_visu, x='date_heure', y='debit_m3s', color='station',
                              labels={'date_heure': 'Date/Heure', 'debit_m3s': 'Débit (m³/s)'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
            
