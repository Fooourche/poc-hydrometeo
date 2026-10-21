import os
import requests
import pandas as pd
import duckdb
import plotly.express as px
import streamlit as st
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="POC Hydrométéo Autonome", layout="wide")
st.title("🌊 Assistant Hydrométéo Autonome (POC)")

# Clé API Gemini
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Clé API Gemini", type="password")
if not api_key:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

# Initialisation de la base DuckDB au démarrage
conn = duckdb.connect("hydrometeo.duckdb")
conn.execute("CREATE TABLE IF NOT EXISTS debits_france (date_heure VARCHAR, debit_m3s DOUBLE, station VARCHAR, code_station VARCHAR)")
conn.close()

# Définition de l'outil Hub'Eau utilisable par le LLM
@tool
def charger_donnees_hubeau(recherche: str) -> str:
    """
    Interroge l'API Hub'Eau pour récupérer les débits récents et les charger dans DuckDB.
    'recherche' peut être un nom de cours d'eau (ex: 'Isère', 'Loire') ou un numéro de département (ex: '2A', '2B', '38').
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
            conn = duckdb.connect("hydrometeo.duckdb")
            conn.execute("CREATE OR REPLACE TABLE debits_france AS SELECT * FROM df")
            conn.close()
            return f"Succès : {len(df)} observations chargées dans DuckDB pour '{query}'."
        else:
            return f"Aucune donnée trouvée sur Hub'Eau pour '{query}'."
    except Exception as e:
        return f"Erreur lors de la récupération des données Hub'Eau : {e}"

# Interface de Chat
st.subheader("💬 Poser une question à l'Agent IA")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Y a-t-il des débits forts en Corse (2A) ou sur la Loire ?"):
    if not api_key:
        st.error("Renseigne ta clé API Gemini dans le panneau de gauche.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("L'agent réfléchit et va chercher les données nécessaires..."):
            try:
                db = SQLDatabase.from_uri("duckdb:///hydrometeo.duckdb", include_tables=['debits_france'])
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
                
                prefix_prompt = (
                    "Tu es un expert hydrométéorologue. "
                    "Avant d'interroger la table `debits_france`, utilise systématiquement l'outil `charger_donnees_hubeau` "
                    "pour aller chercher les données correspondant au cours d'eau ou au département mentionné par l'utilisateur. "
                    "Ensuite, fais la requête SQL sur DuckDB et rédige une synthèse concise."
                )
                
                # On fournit l'outil Hub'Eau en plus des outils SQL standards
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
                output_text = f"Erreur d'exécution : {e}"

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)
        
        # Affichage dynamique du graphique si des données sont présentes
        conn = duckdb.connect("hydrometeo.duckdb")
        df_visu = conn.execute("SELECT * FROM debits_france").df()
        conn.close()
        if not df_visu.empty:
            st.subheader("📊 Graphique des données analysées par l'Agent")
            fig = px.line(df_visu, x='date_heure', y='debit_m3s', color='station', labels={'date_heure': 'Date/Heure', 'debit_m3s': 'Débit (m³/s)'})
            st.plotly_chart(fig, use_container_width=True)
            
