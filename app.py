import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="POC Hydrométéo", layout="wide")
st.title("🌊 Assistant Hydrométéo (POC)")

# Clé API depuis Secrets ou variable d'environnement
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Clé API Gemini", type="password")

# SQLite en mémoire
if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite:///:memory:", echo=False)

def fetch_hubeau(query_str: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    q = query_str.strip()
    
    if q.isdigit() or q.upper() in ['2A', '2B']:
        url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_departement={q.upper()}&size=200&grandeur_hydro=Q"
    else:
        url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?libelle_cours_eau={q}&size=200&grandeur_hydro=Q"
    
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
            return df
    except Exception:
        pass
    return pd.DataFrame()

# Historique du Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Quels sont les débits récents en Corse (2A) ?"):
    if not api_key:
        st.error("Ajoute ta clé API Gemini dans les Secrets ou le panneau latéral.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Analyse et récupération des données..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                # Étape 1 : Gemini extrait la zone / cours d'eau demandé
                extract_prompt = (
                    f"Extrais uniquement le nom du cours d'eau ou le numéro de département "
                    f"(ex: 2A, 2B, 38, Isère, Loire) depuis cette question. "
                    f"Ne réponds QUE par la chaîne extraite, sans aucun autre mot : '{prompt}'"
                )
                target = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()
                
                # Étape 2 : Appel Hub'Eau et stockage SQLite
                df = fetch_hubeau(target)
                
                if df.empty:
                    output_text = f"Aucune donnée d'observation trouvée sur Hub'Eau pour la recherche '{target}'."
                else:
                    # Étape 3 : Calcul des stats SQL et rédaction de la synthèse
                    stats = pd.read_sql(
                        "SELECT station, MIN(debit_m3s) as min_m3s, MAX(debit_m3s) as max_m3s, AVG(debit_m3s) as moy_m3s "
                        "FROM debits GROUP BY station", 
                        con=st.session_state.engine
                    )
                    
                    synth_prompt = (
                        f"Tu es un expert hydrométéorologue. "
                        f"Question de l'utilisateur : {prompt}\n\n"
                        f"Voici les données réelles Hub'Eau résumées depuis SQLite :\n"
                        f"{stats.to_string(index=False)}\n\n"
                        f"Rédige une synthèse courte, chiffrée et professionnelle."
                    )
                    output_text = llm.invoke([HumanMessage(content=synth_prompt)]).content
                    
            except Exception as e:
                output_text = f"Erreur : {e}"

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)

        # Affichage du graphique Plotly
        try:
            df_visu = pd.read_sql("SELECT * FROM debits", con=st.session_state.engine)
            if not df_visu.empty:
                st.subheader("📊 Graphique des débits analysés")
                fig = px.line(df_visu, x='date_heure', y='debit_m3s', color='station',
                              labels={'date_heure': 'Date/Heure', 'debit_m3s': 'Débit (m³/s)'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
            
