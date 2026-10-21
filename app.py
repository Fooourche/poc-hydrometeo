import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="POC Hydrométéo", layout="wide")
st.title("🌊 Assistant Hydrométéo (POC)")

# Clé API Gemini
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Clé API Gemini", type="password")

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite:///:memory:", echo=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return "".join(parts)
    return str(content)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hubeau_data_cached(query_terms):
    """Fonction pure mise en cache pour interroger Hub'Eau"""
    headers = {"User-Agent": "Mozilla/5.0"}
    station_codes = []
    
    # 1. Référentiel
    for q in query_terms:
        if q.isdigit() or q in ['2A', '2B']:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?code_departement={q}&en_service=true&size=50"
        else:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?libelle_cours_eau={q}&en_service=true&size=50"
            
        try:
            res_ref = requests.get(ref_url, headers=headers, timeout=10).json()
            for st_info in res_ref.get("data", []):
                code = st_info.get("code_station") or st_info.get("code_site")
                if code and code not in station_codes:
                    station_codes.append(code)
        except Exception:
            pass 

    if not station_codes:
        return pd.DataFrame()

    # 2. Observations avec Pagination
    all_obs = []
    chunk_size = 20
    
    for i in range(0, len(station_codes), chunk_size):
        chunk = station_codes[i:i + chunk_size]
        codes_param = ",".join(chunk)
        obs_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_entite={codes_param}&size=20000&grandeur_hydro=Q"
        
        pages_fetched = 0
        max_pages = 3 
        
        while obs_url and pages_fetched < max_pages:
            try:
                res_obs = requests.get(obs_url, headers=headers, timeout=15).json()
                
                obs = [{
                    'date_heure': x['date_obs'], 
                    'debit_m3s': x['resultat_obs'], 
                    'code_station': x['code_entite'],
                    'station': x.get('nom_station', x['code_entite'])
                } for x in res_obs.get('data', [])]
                all_obs.extend(obs)
                
                obs_url = res_obs.get('next')
                pages_fetched += 1
            except Exception:
                break 

    # 3. Nettoyage Pandas
    df = pd.DataFrame(all_obs)
    if not df.empty:
        df['date_heure'] = pd.to_datetime(df['date_heure'], utc=True)
        df = df.dropna(subset=['debit_m3s']) 
        df = df.sort_values('date_heure')
        
    return df


def fetch_hubeau(query_str: str, status_box) -> pd.DataFrame:
    """Wrapper pour l'UI, SQLite et le Session State"""
    raw_terms = str(query_str).replace("et", ",").split(",")
    query_terms = [q.strip().upper() for q in raw_terms if q.strip()]
    
    status_box.write(f"🗺️ Interrogation Hub'Eau pour : `{', '.join(query_terms)}`...")
    
    df = fetch_hubeau_data_cached(query_terms)
    
    if df.empty:
        return pd.DataFrame()
        
    status_box.write(f"📊 {len(df)} observations récupérées et nettoyées.")
    
    # Injection dans SQLite et sauvegarde dans le state pour Plotly
    df.to_sql("debits", con=st.session_state.engine, if_exists="replace", index=False)
    st.session_state.df_visu = df 
    
    return df

# --- INTERFACE UTILISATEUR CHAT ---

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Quels sont les débits récents en Isère ?"):
    if not api_key:
        st.error("Ajoute ta clé API Gemini dans les Secrets ou la barre latérale.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.status("🔍 Traitement en cours...", expanded=True) as status:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                # Étape 1 : Extraction
                status.write("🧠 **Étape 1/4** : Identification des zones géographiques...")
                extract_prompt = (
                    f"Extrais uniquement les noms de cours d'eau, villes ou numéros de départements "
                    f"(ex: 2A, 2B, 38, Isère, Loire, Corse) séparés par une virgule depuis cette question. "
                    f"Si la question parle de 'Corse', réponds '2A, 2B'. "
                    f"Ne réponds QUE par les valeurs extraites : '{prompt}'"
                )
                res_extract = llm.invoke([HumanMessage(content=extract_prompt)])
                target = extract_text(res_extract).strip()
                status.write(f"👉 Recherche ciblée : `{target}`")
                
                # Étape 2 : Hub'Eau
                status.write("📊 **Étape 2/4** : Consultation du référentiel Hub'Eau...")
                df = fetch_hubeau(target, status_box=status)
                
                if df.empty:
                    output_text = f"Aucune donnée d'observation récente disponible sur Hub'Eau pour '{target}'."
                    status.update(label="⚠️ Aucune donnée trouvée", state="complete")
                else:
                    # Étape 3 : SQLite
                    status.write("⚡ **Étape 3/4** : Traitement des données dans SQLite...")
                    stats = pd.read_sql(
                        "SELECT station, MIN(debit_m3s) as min_m3s, MAX(debit_m3s) as max_m3s, AVG(debit_m3s) as moy_m3s "
                        "FROM debits GROUP BY station", 
                        con=st.session_state.engine
                    )
                    
                    # Étape 4 : Synthèse IA
                    status.write("✍️ **Étape 4/4** : Synthèse par l'expert IA...")
                    synth_prompt = (
                        f"Tu es un expert hydrométéorologue. "
                        f"Question : {prompt}\n\n"
                        f"Données réelles des stations Hub'Eau (SQLite) :\n"
                        f"{stats.to_string(index=False)}\n\n"
                        f"Rédige une synthèse professionnelle chiffrée avec les débits max/min observés."
                    )
                    res_synth = llm.invoke([HumanMessage(content=synth_prompt)])
                    output_text = extract_text(res_synth)
                    status.update(label="✅ Traitement terminé !", state="complete")
                    
            except Exception as e:
                output_text = f"Erreur de traitement : {e}"
                status.update(label="❌ Erreur de traitement", state="error")

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)

        # Rendu du graphique Plotly
        try:
            if "df_visu" in st.session_state and not st.session_state.df_visu.empty:
                st.subheader("📊 Débits observés par station")
                fig = px.line(st.session_state.df_visu, x='date_heure', y='debit_m3s', color='station',
                              labels={'date_heure': 'Date/Heure', 'debit_m3s': 'Débit (m³/s)'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")
            
