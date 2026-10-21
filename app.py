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

# Clé API
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Clé API Gemini", type="password")

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite:///:memory:", echo=False)

def fetch_hubeau(query_str: str, status_box) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Nettoyage et séparation des cibles (ex: "2A, 2B")
    raw_terms = query_str.replace("et", ",").split(",")
    query_terms = [q.strip().upper() for q in raw_terms if q.strip()]
    
    all_obs = []
    for q in query_terms:
        status_box.write(f"🌐 Interrogation Hub'Eau pour : `{q}`...")
        print(f"[LOG SERVER] Requete Hub'Eau envoyee pour : {q}")
        
        if q.isdigit() or q in ['2A', '2B']:
            url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_departement={q}&size=100&grandeur_hydro=Q"
        else:
            url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?libelle_cours_eau={q}&size=100&grandeur_hydro=Q"
        
        try:
            res = requests.get(url, headers=headers, timeout=8).json()
            obs = [{
                'date_heure': x['date_obs'], 
                'debit_m3s': x['resultat_obs'], 
                'code_station': x['code_entite'],
                'station': x.get('nom_station', x['code_entite'])
            } for x in res.get('data', [])]
            all_obs.extend(obs)
            status_box.write(f"✅ `{q}` : {len(obs)} observations reçues.")
            print(f"[LOG SERVER] `{q}` -> {len(obs)} observations")
        except Exception as e:
            status_box.write(f"⚠️ Erreur Hub'Eau sur `{q}` : {e}")
            print(f"[LOG SERVER] Erreur sur {q} : {e}")

    if all_obs:
        df = pd.DataFrame(all_obs)
        df.to_sql("debits", con=st.session_state.engine, if_exists="replace", index=False)
        return df
    return pd.DataFrame()

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Quels sont les débits récents en 2A et 2B ?"):
    if not api_key:
        st.error("Ajoute ta clé API Gemini dans les Secrets.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Bloc d'avancement dynamique sur l'écran
        with st.status("🔍 Traitement en cours...", expanded=True) as status:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                # Étape 1
                status.write("🧠 **Étape 1/4** : Extraction des zones par Gemini...")
                print("[LOG SERVER] Etape 1: Extraction des entites...")
                extract_prompt = (
                    f"Extrais uniquement les noms de cours d'eau ou numéros de départements "
                    f"(ex: 2A, 2B, 38, Isère, Loire) séparés par une virgule depuis cette question. "
                    f"Ne réponds QUE par les valeurs extraites séparées par une virgule : '{prompt}'"
                )
                target = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()
                status.write(f"👉 Zones extraites : `{target}`")
                print(f"[LOG SERVER] Cibles extraites : {target}")
                
                # Étape 2
                status.write("📊 **Étape 2/4** : Téléchargement des données Hub'Eau...")
                df = fetch_hubeau(target, status_box=status)
                
                if df.empty:
                    output_text = f"Aucune donnée trouvée sur Hub'Eau pour la recherche '{target}'."
                    status.update(label="⚠️ Aucune donnée trouvée", state="complete")
                else:
                    # Étape 3
                    status.write("⚡ **Étape 3/4** : Calculs statistiques dans SQLite...")
                    print("[LOG SERVER] Etape 3: Requete SQL SQLite...")
                    stats = pd.read_sql(
                        "SELECT station, MIN(debit_m3s) as min_m3s, MAX(debit_m3s) as max_m3s, AVG(debit_m3s) as moy_m3s "
                        "FROM debits GROUP BY station", 
                        con=st.session_state.engine
                    )
                    
                    # Étape 4
                    status.write("✍️ **Étape 4/4** : Génération de la synthèse métier...")
                    print("[LOG SERVER] Etape 4: Synthese Gemini...")
                    synth_prompt = (
                        f"Tu es un expert hydrométéorologue. "
                        f"Question : {prompt}\n\n"
                        f"Données réelles Hub'Eau (SQLite) :\n"
                        f"{stats.to_string(index=False)}\n\n"
                        f"Rédige une synthèse courte et chiffrée."
                    )
                    output_text = llm.invoke([HumanMessage(content=synth_prompt)]).content
                    status.update(label="✅ Traitement terminé !", state="complete")
                    
            except Exception as e:
                output_text = f"Erreur de traitement : {e}"
                status.update(label="❌ Erreur de traitement", state="error")
                print(f"[LOG SERVER] Erreur globale : {e}")

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
            
