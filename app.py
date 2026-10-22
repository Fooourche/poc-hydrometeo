import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="POC Hydrométéo", layout="wide")
st.title("🌊 Assistant Hydrométéo (POC restreint)")

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

class RequeteHydro(BaseModel):
    lieux: List[str] = Field(description="Liste des départements, cours d'eau ou villes mentionnés")
    jours_historique: int = Field(default=3, description="Nombre de jours d'historique souhaités")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hubeau_data_cached(query_terms):
    headers = {"User-Agent": "Mozilla/5.0"}
    stations_meta = {}
    api_logs = [] # 🛠️ NOUVEAU : Liste pour tracer l'activité réseau
    
    for q in query_terms:
        if q.isdigit() or q in ['2A', '2B']:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?code_departement={q}&en_service=true"
        else:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?libelle_cours_eau={q}&en_service=true"
            
        api_logs.append(f"🔍 [RÉFÉRENTIEL] URL : {ref_url}")
        try:
            res_ref = requests.get(ref_url, headers=headers, timeout=10)
            api_logs.append(f"🔄 [RÉFÉRENTIEL] Code HTTP : {res_ref.status_code}")
            
            data_ref = res_ref.json()
            stations_trouvees = data_ref.get("data", [])
            api_logs.append(f"📍 [RÉFÉRENTIEL] Stations listées pour '{q}' : {len(stations_trouvees)}")
            
            for st_info in stations_trouvees:
                code = st_info.get("code_station") or st_info.get("code_site")
                if code and code not in stations_meta:
                    bv = st_info.get("surface_bassin_versant_topographique") or st_info.get("surface_bassin_versant_reel") or 0.0
                    stations_meta[code] = {
                        "nom_station": st_info.get("libelle_station", code),
                        "cours_eau": st_info.get("libelle_cours_eau", "Inconnu"),
                        "surface_bv": bv
                    }
        except Exception as e:
            api_logs.append(f"❌ [RÉFÉRENTIEL] Erreur : {e}")

    if not stations_meta:
        return pd.DataFrame(), api_logs

    station_codes = list(stations_meta.keys())
    all_obs = []
    chunk_size = 20
    
    for i in range(0, len(station_codes), chunk_size):
        chunk = station_codes[i:i + chunk_size]
        codes_param = ",".join(chunk)
        
        obs_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_entite={codes_param}&size=3000&grandeur_hydro=Q"
        api_logs.append(f"🌊 [OBSERVATIONS] URL Chunk : {obs_url}")
        
        pages_fetched = 0
        max_pages = 3 
        
        while obs_url and pages_fetched < max_pages:
            try:
                res_obs = requests.get(obs_url, headers=headers, timeout=15)
                api_logs.append(f"🔄 [OBSERVATIONS] Code HTTP (Page {pages_fetched+1}) : {res_obs.status_code}")
                
                data_obs = res_obs.json()
                obs_list = data_obs.get('data', [])
                api_logs.append(f"💧 [OBSERVATIONS] Lignes de débits récupérées : {len(obs_list)}")
                
                for x in obs_list:
                    code = x['code_entite']
                    meta = stations_meta.get(code, {})
                    
                    q_m3s = x['resultat_obs']
                    bv = meta.get('surface_bv', 0.0)
                    
                    q_sp = (q_m3s * 1000 / bv) if bv > 0 else None
                    bv_str = f"{bv} km²" if bv > 0 else "BV Inconnu"
                    label_complet = f"{meta.get('nom_station', code)} ({meta.get('cours_eau', '')}) - {bv_str}"

                    all_obs.append({
                        'date_heure': x['date_obs'], 
                        'debit_m3s': q_m3s, 
                        'debit_specifique': q_sp,
                        'code_station': code,
                        'station': meta.get('nom_station', code),
                        'cours_eau': meta.get('cours_eau', ''),
                        'surface_bv': bv,
                        'label_complet': label_complet
                    })
                
                obs_url = data_obs.get('next')
                if obs_url:
                    api_logs.append("➡️ [OBSERVATIONS] Pagination détectée, requête page suivante...")
                pages_fetched += 1
            except Exception as e:
                api_logs.append(f"❌ [OBSERVATIONS] Erreur : {e}")
                break 

    df = pd.DataFrame(all_obs)
    if not df.empty:
        df['date_heure'] = pd.to_datetime(df['date_heure'], utc=True)
        df = df.dropna(subset=['debit_m3s']) 
        df = df.sort_values('date_heure')
        
    return df, api_logs

def fetch_hubeau(query_str: str, status_box) -> pd.DataFrame:
    raw_terms = str(query_str).replace("et", ",").split(",")
    query_terms = [q.strip().upper() for q in raw_terms if q.strip()]
    
    status_box.write(f"🗺️ Interrogation Hub'Eau pour : `{', '.join(query_terms)}`...")
    
    # 🛠️ NOUVEAU : Récupération du dataframe et des logs
    df, api_logs = fetch_hubeau_data_cached(query_terms)
    st.session_state.api_logs = api_logs 
    
    if df.empty:
        return pd.DataFrame()
        
    status_box.write(f"📊 {len(df)} observations récupérées avec métadonnées.")
    
    df.to_sql("debits", con=st.session_state.engine, if_exists="replace", index=False)
    st.session_state.df_visu = df 
    
    return df

# --- INTERFACE UTILISATEUR CHAT ---

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex : Quelles sont les crues sur le Gard ?"):
    if not api_key:
        st.error("Ajoute ta clé API Gemini dans les Secrets ou la barre latérale.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        extract_prompt = ""
        raw_json = ""
        synth_prompt = ""
        st.session_state.api_logs = []
        
        with st.status("🔍 Traitement en cours...", expanded=True) as status:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                status.write("🧠 **Étape 1/4** : Analyse sémantique (Pydantic)...")
                
                extract_prompt = f"""
                Analyse la question de l'utilisateur. Extrais les lieux et estime la durée.
                ⚠️ RÈGLE VITALE : Si le lieu est un département, tu DOIS retourner UNIQUEMENT son NUMÉRO officiel en chaîne de caractères (ex: 'Gard' -> '30', 'Isère' -> '38', 'Corse' -> '2A', '2B').
                Tu dois OBLIGATOIREMENT répondre avec un objet JSON valide, selon cette structure exacte :
                {{
                    "lieux": ["30", "2A"], 
                    "jours_historique": entier
                }}
                Question : '{prompt}'
                """
                res_extract = llm.invoke([HumanMessage(content=extract_prompt)])
                raw_json = extract_text(res_extract).strip()
                
                cleaned_json = raw_json
                if cleaned_json.startswith("```json"):
                    cleaned_json = cleaned_json[7:-3].strip()
                elif cleaned_json.startswith("```"):
                    cleaned_json = cleaned_json[3:-3].strip()
                
                try:
                    data_brute = json.loads(cleaned_json)
                    params = RequeteHydro(**data_brute)
                    target = ",".join(params.lieux)
                    jours = params.jours_historique
                except Exception:
                    target = prompt
                    jours = 3 
                
                status.write(f"👉 Cible : `{target}` (Historique analysé : {jours} jours)")
                
                status.write("📊 **Étape 2/4** : Consultation de Hub'Eau...")
                df = fetch_hubeau(target, status_box=status)
                
                if df.empty:
                    output_text = f"Aucune donnée d'observation récente disponible sur Hub'Eau pour '{target}'."
                    status.update(label="⚠️ Aucune donnée trouvée", state="complete")
                else:
                    status.write("⚡ **Étape 3/4** : Traitement des données dans SQLite...")
                    stats = pd.read_sql(
                        "SELECT station, cours_eau, surface_bv, "
                        "MIN(debit_m3s) as min_m3s, MAX(debit_m3s) as max_m3s, AVG(debit_m3s) as moy_m3s "
                        "FROM debits GROUP BY station, cours_eau, surface_bv", 
                        con=st.session_state.engine
                    )
                    
                    stats_top5 = stats.nlargest(5, 'max_m3s') if not stats.empty and len(stats) > 5 else stats
                    
                    status.write("✍️ **Étape 4/4** : Synthèse par l'expert IA...")
                    synth_prompt = (
                        f"Tu es un expert hydrométéorologue. "
                        f"Question de l'utilisateur : {prompt}\n\n"
                        f"Données des stations (Top 5 des débits max uniquement) :\n{stats_top5.to_string(index=False)}\n\n"
                        f"Rédige une synthèse professionnelle courte et chiffrée. N'invente AUCUN débit."
                    )
                    res_synth = llm.invoke([HumanMessage(content=synth_prompt)])
                    output_text = extract_text(res_synth)
                    status.update(label="✅ Traitement terminé !", state="complete")
                    
            except Exception as e:
                output_text = f"Erreur de traitement : {e}"
                status.update(label="❌ Erreur de traitement", state="error")

        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.chat_message("assistant").write(output_text)

        try:
            if "df_visu" in st.session_state and not st.session_state.df_visu.empty:
                st.subheader("📊 Dynamique hydrologique observée")
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                colors = px.colors.qualitative.Plotly
                
                for idx, (label, df_station) in enumerate(st.session_state.df_visu.groupby('label_complet')):
                    color = colors[idx % len(colors)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_station['date_heure'], 
                            y=df_station['debit_m3s'], 
                            name=f"Q - {label}",
                            mode='lines',
                            line=dict(color=color),
                            legendgroup=label,
                            hovertemplate="<b>%{x}</b><br>Débit: %{y:.2f} m³/s<extra></extra>"
                        ),
                        secondary_y=False
                    )
                    
                    if df_station['debit_specifique'].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=df_station['date_heure'], 
                                y=df_station['debit_specifique'], 
                                name=f"Qsp - {label}",
                                mode='lines',
                                line=dict(color=color, dash='dot'),
                                legendgroup=label,
                                showlegend=False,
                                hovertemplate="<b>%{x}</b><br>Débit Spé: %{y:.2f} l/s/km²<extra></extra>"
                            ),
                            secondary_y=True
                        )

                fig.update_layout(
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                fig.update_yaxes(title_text="<b>Débit</b> (m³/s)", secondary_y=False)
                fig.update_yaxes(title_text="<b>Débit spécifique</b> (l/s/km²)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erreur graphique : {e}")

        # 🛠️ NOUVEAU : Bloc de transparence enrichi avec les logs réseau Hub'Eau
        with st.expander("🔍 Auditer le raisonnement et le réseau"):
            st.markdown("### 🌐 Journal Réseau Hub'Eau (Logs API)")
            if st.session_state.api_logs:
                for log in st.session_state.api_logs:
                    st.text(log)
            else:
                st.text("Aucun appel réseau effectué (ou données servies depuis le cache).")
                
            if extract_prompt:
                st.markdown("---")
                st.markdown("### 🧠 Traces LLM (Gemini)")
                st.markdown("**1. Requête d'extraction :**")
                st.code(extract_prompt, language="text")
                st.markdown("**2. Structure JSON retournée :**")
                st.code(raw_json, language="json")
                if synth_prompt:
                    st.markdown("**3. Synthèse demandée :**")
                    st.code(synth_prompt, language="text")
                    
