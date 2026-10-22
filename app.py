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

# 🔒 BRIDE 1 : Historique par défaut de 3 jours
class RequeteHydro(BaseModel):
    lieux: List[str] = Field(description="Liste des départements, cours d'eau ou villes mentionnés")
    jours_historique: int = Field(default=3, description="Nombre de jours d'historique souhaités")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hubeau_data_cached(query_terms):
    headers = {"User-Agent": "Mozilla/5.0"}
    stations_meta = {}
    
    # 🔒 BRIDE 2 : Limite à 15 stations max par recherche
    for q in query_terms:
        if q.isdigit() or q in ['2A', '2B']:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?code_departement={q}&en_service=true&size=15"
        else:
            ref_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?libelle_cours_eau={q}&en_service=true&size=15"
            
        try:
            res_ref = requests.get(ref_url, headers=headers, timeout=10).json()
            for st_info in res_ref.get("data", []):
                code = st_info.get("code_station") or st_info.get("code_site")
                if code and code not in stations_meta:
                    bv = st_info.get("surface_bassin_versant_topographique") or st_info.get("surface_bassin_versant_reel") or 0.0
                    stations_meta[code] = {
                        "nom_station": st_info.get("libelle_station", code),
                        "cours_eau": st_info.get("libelle_cours_eau", "Inconnu"),
                        "surface_bv": bv
                    }
        except Exception:
            pass 

    if not stations_meta:
        return pd.DataFrame()

    station_codes = list(stations_meta.keys())
    all_obs = []
    chunk_size = 20
    
    for i in range(0, len(station_codes), chunk_size):
        chunk = station_codes[i:i + chunk_size]
        codes_param = ",".join(chunk)
        
        # 🔧 CORRECTION : size=3000 au lieu de 20000 pour éviter le timeout Hub'Eau
        obs_url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?code_entite={codes_param}&size=3000&grandeur_hydro=Q"
        
        pages_fetched = 0
        max_pages = 3 
        
        while obs_url and pages_fetched < max_pages:
            try:
                res_obs = requests.get(obs_url, headers=headers, timeout=15).json()
                
                for x in res_obs.get('data', []):
                    code = x['code_entite']
                    meta = stations_meta.get(code, {})
                    
                    q_m3s = x['resultat_obs']
                    bv = meta['surface_bv']
                    
                    q_sp = (q_m3s * 1000 / bv) if bv > 0 else None
                    bv_str = f"{bv} km²" if bv > 0 else "BV Inconnu"
                    label_complet = f"{meta['nom_station']} ({meta['cours_eau']}) - {bv_str}"

                    all_obs.append({
                        'date_heure': x['date_obs'], 
                        'debit_m3s': q_m3s, 
                        'debit_specifique': q_sp,
                        'code_station': code,
                        'station': meta['nom_station'],
                        'cours_eau': meta['cours_eau'],
                        'surface_bv': bv,
                        'label_complet': label_complet
                    })
                
                obs_url = res_obs.get('next')
                pages_fetched += 1
            except Exception:
                break 

    df = pd.DataFrame(all_obs)
    if not df.empty:
        df['date_heure'] = pd.to_datetime(df['date_heure'], utc=True)
        df = df.dropna(subset=['debit_m3s']) 
        df = df.sort_values('date_heure')
        
    return df

def fetch_hubeau(query_str: str, status_box) -> pd.DataFrame:
    raw_terms = str(query_str).replace("et", ",").split(",")
    query_terms = [q.strip().upper() for q in raw_terms if q.strip()]
    
    status_box.write(f"🗺️ Interrogation Hub'Eau pour : `{', '.join(query_terms)}`...")
    
    df = fetch_hubeau_data_cached(query_terms)
    
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
        
        with st.status("🔍 Traitement en cours...", expanded=True) as status:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0, 
                    google_api_key=api_key
                )
                
                status.write("🧠 **Étape 1/4** : Analyse sémantique (Pydantic)...")
                
                # 🔧 CORRECTION : Règle vitale pour forcer la conversion des départements en numéros
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
                    
                    # 🔒 BRIDE 3 : Filtre Top 5 pour soulager les jetons LLM
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

        # Graphique
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

        # Bloc de transparence pour auditer l'IA
        if extract_prompt:
            with st.expander("🔍 Auditer le raisonnement et les données de l'IA"):
                st.markdown("**1. Requête d'extraction temporelle et spatiale :**")
                st.code(extract_prompt, language="text")
                
                st.markdown("**2. Structure JSON retournée par le LLM :**")
                st.code(raw_json, language="json")
                
                st.markdown("**3. Synthèse demandée (avec les données Hub'Eau injectées) :**")
                st.code(synth_prompt, language="text")
                
