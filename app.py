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

# 🔒 BRIDE 1 : Historique par défaut de 3 jours
class RequeteHydro(BaseModel):
    lieux: List[str] = Field(description="Liste des départements, cours d'eau ou villes mentionnés")
    jours_historique: int = Field(default=3, description="Nombre de jours d'historique souhaités")

# API Hydrométrie v2 de Hub'Eau (la v1 répond 403, vérifié le 02/11/2026).
# Débits (resultat_obs) en l/s ; une réponse paginée revient en HTTP 206.
HUBEAU = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
HTTP_OK = (200, 206)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hubeau_data_cached(query_terms, jours):
    # 🛡️ User-Agent explicite pour éviter les blocages antibot
    headers = {"User-Agent": "POC_Hydrometeo/1.0"}
    stations_meta = {}
    api_logs = []

    for q in query_terms:
        if q.isdigit() or q in ['2A', '2B']:
            filtre = f"code_departement={q}"
        else:
            filtre = f"libelle_cours_eau={q}"
        ref_url = f"{HUBEAU}/referentiel/stations?{filtre}&en_service=true"
        # La surface du bassin versant est dans le référentiel des sites, pas des stations
        sites_url = f"{HUBEAU}/referentiel/sites?{filtre}&fields=code_site,surface_bv"

        surfaces = {}
        try:
            res_sites = requests.get(sites_url, headers=headers, timeout=10)
            if res_sites.status_code in HTTP_OK:
                surfaces = {s["code_site"]: s.get("surface_bv") for s in res_sites.json().get("data", [])}
            api_logs.append(f"📐 [SITES] Surfaces de bassin lues pour '{q}' : {len(surfaces)} (HTTP {res_sites.status_code})")
        except Exception as e:
            api_logs.append(f"❌ [SITES] Erreur technique : {e}")

        api_logs.append(f"🔍 [RÉFÉRENTIEL] URL : {ref_url}")
        try:
            res_ref = requests.get(ref_url, headers=headers, timeout=10)
            api_logs.append(f"🔄 [RÉFÉRENTIEL] Code HTTP : {res_ref.status_code}")

            # 🛡️ On s'assure que Hub'Eau a accepté la requête (200, ou 206 si paginée) avant de lire le JSON
            if res_ref.status_code in HTTP_OK:
                data_ref = res_ref.json()
                stations_trouvees = data_ref.get("data", [])
                api_logs.append(f"📍 [RÉFÉRENTIEL] Stations listées pour '{q}' : {len(stations_trouvees)}")

                for st_info in stations_trouvees:
                    code = st_info.get("code_station")
                    if code and code not in stations_meta:
                        bv = surfaces.get(st_info.get("code_site")) or 0.0
                        stations_meta[code] = {
                            "nom_station": st_info.get("libelle_station", code),
                            "cours_eau": st_info.get("libelle_cours_eau", "Inconnu"),
                            "surface_bv": bv
                        }
            else:
                api_logs.append(f"❌ [RÉFÉRENTIEL] Requête bloquée par Hub'Eau (WAF/Cloudflare). Code erreur: {res_ref.status_code}")
        except Exception as e:
            api_logs.append(f"❌ [RÉFÉRENTIEL] Erreur technique : {e}")

    if not stations_meta:
        return pd.DataFrame(), api_logs

    station_codes = list(stations_meta.keys())
    all_obs = []
    chunk_size = 20
    debut = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=jours)).strftime("%Y-%m-%d")

    for i in range(0, len(station_codes), chunk_size):
        chunk = station_codes[i:i + chunk_size]
        codes_param = ",".join(chunk)

        # 🔒 BRIDE 2 : size=3000 pour éviter le timeout Hub'Eau
        obs_url = f"{HUBEAU}/observations_tr?code_entite={codes_param}&size=3000&grandeur_hydro=Q&date_debut_obs={debut}"
        api_logs.append(f"🌊 [OBSERVATIONS] URL Chunk : {obs_url}")
        
        pages_fetched = 0
        max_pages = 3 
        
        while obs_url and pages_fetched < max_pages:
            try:
                res_obs = requests.get(obs_url, headers=headers, timeout=15)
                api_logs.append(f"🔄 [OBSERVATIONS] Code HTTP (Page {pages_fetched+1}) : {res_obs.status_code}")
                
                # 🛡️ Vérification HTTP (200, ou 206 si paginée) pour les observations
                if res_obs.status_code in HTTP_OK:
                    data_obs = res_obs.json()
                    obs_list = data_obs.get('data', [])
                    api_logs.append(f"💧 [OBSERVATIONS] Lignes de débits récupérées : {len(obs_list)}")

                    for x in obs_list:
                        code = x['code_station']
                        meta = stations_meta.get(code, {})

                        # Hub'Eau donne les débits en l/s
                        q_m3s = x['resultat_obs'] / 1000 if x['resultat_obs'] is not None else None
                        bv = meta.get('surface_bv', 0.0)
                        
                        q_sp = (q_m3s * 1000 / bv) if bv > 0 and q_m3s is not None else None
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
                    pages_fetched += 1
                    if obs_url and pages_fetched < max_pages:
                        api_logs.append("➡️ [OBSERVATIONS] Pagination détectée, requête page suivante...")
                    elif obs_url:
                        api_logs.append(f"⚠️ [OBSERVATIONS] Limite de {max_pages} pages atteinte : les observations les plus anciennes de ce groupe de stations sont tronquées.")
                else:
                    api_logs.append(f"❌ [OBSERVATIONS] Requête bloquée par Hub'Eau. Code erreur: {res_obs.status_code}")
                    break # On arrête la pagination si on est bloqué
            except Exception as e:
                api_logs.append(f"❌ [OBSERVATIONS] Erreur technique : {e}")
                break 

    df = pd.DataFrame(all_obs)
    if not df.empty:
        df['date_heure'] = pd.to_datetime(df['date_heure'], utc=True)
        df = df.dropna(subset=['debit_m3s']) 
        df = df.sort_values('date_heure')
        
    return df, api_logs

def fetch_hubeau(query_str: str, jours: int, status_box) -> pd.DataFrame:
    raw_terms = str(query_str).replace("et", ",").split(",")
    query_terms = [q.strip().upper() for q in raw_terms if q.strip()]
    
    status_box.write(f"🗺️ Interrogation Hub'Eau pour : `{', '.join(query_terms)}`...")
    
    df, api_logs = fetch_hubeau_data_cached(query_terms, jours)
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
                df = fetch_hubeau(target, jours, status_box=status)
                
                if df.empty:
                    output_text = f"Aucune donnée d'observation récente disponible sur Hub'Eau pour '{target}'. Consultez les logs réseau ci-dessous pour vérifier si l'API a bloqué la requête."
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

        # Bloc de transparence
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
                    
