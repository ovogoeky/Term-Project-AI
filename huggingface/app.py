import gradio as gr
import pandas as pd
import joblib

# 1. Modell laden
model = joblib.load("mvp_rf.pkl")

# 2. Roh-Spalten, die in der Upload-CSV vorhanden sein müssen
RAW_COLS = [
    "Player", "Team",
    "PPG", "RPG", "APG", "Win%", "WS",
    "ORtg", "DRtg", "PER", "BPM", "VORP"
]

# 3. Modell-Features (abgeleitete)
FEATURES = [
    "Efficiency_score",
    "VORP",
    "WS",
    "BPM",
    "APG",
    "Standings",
    "Net_Rating",
    "DRtg",
    "PER"
]

def predict_df(df):
    # 3.1 Konsistente Spaltennamen: WinPct → Win%
    if "WinPct" in df.columns:
        df.rename(columns={"WinPct": "Win%"}, inplace=True)

    # 3.2 Spalten-Check
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    # 4. Abgeleitete Features berechnen
    df["Standings"]        = df["Win%"].rank(method="dense", ascending=False).astype(int)
    df["Net_Rating"]       = df["ORtg"] - df["DRtg"]
    df["Efficiency_score"] = (df["PPG"] + df["RPG"] + df["APG"]) * df["Win%"]

    # 5. Modellvorhersage
    X = df[FEATURES]
    df["MVP Probability"] = model.predict_proba(X)[:,1]

    # 6. Top-3 MVP-Kandidaten
    top3 = (
        df[["Player","MVP Probability"]]
        .sort_values("MVP Probability", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )
    top3["MVP Prediction"] = 0
    if not top3.empty:
        top3.at[0,"MVP Prediction"] = 1

    # 7. Zusammenfassung
    mvp, prob = top3.loc[0,"Player"], top3.loc[0,"MVP Probability"]
    summary = f"🏆 Vorhergesagter MVP: **{mvp}** ({prob:.2f})

    return top3, summary

def predict_top3(csv_file):
    df = pd.read_csv(csv_file.name)
    return predict_df(df)

def predict_default():
    df = pd.read_csv("24_25.csv")
    return predict_df(df)

# 8. Gradio-Interface
with gr.Blocks() as app:
    gr.Markdown("## NBA MVP Predictor – Top 3 🏀")

    with gr.Row():
        upload      = gr.File(label="Eigene CSV hochladen")
        default_btn = gr.Button("Mit den Daten der 24/25 Saison füllen")

    output_table = gr.Dataframe(label="Top-3 MVP-Kandidaten")
    output_md    = gr.Markdown(label="Vorhersage")

    upload.change(fn=predict_top3, inputs=upload, outputs=[output_table, output_md])
    default_btn.click(fn=predict_default, inputs=None, outputs=[output_table, output_md])

    gr.Markdown(
        "**Benötigte Spalten für die Saison-CSV (z.B. 25/26):**\n"
        "- Player, Team\n"
        "- PPG, RPG, APG, Win%, WS\n"
        "- ORtg, DRtg, PER, BPM, VORP\n\n"
        "Um das Tool für eine andere Saison (z.B. 25/26) zu nutzen, "
        "lade eine entsprechend formatierte CSV mit den obigen Spalten hoch."
    )

if __name__ == "__main__":
    app.launch()
