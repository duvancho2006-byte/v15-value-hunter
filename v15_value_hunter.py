
"""
V15 — VALUE HUNTER
Bot experimental de análisis estadístico de fútbol.

Objetivo:
- Entrenar únicamente con información disponible antes de cada partido.
- Estimar goles mediante un modelo Poisson con ataque/defensa y ajuste de forma.
- Convertir las expectativas en probabilidades de Over/Under y BTTS.
- Comparar probabilidad del modelo con cuotas históricas.
- Medir Value Gap, EV, ROI, Brier Score y drawdown.
- Hacer un test fuera de muestra del 15 al 28 de agosto de 2026.

Fuente principal:
https://www.football-data.co.uk/
La fuente publica resultados FT/HT, estadísticas de partido y cuotas; desde 2019/20
incluye conjuntos de cuotas de apertura y cierre.

NOTA:
Este bot es un sistema de investigación, NO garantiza rentabilidad.
No utiliza resultados futuros para generar una predicción.
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import poisson
except ImportError as e:
    raise SystemExit("Instala scipy: pip install scipy") from e

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------

CUTOFF = pd.Timestamp("2026-08-14")
TEST_START = pd.Timestamp("2026-08-15")
TEST_END = pd.Timestamp("2026-08-28")

# Ligas europeas principales + segundas divisiones con buena cobertura histórica.
LEAGUES = {
    "ENG_E0": ("England", "E0"),
    "ENG_E1": ("England", "E1"),
    "ESP_SP1": ("Spain", "SP1"),
    "ESP_SP2": ("Spain", "SP2"),
    "ITA_I1": ("Italy", "I1"),
    "ITA_I2": ("Italy", "I2"),
    "GER_D1": ("Germany", "D1"),
    "GER_D2": ("Germany", "D2"),
    "FRA_F1": ("France", "F1"),
    "FRA_F2": ("France", "F2"),
    "NED_N1": ("Netherlands", "N1"),
    "BEL_B1": ("Belgium", "B1"),
    "POR_P1": ("Portugal", "P1"),
    "TUR_T1": ("Turkey", "T1"),
    "SCO_SC0": ("Scotland", "SC0"),
}

SEASON_CODES = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2020, 2026)]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"

# Filtro conservador para evitar apuestas a cuotas extremas.
MIN_ODDS = 1.35
MAX_ODDS = 3.50
MIN_VALUE_GAP = 0.05
MIN_MODEL_PROB = 0.58

# EWMA: partidos recientes pesan más.
FORM_HALFLIFE = 8.0
TEAM_HISTORY = 80

# ---------------- HELPERS ----------------

def season_codes_until_cutoff():
    """Temporadas completas y la actual, sin usar resultados posteriores al corte."""
    return SEASON_CODES + ["2627"]


def download_league(country, division, season):
    url = BASE_URL.format(season=season, division=division)
    try:
        df = pd.read_csv(url, encoding="cp1252")
    except Exception:
        return pd.DataFrame()

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["country"] = country
    df["division"] = division
    df["season_code"] = season
    return df


def load_data(include_test=True):
    """
    Carga la base histórica de V15.

    include_test=True:
        Conserva los partidos del período de prueba ciega.

    include_test=False:
        Devuelve únicamente información disponible antes del corte.
    """
    frames = []

    for _, (country, division) in LEAGUES.items():
        for season in season_codes_until_cutoff():
            df = download_league(country, division, season)

            if not df.empty:
                frames.append(df)

    if not frames:
        raise RuntimeError(
            "No se pudieron descargar datos. Comprueba conexión a internet "
            "y que football-data.co.uk esté accesible."
        )

    data = pd.concat(frames, ignore_index=True, sort=False)

    data = data.dropna(
        subset=["Date", "HomeTeam", "AwayTeam"]
    )

    # Orden cronológico antes de cualquier cálculo.
    data = data.sort_values(
        ["Date", "country", "division"]
    ).reset_index(drop=True)

    if not include_test:
        data = data[data["Date"] <= CUTOFF].copy()
        data = data.reset_index(drop=True)

    return data


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_odds_columns(df):
    """
    Busca columnas de Over/Under 2.5 en diferentes versiones de los CSV.
    """
    over25 = first_existing(df, [
        "B365>2.5", "P>2.5", "Max>2.5", "Avg>2.5"
    ])
    under25 = first_existing(df, [
        "B365<2.5", "P<2.5", "Max<2.5", "Avg<2.5"
    ])

    # Bet365 1X2 como fallback para mercados de resultado.
    home = first_existing(df, ["B365H", "PSH", "MaxH", "AvgH"])
    draw = first_existing(df, ["B365D", "PSD", "MaxD", "AvgD"])
    away = first_existing(df, ["B365A", "PSA", "MaxA", "AvgA"])

    return {
        "over25": over25,
        "under25": under25,
        "home": home,
        "draw": draw,
        "away": away,
    }


def safe_mean(values, default):
    values = [x for x in values if pd.notna(x)]
    return float(np.mean(values)) if values else float(default)


def ewma(values, half_life=FORM_HALFLIFE):
    if not values:
        return np.nan
    alpha = 1 - math.exp(math.log(0.5) / half_life)
    s = None
    for x in values:
        s = x if s is None else alpha * x + (1 - alpha) * s
    return float(s)


# ---------------- TEAM MODEL ----------------

@dataclass
class TeamStats:
    home_attack: float
    away_attack: float
    home_defence: float
    away_defence: float
    home_games: int
    away_games: int


def team_history(data, team, date, league_key=None):
    """
    Solo devuelve partidos estrictamente anteriores a date.
    """
    h = data[
        (data["HomeTeam"] == team) &
        (data["Date"] < date)
    ].copy()

    a = data[
        (data["AwayTeam"] == team) &
        (data["Date"] < date)
    ].copy()

    if league_key is not None:
        country, division = league_key
        h = h[(h["country"] == country) & (h["division"] == division)]
        a = a[(a["country"] == country) & (a["division"] == division)]

    return h.tail(TEAM_HISTORY), a.tail(TEAM_HISTORY)


def league_baseline(data, date, country, division):
    d = data[
        (data["Date"] < date) &
        (data["country"] == country) &
        (data["division"] == division)
    ].dropna(subset=["FTHG", "FTAG"])

    # Ventana reciente para no congelar el modelo en una época demasiado antigua.
    d = d.tail(250)

    if len(d) < 20:
        # fallback amplio
        d = data[
            (data["Date"] < date) &
            (data["country"] == country) &
            (data["division"] == division)
        ].dropna(subset=["FTHG", "FTAG"])

    if len(d) == 0:
        return 1.35, 1.10

    return max(0.50, float(d["FTHG"].mean())), max(0.35, float(d["FTAG"].mean()))


def estimate_team_strength(data, team, date, country, division):
    h, a = team_history(data, team, date, (country, division))
    home_avg, away_avg = league_baseline(data, date, country, division)

    if len(h):
        scored_home = ewma(h["FTHG"].astype(float).tolist())
        conceded_home = ewma(h["FTAG"].astype(float).tolist())
    else:
        scored_home = conceded_home = np.nan

    if len(a):
        scored_away = ewma(a["FTAG"].astype(float).tolist())
        conceded_away = ewma(a["FTHG"].astype(float).tolist())
    else:
        scored_away = conceded_away = np.nan

    # Shrinkage hacia media de liga para muestras pequeñas.
    n_h, n_a = len(h), len(a)
    wh = min(1.0, n_h / 10.0)
    wa = min(1.0, n_a / 10.0)

    home_attack = wh * (scored_home / home_avg if pd.notna(scored_home) else 1.0) + (1-wh)
    home_def = wh * (conceded_home / away_avg if pd.notna(conceded_home) else 1.0) + (1-wh)

    away_attack = wa * (scored_away / away_avg if pd.notna(scored_away) else 1.0) + (1-wa)
    away_def = wa * (conceded_away / home_avg if pd.notna(conceded_away) else 1.0) + (1-wa)

    return TeamStats(
        home_attack=float(home_attack),
        away_attack=float(away_attack),
        home_defence=float(home_def),
        away_defence=float(away_def),
        home_games=n_h,
        away_games=n_a,
    )


def expected_goals(data, date, home, away, country, division):
    league_home, league_away = league_baseline(data, date, country, division)

    hs = estimate_team_strength(data, home, date, country, division)
    aws = estimate_team_strength(data, away, date, country, division)

    # Ataque local × defensa visitante.
    lam_home = league_home * hs.home_attack * aws.away_defence

    # Ataque visitante × defensa local.
    lam_away = league_away * aws.away_attack * hs.home_defence

    # Pequeño ajuste de regresión hacia la media en muestras muy pequeñas.
    total_games = hs.home_games + hs.away_games + aws.home_games + aws.away_games
    if total_games < 8:
        lam_home = 0.65 * lam_home + 0.35 * league_home
        lam_away = 0.65 * lam_away + 0.35 * league_away

    return float(np.clip(lam_home, 0.15, 4.0)), float(np.clip(lam_away, 0.10, 3.5))


# ---------------- GOAL PROBABILITIES ----------------

def goal_matrix(lam_home, lam_away, max_goals=8):
    h = np.array([poisson.pmf(i, lam_home) for i in range(max_goals + 1)])
    a = np.array([poisson.pmf(i, lam_away) for i in range(max_goals + 1)])
    return np.outer(h, a)


def market_probabilities(lam_home, lam_away):
    m = goal_matrix(lam_home, lam_away)
    p_over_15 = 1 - m[0, 0] - m[0, 1] - m[1, 0]
    # P(total <= 2)
    p_under_25 = sum(m[i, j] for i in range(9) for j in range(9) if i + j <= 2)
    p_over_25 = 1 - p_under_25
    p_btts = 1 - sum(m[0, :]) - sum(m[:, 0]) + m[0, 0]
    p_home_goal = 1 - sum(m[0, :])
    p_away_goal = 1 - sum(m[:, 0])

    return {
        "over_1_5": float(p_over_15),
        "over_2_5": float(p_over_25),
        "under_2_5": float(p_under_25),
        "btts": float(p_btts),
        "home_over_0_5": float(p_home_goal),
        "away_over_0_5": float(p_away_goal),
    }


def implied_probability(odds):
    if pd.isna(odds) or odds <= 1:
        return np.nan
    return 1.0 / float(odds)


def expected_value(prob, odds):
    if pd.isna(odds):
        return np.nan
    return prob * odds - 1


def value_score(prob, odds):
    if pd.isna(odds) or odds <= 1:
        return np.nan

    imp = implied_probability(odds)
    gap = prob - imp

    # Score 50 = neutral. Gap de 10 puntos = ~80.
    score = 50 + 300 * gap

    # Penaliza cuotas extremas.
    if odds < MIN_ODDS or odds > MAX_ODDS:
        score -= 10

    return float(np.clip(score, 0, 100))


# ---------------- SCAN ----------------

def prediction_for_match(data, row):
    date = row["Date"]
    country = row["country"]
    division = row["division"]
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    lh, la = expected_goals(
        data, date, home, away, country, division
    )

    probs = market_probabilities(lh, la)
    odds = normalize_odds_columns(pd.DataFrame([row]))

    out = {
        "date": date,
        "league": f"{country} {division}",
        "home": home,
        "away": away,
        "lambda_home": lh,
        "lambda_away": la,
    }

    # Los mercados disponibles dependen del CSV.
    if odds["over25"]:
        o25 = float(row[odds["over25"]]) if pd.notna(row[odds["over25"]]) else np.nan
        out["o25_odds"] = o25
        out["o25_prob"] = probs["over_2_5"]
        out["o25_gap"] = probs["over_2_5"] - implied_probability(o25)
        out["o25_ev"] = expected_value(probs["over_2_5"], o25)
        out["o25_score"] = value_score(probs["over_2_5"], o25)

    if odds["under25"]:
        u25 = float(row[odds["under25"]]) if pd.notna(row[odds["under25"]]) else np.nan
        out["u25_odds"] = u25
        out["u25_prob"] = probs["under_2_5"]
        out["u25_gap"] = probs["under_2_5"] - implied_probability(u25)
        out["u25_ev"] = expected_value(probs["under_2_5"], u25)
        out["u25_score"] = value_score(probs["under_2_5"], u25)

    # Resultado real — solo para evaluar después.
    if pd.notna(row.get("FTHG", np.nan)) and pd.notna(row.get("FTAG", np.nan)):
        total = row["FTHG"] + row["FTAG"]
        out["result_over25"] = int(total > 2.5)
        out["result_under25"] = int(total < 2.5)
        out["result_btts"] = int(row["FTHG"] > 0 and row["FTAG"] > 0)

    return out


def walk_forward_test(data):
    """
    Genera predicciones partido por partido.
    Cada partido solo puede usar información con fecha anterior.
    """
    test = data[
        (data["Date"] >= TEST_START) &
        (data["Date"] <= TEST_END) &
        data["FTHG"].notna() &
        data["FTAG"].notna()
    ].copy()

    predictions = []
    for _, row in test.iterrows():
        predictions.append(prediction_for_match(data, row))

    return pd.DataFrame(predictions)


# ---------------- BACKTEST METRICS ----------------

def settle_market(row, market):
    if market == "over_2_5":
        return row["result_over25"]
    if market == "under_2_5":
        return row["result_under25"]
    raise ValueError(market)


def evaluate_market(predictions, market, min_score=65):
    pcol = f"{'o25' if market=='over_2_5' else 'u25'}_prob"
    ocol = f"{'o25' if market=='over_2_5' else 'u25'}_odds"
    scol = f"{'o25' if market=='over_2_5' else 'u25'}_score"

    cols = [pcol, ocol, scol]
    if not all(c in predictions.columns for c in cols):
        return pd.DataFrame()

    d = predictions.dropna(subset=cols).copy()
    d = d[d[scol] >= min_score].copy()

    if d.empty:
        return d

    d["win"] = d.apply(lambda r: settle_market(r, market), axis=1)
    d["profit_1u"] = np.where(
        d["win"] == 1,
        d[ocol] - 1,
        -1
    )
    d["roi"] = d["profit_1u"].mean()
    d["cum_profit"] = d["profit_1u"].cumsum()
    d["drawdown"] = d["cum_profit"] - d["cum_profit"].cummax()

    return d


def calibration_table(predictions, market, bins=5):
    pcol = f"{'o25' if market=='over_2_5' else 'u25'}_prob"
    rcol = "result_over25" if market == "over_2_5" else "result_under25"

    d = predictions.dropna(subset=[pcol, rcol]).copy()
    if d.empty:
        return d

    d["bin"] = pd.cut(
        d[pcol],
        bins=np.linspace(0.5, 1.0, bins + 1),
        include_lowest=True
    )

    return d.groupby("bin", observed=False).agg(
        n=(rcol, "size"),
        predicted=(pcol, "mean"),
        observed=(rcol, "mean")
    ).reset_index()


def league_market_report(predictions):
    rows = []

    for market in ["over_2_5", "under_2_5"]:
        d = evaluate_market(predictions, market, min_score=65)
        if d.empty:
            continue

        g = d.groupby("league").agg(
            bets=("profit_1u", "size"),
            wins=("win", "sum"),
            roi=("profit_1u", "mean"),
            avg_ev=(f"{'o25' if market=='over_2_5' else 'u25'}_ev", "mean"),
            avg_score=(f"{'o25' if market=='over_2_5' else 'u25'}_score", "mean")
        ).reset_index()

        g["market"] = market
        rows.append(g)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values(
        ["roi", "bets"], ascending=[False, False]
    )


# ---------------- SYSTEMS ----------------

def system_combinations(picks, k):
    return list(itertools.combinations(picks, k))


def evaluate_system(picks, k, stake_per_combo=1.0):
    """
    Picks: DataFrame con una fila por selección y columnas odds/win.
    Calcula un sistema k/n de forma exacta.
    """
    if len(picks) < k:
        return None

    total_investment = math.comb(len(picks), k) * stake_per_combo
    gross_return = 0.0

    for combo in itertools.combinations(range(len(picks)), k):
        odds = 1.0
        valid = True
        for i in combo:
            if not bool(picks.iloc[i]["win"]):
                valid = False
                break
            odds *= float(picks.iloc[i]["odds"])

        if valid:
            gross_return += stake_per_combo * odds

    net = gross_return - total_investment
    return {
        "n": len(picks),
        "k": k,
        "combinations": math.comb(len(picks), k),
        "investment": total_investment,
        "gross_return": gross_return,
        "net": net,
        "roi": net / total_investment if total_investment else np.nan
    }


# ---------------- MAIN ----------------

def main():
    print("V15 — VALUE HUNTER")
    print("=" * 60)
    print("Corte:", CUTOFF.date())
    print("Test ciego:", TEST_START.date(), "→", TEST_END.date())

    data = load_data()

    # IMPORTANTE: aunque el dataset actual ya contiene resultados posteriores,
    # el modelo de cada partido usa Date < partido. Por tanto no hay look-ahead.
    data = data.sort_values("Date").reset_index(drop=True)

    print(f"Partidos cargados: {len(data):,}")

    pred = walk_forward_test(data)
    pred.to_csv("v15_test_predictions.csv", index=False)

    print(f"Partidos de test: {len(pred):,}")
    print("\nMEJORES OPORTUNIDADES V15")
    print("-" * 60)

    score_cols = [c for c in ["o25_score", "u25_score"] if c in pred.columns]

    if score_cols:
        for c in score_cols:
            tmp = pred.dropna(subset=[c]).sort_values(c, ascending=False).head(15)
            print(f"\n{c.upper()}")
            print(tmp[[
                "date", "league", "home", "away",
                "lambda_home", "lambda_away", c
            ]].to_string(index=False))

    report = league_market_report(pred)
    report.to_csv("v15_league_market_report.csv", index=False)

    print("\n\nRANKING LIGA × MERCADO")
    print("-" * 60)
    if not report.empty:
        print(report.head(30).to_string(index=False))
    else:
        print("No hubo suficientes cuotas/selecciones para generar ranking.")

    for market in ["over_2_5", "under_2_5"]:
        d = evaluate_market(pred, market, min_score=65)
        if not d.empty:
            print(f"\n{market.upper()} — SCORE >= 65")
            print("Apuestas:", len(d))
            print("Acierto:", round(d["win"].mean() * 100, 2), "%")
            print("ROI:", round(d["profit_1u"].mean() * 100, 2), "%")
            print("Beneficio en 1u:", round(d["profit_1u"].sum(), 3), "u")
            print("Máx. drawdown:", round(d["drawdown"].min(), 3), "u")

            cal = calibration_table(pred, market)
            cal.to_csv(f"v15_calibration_{market}.csv", index=False)

    print("\nArchivos generados:")
    print(" - v15_test_predictions.csv")
    print(" - v15_league_market_report.csv")
    print(" - v15_calibration_over_2_5.csv")
    print(" - v15_calibration_under_2_5.csv")
    print("\nIMPORTANTE: esto es investigación estadística, no garantía de rentabilidad.")


if __name__ == "__main__":
    main()
