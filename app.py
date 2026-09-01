import streamlit as st
import pandas as pd

st.set_page_config(
    page_title='V15 — Value Hunter',
    page_icon='⚽',
    layout='wide'
)

st.title('⚽ V15 — VALUE HUNTER')
st.caption(
    'Motor experimental de análisis estadístico y búsqueda de valor en fútbol'
)

try:
    from v15_value_hunter import (
        load_data,
        walk_forward_test,
        MIN_MODEL_PROB,
        MIN_VALUE_GAP,
        MIN_ODDS,
        MAX_ODDS,
    )
except Exception as e:
    st.error(f'No se pudo cargar el motor V15: {e}')
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header('⚙️ V15')

page = st.sidebar.radio(
    'Sección',
    [
        '🏠 Resumen',
        '💰 Value Scanner',
        '🧪 Test 15–28 agosto',
        '📊 V15 Auditor',
        '📚 Datos'
    ]
)


# ============================================================
# CARGA DE DATOS
# ============================================================

with st.spinner('Cargando datos históricos de Football-Data...'):

    data = load_data(include_test=True)
    training_data = load_data(include_test=False)


if data.empty:

    st.error('No se pudieron cargar datos históricos.')

    st.stop()


if training_data.empty:

    st.error('No se pudo construir la base de entrenamiento.')

    st.stop()


# ============================================================
# RESUMEN
# ============================================================

if page == '🏠 Resumen':

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        'Partidos cargados',
        f'{len(data):,}'
    )

    c2.metric(
        'Ligas configuradas',
        '15'
    )

    c3.metric(
        'Temporadas',
        '7'
    )

    c4.metric(
        'Corte',
        '14/08/2026'
    )

    st.subheader('🔒 Diseño del experimento')

    st.write(
        'El motor calcula cada predicción usando únicamente '
        'información anterior al partido. El período 15–28 de '
        'agosto de 2026 queda reservado como prueba fuera de muestra.'
    )

    st.info(
        'V15 es una herramienta experimental. Un Value Gap positivo '
        'no garantiza una apuesta ganadora ni demuestra por sí solo '
        'una ventaja sostenible.'
    )


# ============================================================
# DATOS
# ============================================================

elif page == '📚 Datos':

    st.subheader('📚 Datos históricos')

    st.dataframe(
        training_data[
            [
                'Date',
                'country',
                'division',
                'HomeTeam',
                'AwayTeam'
            ]
        ].tail(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TEST Y VALUE SCANNER
# ============================================================

else:

    with st.spinner(
        'Calculando el test fuera de muestra...'
    ):

        pred = walk_forward_test(data)


    if pred.empty:

        st.warning(
            'No se encontraron partidos en la ventana de prueba.'
        )

        st.stop()


    # ========================================================
    # TEST 15–28 AGOSTO
    # ========================================================

    if page == '🧪 Test 15–28 agosto':

        st.subheader(
            '🧪 Test fuera de muestra: 15–28 agosto 2026'
        )

        st.metric(
            'Partidos evaluados',
            len(pred)
        )

        st.dataframe(
            pred,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # VALUE SCANNER
    # ========================================================

    elif page == '📊 V15 Auditor':
    st.subheader('📊 V15 AUDITOR')
    st.caption(
        'Evaluación de las señales generadas por V15 '
        'en el período fuera de muestra.'
    )

        # ----------------------------------------------------
        # Preparar resultados
        # ----------------------------------------------------

        auditor_tables = []

        market_config = {
            'Over 2.5': {
                'score': 'o25_score',
                'prob': 'o25_prob',
                'odds': 'o25_odds',
                'gap': 'o25_gap',
                'result': 'result_over25'
            },
            'Under 2.5': {
                'score': 'u25_score',
                'prob': 'u25_prob',
                'odds': 'u25_odds',
                'gap': 'u25_gap',
                'result': 'result_under25'
            }
        }

        # ----------------------------------------------------
        # Evaluar cada mercado
        # ----------------------------------------------------

        for market, cfg in market_config.items():

            required = [
                cfg['score'],
                cfg['prob'],
                cfg['odds'],
                cfg['gap'],
                cfg['result']
            ]

            if not all(
                col in pred.columns
                for col in required
            ):
                continue

            d = pred[
                pred[cfg['score']].notna() &
                pred[cfg['prob']].notna() &
                pred[cfg['odds']].notna() &
                pred[cfg['gap']].notna() &
                pred[cfg['result']].notna()
            ].copy()

            if d.empty:
                continue

            d['Mercado'] = market
            d['Score'] = d[cfg['score']]
            d['Probabilidad'] = d[cfg['prob']]
            d['Cuota'] = d[cfg['odds']]
            d['Gap'] = d[cfg['gap']]
            d['Resultado'] = d[cfg['result']]

            d['Win'] = d['Resultado'].astype(int)

            d['Profit_1U'] = (
                d['Cuota'] - 1
            ).where(
                d['Win'] == 1,
                -1
            )

            auditor_tables.append(d)


        # ----------------------------------------------------
        # Mostrar auditoría
        # ----------------------------------------------------

        if not auditor_tables:

            st.warning(
                'No hay suficientes datos de resultados '
                'para realizar la auditoría.'
            )

        else:

            audit = pd.concat(
                auditor_tables,
                ignore_index=True
            )

            st.metric(
                'Señales evaluables',
                len(audit)
            )

            # ------------------------------------------------
            # Resumen general
            # ------------------------------------------------

            total_bets = len(audit)
            wins = int(audit['Win'].sum())
            losses = total_bets - wins

            hit_rate = (
                wins / total_bets
                if total_bets > 0
                else 0
            )

            total_profit = audit['Profit_1U'].sum()

            roi = (
                total_profit / total_bets
                if total_bets > 0
                else 0
            )

            avg_ev = (
                audit['Probabilidad'] * audit['Cuota'] - 1
            ).mean()

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                'Señales',
                total_bets
            )

            c2.metric(
                'Aciertos',
                wins
            )

            c3.metric(
                'Fallos',
                losses
            )

            c4.metric(
                'Hit Rate',
                f'{hit_rate:.2%}'
            )

            c5.metric(
                'ROI',
                f'{roi:.2%}'
            )

            st.metric(
                'EV medio del modelo',
                f'{avg_ev:.2%}'
            )


            # ------------------------------------------------
            # Over vs Under
            # ------------------------------------------------

            st.subheader('⚽ Rendimiento por mercado')

            market_summary = (
                audit
                .groupby('Mercado')
                .agg(
                    Señales=('Win', 'size'),
                    Aciertos=('Win', 'sum'),
                    Beneficio=('Profit_1U', 'sum'),
                    Probabilidad_media=('Probabilidad', 'mean'),
                    EV_medio=('Profit_1U', 'mean')
                )
                .reset_index()
            )

            market_summary['Hit Rate'] = (
                market_summary['Aciertos']
                / market_summary['Señales']
            )

            market_summary['ROI'] = (
                market_summary['Beneficio']
                / market_summary['Señales']
            )

            st.dataframe(
                market_summary,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            st.subheader('🎯 Rendimiento según Value Score')

            bins = [
                65,
                70,
                75,
                80,
                85,
                90,
                95,
                101
            ]

            labels = [
                '65–69',
                '70–74',
                '75–79',
                '80–84',
                '85–89',
                '90–94',
                '95–100'
            ]

            audit['Score_Rango'] = pd.cut(
                audit['Score'],
                bins=bins,
                labels=labels,
                right=False
            )

            score_summary = (
                audit
                .groupby(
                    'Score_Rango',
                    observed=False
                )
                .agg(
                    Señales=('Win', 'size'),
                    Aciertos=('Win', 'sum'),
                    Beneficio=('Profit_1U', 'sum')
                )
                .reset_index()
            )

            score_summary['Hit Rate'] = (
                score_summary['Aciertos']
                / score_summary['Señales']
            )

            score_summary['ROI'] = (
                score_summary['Beneficio']
                / score_summary['Señales']
            )

            st.dataframe(
                score_summary,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Tabla de señales auditadas
            # ------------------------------------------------

            st.subheader('🔎 Señales auditadas')

            audit_cols = [
                'date',
                'league',
                'home',
                'away',
                'Mercado',
                'Score',
                'Probabilidad',
                'Cuota',
                'Gap',
                'Profit_1U'
            ]

            audit_view = audit[
                [
                    col for col in audit_cols
                    if col in audit.columns
                ]
            ].sort_values(
                'Score',
                ascending=False
            )

            st.dataframe(
                audit_view,
                use_container_width=True,
                hide_index=True
            )

        score_cols = [
            c
            for c in [
                'o25_score',
                'u25_score'
            ]
            if c in pred.columns
        ]


        if not score_cols:

            st.warning(
                'Todavía no hay cuotas O/U 2.5 disponibles '
                'para el escáner.'
            )

            st.stop()


        min_score = st.slider(
            'Value Score mínimo',
            50,
            100,
            65
        )


        tables = []


        # ====================================================
        # BUSCAR OPORTUNIDADES
        # ====================================================

        for c in score_cols:

            prefix = (
                "o25"
                if c == "o25_score"
                else "u25"
            )

            pcol = f"{prefix}_prob"
            ocol = f"{prefix}_odds"
            gcol = f"{prefix}_gap"


            required = [
                c,
                pcol,
                ocol,
                gcol
            ]


            if not all(
                x in pred.columns
                for x in required
            ):

                continue


            d = pred[
                (pred[c].fillna(-1) >= min_score) &
                (pred[pcol].fillna(-1) >= MIN_MODEL_PROB) &
                (pred[gcol].fillna(-1) >= MIN_VALUE_GAP) &
                (pred[ocol].fillna(-1) >= MIN_ODDS) &
                (pred[ocol].fillna(-1) <= MAX_ODDS)
            ].copy()


            if not d.empty:

                d["Mercado"] = (
                    "Over 2.5"
                    if c == "o25_score"
                    else "Under 2.5"
                )

                d["Score"] = d[c]

                d["Probabilidad"] = d[pcol]

                d["Cuota"] = d[ocol]

                d["Gap"] = d[gcol]

                d["EV"] = (
                    d["Probabilidad"]
                    * d["Cuota"]
                    - 1
                )

                tables.append(d)


        # ====================================================
        # MOSTRAR RESULTADOS
        # ====================================================

        if not tables:

            st.info(
                "No aparecen oportunidades con este filtro."
            )


        else:

            result = pd.concat(
                tables,
                ignore_index=True
            )


            cols = [
                "date",
                "league",
                "home",
                "away",
                "Mercado",
                "Score",
                "Probabilidad",
                "Cuota",
                "Gap",
                "EV",
                "lambda_home",
                "lambda_away"
            ]


            st.dataframe(
                result[
                    cols
                ].sort_values(
                    "Score",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )
