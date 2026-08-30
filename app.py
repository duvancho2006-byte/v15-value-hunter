import streamlit as st
import pandas as pd

st.set_page_config(page_title='V15 — Value Hunter', page_icon='⚽', layout='wide')

st.title('⚽ V15 — VALUE HUNTER')
st.caption('Motor experimental de análisis estadístico y búsqueda de valor en fútbol')

try:
    from v15_value_hunter import load_data, walk_forward_test
except Exception as e:
    st.error(f'No se pudo cargar el motor V15: {e}')
    st.stop()

st.sidebar.header('⚙️ V15')
page = st.sidebar.radio('Sección', ['🏠 Resumen', '💰 Value Scanner', '🧪 Test 15–28 agosto', '📚 Datos'])

with st.spinner('Cargando datos históricos de Football-Data...'):
    data = load_data()

if data.empty:
    st.error('No se pudieron cargar datos históricos.')
    st.stop()

if page == '🏠 Resumen':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Partidos cargados', f'{len(data):,}')
    c2.metric('Ligas configuradas', '15')
    c3.metric('Temporadas', '7')
    c4.metric('Corte', '14/08/2026')

    st.subheader('🔒 Diseño del experimento')
    st.write('El motor calcula cada predicción usando únicamente información anterior al partido. El período 15–28 de agosto de 2026 queda reservado como prueba fuera de muestra.')
    st.info('V15 es una herramienta experimental. Un Value Gap positivo no garantiza una apuesta ganadora ni demuestra por sí solo una ventaja sostenible.')

elif page == '📚 Datos':
    st.subheader('📚 Datos históricos')
    st.dataframe(data[['Date','country','division','HomeTeam','AwayTeam']].tail(100), use_container_width=True, hide_index=True)

else:
    with st.spinner('Calculando el test fuera de muestra...'):
        pred = walk_forward_test(data)

    if pred.empty:
        st.warning('No se encontraron partidos en la ventana de prueba.')
        st.stop()

    if page == '🧪 Test 15–28 agosto':
        st.subheader('🧪 Test fuera de muestra: 15–28 agosto 2026')
        st.metric('Partidos evaluados', len(pred))
        st.dataframe(pred, use_container_width=True, hide_index=True)

    else:
        st.subheader('💰 Value Scanner')
        score_cols = [c for c in ['o25_score','u25_score'] if c in pred.columns]
        if not score_cols:
            st.warning('Todavía no hay cuotas O/U 2.5 disponibles para el escáner.')
            st.stop()

        min_score = st.slider('Value Score mínimo', 50, 100, 65)
        tables = []
        for c in score_cols:
            d = pred[pred[c].fillna(-1) >= min_score].copy()
            if not d.empty:
                d['Mercado'] = 'Over 2.5' if c == 'o25_score' else 'Under 2.5'
                d['Score'] = d[c]
                tables.append(d)
        if not tables:
            st.info('No aparecen oportunidades con este filtro.')
        else:
            result = pd.concat(tables, ignore_index=True)
            cols = ['date','league','home','away','Mercado','Score','lambda_home','lambda_away']
            st.dataframe(result[cols].sort_values('Score', ascending=False), use_container_width=True, hide_index=True)
