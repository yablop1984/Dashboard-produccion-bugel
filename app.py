import streamlit as st
import pandas as pd

# --- Configuración página ---
st.set_page_config(page_title="Dashboard Producción", layout="wide")
st.title("📊 Dashboard Producción Bugel")

# --- Carga y preprocesamiento ---
url = (
    "https://docs.google.com/spreadsheets/"
    "d/1YtUaTmcVQR7N9FvXb5mYmvrUKkOnufwQV9HoF9Gf75I/"
    "export?format=csv&gid=0"
)
df = pd.read_csv(url)
df['fecha_inicio_dt'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
df['fecha_fin_dt']    = pd.to_datetime(df['fecha_fin'],    errors='coerce')
df['tiempo_minutos']  = (df['fecha_fin_dt'] - df['fecha_inicio_dt']).dt.total_seconds() / 60

# --- Define las pestañas ---
tab1, tab2 = st.tabs(["📈 Vista General", "🔍 Análisis por Empleado"])

# --- PESTAÑA 1: Vista General ---
with tab1:
    st.subheader("🔢 Indicadores Globales")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧩 Total piezas", int(df['piezas'].sum()))
    col2.metric("📋 Total registros", len(df))
    prom_global = df['piezas'].mean() if not df.empty else 0
    col3.metric("📊 Promedio pzas/registro", round(prom_global, 2))

    st.subheader("🏗️ Piezas por Proyecto")
    st.bar_chart(df.groupby('proyecto')['piezas'].sum())

    st.subheader("📅 Producción por Día")
    df_dia = df.copy()
    df_dia['fecha'] = df_dia['fecha_inicio_dt'].dt.date
    st.line_chart(df_dia.groupby('fecha')['piezas'].sum())

# --- PESTAÑA 2: Análisis por Empleado ---
with tab2:
    st.sidebar.header("🔍 Filtros Detallados")
    sel_empleados = st.sidebar.multiselect(
        "Empleado", options=df['nombre'].unique(), default=df['nombre'].unique()
    )
    df_emp = df[df['nombre'].isin(sel_empleados)]

    sel_proyectos = st.sidebar.multiselect(
        "Proyecto", options=df_emp['proyecto'].unique(), default=df_emp['proyecto'].unique()
    )
    df_proj = df_emp[df_emp['proyecto'].isin(sel_proyectos)]

    sel_maquinas = st.sidebar.multiselect(
        "Máquina", options=df_proj['maquina'].dropna().unique(), default=df_proj['maquina'].dropna().unique()
    )
    
    sel_procesos = st.sidebar.multiselect(
        "Proceso", options=df_proj['proceso'].dropna().unique(), default=df_proj['proceso'].dropna().unique()
    )

    fecha_min = df['fecha_inicio_dt'].min().date()
    fecha_max = df['fecha_inicio_dt'].max().date()
    rango_fechas = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])

    df_filtrado = df_proj[
        (df_proj['maquina'].isin(sel_maquinas)) &
        (df_proj['proceso'].isin(sel_procesos)) &
        (df_proj['fecha_inicio_dt'].dt.date >= rango_fechas[0]) &
        (df_proj['fecha_inicio_dt'].dt.date <= rango_fechas[1])
    ]

    # Métricas por empleado
    st.subheader("📊 Métricas por Empleado")
    count_registros = len(df_filtrado)
    last_date = df_filtrado['fecha_inicio_dt'].max().date() if count_registros > 0 else None
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("📋 Número de registros", count_registros)
    mcol2.metric("🗓️ Última fecha registrada", last_date)

    # Gráficos
    st.subheader("👷‍♂️ Piezas por Empleado")
    if not df_filtrado.empty:
        st.bar_chart(df_filtrado.groupby('nombre')['piezas'].sum())

    st.subheader("🗺️ Dispersión Tiempo vs Piezas")
    if not df_filtrado.empty:
        st.vega_lite_chart(
            df_filtrado,
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "tiempo_minutos", "type": "quantitative", "title": "Tiempo (min)"},
                    "y": {"field": "piezas",          "type": "quantitative", "title": "Piezas"},
                    "color": {"field": "proyecto",   "type": "nominal"}
                }
            },
            use_container_width=True
        )

    st.subheader("📄 Detalle de registros")
    st.dataframe(df_filtrado)

    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df_filtrado.to_csv(index=False).encode('utf-8'),
        file_name='registros_filtrados.csv',
        mime='text/csv'
    )


