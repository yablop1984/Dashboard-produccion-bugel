import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Dashboard Producción", layout="wide")
st.title("📊 Dashboard Producción Bugel")

# Cargar datos desde Google Sheets
url = "https://docs.google.com/spreadsheets/d/1YtUaTmcVQR7N9FvXb5mYmvrUKkOnufwQV9HoF9Gf75I/export?format=csv&gid=0"
df = pd.read_csv(url)

# Procesamiento de fechas
df['fecha_inicio_dt'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
df['fecha_fin_dt'] = pd.to_datetime(df['fecha_fin'], errors='coerce')

# Filtros en barra lateral
st.sidebar.header("🔍 Filtros")

empleados = st.sidebar.multiselect("Empleado", options=df['nombre'].unique(), default=df['nombre'].unique())
proyectos = st.sidebar.multiselect("Proyecto", options=df['proyecto'].unique(), default=df['proyecto'].unique())
maquinas = st.sidebar.multiselect("Máquina", options=df['maquina'].dropna().unique(), default=df['maquina'].dropna().unique())
procesos = st.sidebar.multiselect("Proceso", options=df['proceso'].dropna().unique(), default=df['proceso'].dropna().unique())

fecha_min = df['fecha_inicio_dt'].min()
fecha_max = df['fecha_inicio_dt'].max()
rango_fechas = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])

# Aplicar filtros
df_filtrado = df[
    (df['nombre'].isin(empleados)) &
    (df['proyecto'].isin(proyectos)) &
    (df['maquina'].isin(maquinas)) &
    (df['proceso'].isin(procesos)) &
    (df['fecha_inicio_dt'].dt.date >= rango_fechas[0]) &
    (df['fecha_inicio_dt'].dt.date <= rango_fechas[1])
]

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧩 Total piezas", int(df_filtrado['piezas'].sum()))
with col2:
    st.metric("📋 Total registros", len(df_filtrado))
with col3:
    promedio = df_filtrado['piezas'].mean() if not df_filtrado.empty else 0
    st.metric("📊 Promedio piezas/registro", round(promedio, 2))

# Gráfico: Piezas por empleado
st.subheader("👷‍♂️ Piezas por empleado")
if not df_filtrado.empty:
    st.bar_chart(df_filtrado.groupby('nombre')['piezas'].sum())

# Gráfico: Piezas por proyecto
st.subheader("🏗️ Piezas por proyecto")
if not df_filtrado.empty:
    st.bar_chart(df_filtrado.groupby('proyecto')['piezas'].sum())

# Gráfico: Evolución por día
st.subheader("📅 Producción por día")
if not df_filtrado.empty:
    df_filtrado['fecha'] = df_filtrado['fecha_inicio_dt'].dt.date
    st.line_chart(df_filtrado.groupby('fecha')['piezas'].sum())

# Tabla de datos
st.subheader("📄 Detalle de registros")
st.dataframe(df_filtrado)

# Botón de descarga
st.download_button(
    label="⬇️ Descargar datos filtrados (CSV)",
    data=df_filtrado.to_csv(index=False).encode('utf-8'),
    file_name='registros_filtrados.csv',
    mime='text/csv'
)

