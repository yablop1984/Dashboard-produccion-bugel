import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Dashboard Producción Bugel", layout="wide")

st.title("📊 Dashboard Producción Bugel")

# Leer datos desde Google Sheets
url = "https://docs.google.com/spreadsheets/d/1YtUaTmcVQR7N9FvXb5mYmvrUKkOnufwQV9HoF9Gf75I/export?format=csv&gid=0"
df = pd.read_csv(url)

# Conversión de fechas
df['fecha_inicio_dt'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
df['fecha_fin_dt'] = pd.to_datetime(df['fecha_fin'], errors='coerce')

# Cálculo de duración en minutos
df['duracion_min'] = (df['fecha_fin_dt'] - df['fecha_inicio_dt']).dt.total_seconds() / 60

# Sidebar - Filtros
st.sidebar.header("Filtros")
empleados = st.sidebar.multiselect("Empleado", options=df['nombre'].unique(), default=df['nombre'].unique())
proyectos = st.sidebar.multiselect("Proyecto", options=df['proyecto'].unique(), default=df['proyecto'].unique())
maquinas = st.sidebar.multiselect("Máquina", options=df['maquina'].dropna().unique(), default=df['maquina'].dropna().unique())
procesos = st.sidebar.multiselect("Proceso", options=df['proceso'].dropna().unique(), default=df['proceso'].dropna().unique())

# Rango de fechas
fecha_min = df['fecha_inicio_dt'].min()
fecha_max = df['fecha_inicio_dt'].max()
rango_fechas = st.sidebar.date_input("Rango de Fechas", [fecha_min, fecha_max])

# Filtrado de datos
df_filtrado = df[
    (df['nombre'].isin(empleados)) &
    (df['proyecto'].isin(proyectos)) &
    (df['maquina'].isin(maquinas)) &
    (df['proceso'].isin(procesos)) &
    (df['fecha_inicio_dt'].dt.date >= rango_fechas[0]) &
    (df['fecha_inicio_dt'].dt.date <= rango_fechas[1])
]

# KPIs principales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Piezas", int(df_filtrado['piezas'].sum()))
with col2:
    st.metric("Total Registros", len(df_filtrado))
with col3:
    promedio_piezas = df_filtrado['piezas'].mean() if len(df_filtrado) > 0 else 0
    st.metric("Promedio Piezas", round(promedio_piezas, 2))
with col4:
    promedio_tiempo = df_filtrado['duracion_min'].mean() if len(df_filtrado) > 0 else 0
    st.metric("Duración Promedio (min)", round(promedio_tiempo, 2))

# Gráfico: Piezas por Empleado
st.subheader("📊 Piezas por Empleado")
if not df_filtrado.empty:
    st.bar_chart(df_filtrado.groupby('nombre')['piezas'].sum())

# Gráfico: Piezas por Proyecto
st.subheader("📊 Piezas por Proyecto")
if not df_filtrado.empty:
    st.bar_chart(df_filtrado.groupby('proyecto')['piezas'].sum())

# Gráfico: Producción por Día
st.subheader("📈 Producción por Día")
if not df_filtrado.empty:
    df_filtrado['fecha'] = df_filtrado['fecha_inicio_dt'].dt.date
    st.line_chart(df_filtrado.groupby('fecha')['piezas'].sum())

# Tabla completa
st.subheader("📋 Detalle de Registros")
st.dataframe(df_filtrado)

# Botón para descargar datos filtrados
st.download_button(
    label="📥 Descargar datos filtrados (CSV)",
    data=df_filtrado.to_csv(index=False).encode('utf-8'),
    file_name="produccion_filtrada.csv",
    mime="text/csv"
)

