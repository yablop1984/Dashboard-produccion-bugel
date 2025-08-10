import streamlit as st
import pandas as pd

# --- Configuración página ---
st.set_page_config(page_title="Dashboard Producción", layout="wide")
st.title("📊 Dashboard Producción Bugel")

# --- Carga de datos ---
url = (
    "https://docs.google.com/spreadsheets/"
    "d/1YtUaTmcVQR7N9FvXb5mYmvrUKkOnufwQV9HoF9Gf75I/"
    "export?format=csv&gid=0"
)
df = pd.read_csv(url)

# --- Función robusta para limpiar y convertir fechas ---
def limpiar_y_convertir_fecha(serie):
    # Convertir todo a texto y limpiar
    serie = serie.astype(str).str.strip()
    
    # Reemplazar coma por espacio
    serie = serie.str.replace(',', ' ', regex=False)
    
    # Eliminar espacios extra
    serie = serie.str.replace(r'\s+', ' ', regex=True)
    
    # Primer intento: formato latino
    fechas = pd.to_datetime(serie, errors='coerce', dayfirst=True, infer_datetime_format=True)
    
    # Segundo intento: por si algunas están en otro orden
    mask_nat = fechas.isna()
    if mask_nat.any():
        fechas[mask_nat] = pd.to_datetime(serie[mask_nat], errors='coerce', dayfirst=False, infer_datetime_format=True)
    
    return fechas

# --- Aplicar normalización a columnas ---
df['fecha_inicio_dt'] = limpiar_y_convertir_fecha(df['fecha_inicio'])
df['fecha_fin_dt'] = limpiar_y_convertir_fecha(df['fecha_fin'])

# --- Diagnóstico de fechas ---
valid_inicio = df['fecha_inicio_dt'].notna().sum()
valid_fin = df['fecha_fin_dt'].notna().sum()
st.caption(f"✔ Fechas inicio válidas: {valid_inicio}/{len(df)}, Fechas fin válidas: {valid_fin}/{len(df)}")

# --- Filtrar registros sin fecha válida ---
df = df.dropna(subset=['fecha_inicio_dt'])

# --- Calcular duración ---
df['tiempo_minutos'] = (df['fecha_fin_dt'] - df['fecha_inicio_dt']).dt.total_seconds() / 60

# --- Pestañas ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vista General", "🔍 Análisis por Empleado", "📤 Exportar Datos", "🕒 Último día"])

# ------------------------------
# PESTAÑA 1: Vista General
# ------------------------------
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

# ------------------------------
# PESTAÑA 2: Análisis por Empleado
# ------------------------------
with tab2:
    st.sidebar.header("🔍 Filtros Detallados")

    # Filtros dinámicos
    sel_empleados = st.sidebar.multiselect(
        "Empleado", options=sorted(df['nombre'].dropna().unique()), default=list(df['nombre'].dropna().unique())
    )
    df_emp = df[df['nombre'].isin(sel_empleados)]

    sel_proyectos = st.sidebar.multiselect(
        "Proyecto", options=sorted(df_emp['proyecto'].dropna().unique()), default=list(df_emp['proyecto'].dropna().unique())
    )
    df_proj = df_emp[df_emp['proyecto'].isin(sel_proyectos)]

    sel_maquinas = st.sidebar.multiselect(
        "Máquina", options=sorted(df_proj['maquina'].dropna().unique()), default=list(df_proj['maquina'].dropna().unique())
    )

    sel_procesos = st.sidebar.multiselect(
        "Proceso", options=sorted(df_proj['proceso'].dropna().unique()), default=list(df_proj['proceso'].dropna().unique())
    )

    # Rango de fechas
    fecha_min = df['fecha_inicio_dt'].min().date()
    fecha_max = df['fecha_inicio_dt'].max().date()
    rango_fechas = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])

    if len(rango_fechas) == 2:
        start_date, end_date = rango_fechas
    else:
        start_date, end_date = fecha_min, fecha_max

    # Filtrado final
    df_filtrado = df_proj[
        (df_proj['maquina'].isin(sel_maquinas)) &
        (df_proj['proceso'].isin(sel_procesos)) &
        (df_proj['fecha_inicio_dt'].dt.date >= start_date) &
        (df_proj['fecha_inicio_dt'].dt.date <= end_date)
    ]

    st.write(f"✅ Total registros filtrados: **{len(df_filtrado)}**")

    # Gráfico: Piezas por Empleado
    st.subheader("👷‍♂️ Piezas por Empleado")
    if not df_filtrado.empty:
        st.bar_chart(df_filtrado.groupby('nombre')['piezas'].sum())

    # Gráfico: Piezas por Proyecto filtrado
    st.subheader("📊 Piezas por Proyecto (Filtrado)")
    if not df_filtrado.empty:
        st.bar_chart(df_filtrado.groupby('proyecto')['piezas'].sum())

    # Gráfico adicional: Evolución por Empleado
    st.subheader("📈 Evolución por Empleado (Filtrado)")
    if not df_filtrado.empty:
        df_line = df_filtrado.copy()
        df_line['fecha'] = df_line['fecha_inicio_dt'].dt.date
        st.line_chart(df_line.groupby(['fecha', 'nombre'])['piezas'].sum().unstack().fillna(0))

    # Dispersión Tiempo vs Piezas
    st.subheader("🗺️ Dispersión Tiempo vs Piezas")
    if not df_filtrado.empty:
        st.vega_lite_chart(
            df_filtrado,
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "tiempo_minutos", "type": "quantitative", "title": "Tiempo (min)"},
                    "y": {"field": "piezas", "type": "quantitative", "title": "Piezas"},
                    "color": {"field": "proyecto", "type": "nominal"}
                }
            },
            use_container_width=True
        )

    # Tabla detalle
    st.subheader("📄 Detalle de registros")
    st.dataframe(df_filtrado)

    # Botón para descargar CSV filtrado
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df_filtrado.to_csv(index=False).encode('utf-8'),
        file_name='registros_filtrados.csv',
        mime='text/csv'
    )

# ------------------------------
# PESTAÑA 3: Exportar Datos
# ------------------------------
with tab3:
    st.subheader("📤 Descargar Dataset Completo")
    st.write(f"📦 Total registros en el dataset: **{len(df)}**")
    st.download_button(
        "⬇️ Descargar dataset completo (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='produccion_completo.csv',
        mime='text/csv'
    )
# ------------------------------
# PESTAÑA 4: Trabajo del último día ingresado
# ------------------------------
with tab4:
    st.subheader("🕒 Trabajo del último día ingresado")

    # Fecha más reciente (por fecha_inicio_dt)
    ultima_fecha = df['fecha_inicio_dt'].dt.date.max()
    if pd.isna(ultima_fecha):
        st.info("No hay fechas válidas en el dataset.")
    else:
        st.caption(f"Última fecha detectada en los datos: **{ultima_fecha}**")

        # Filtrar registros del último día
        df_last = df[df['fecha_inicio_dt'].dt.date.eq(ultima_fecha)].copy()

        # Asegurar columna de tiempo en minutos
        if 'minutos_ajustados' in df_last.columns:
            df_last['min_trabajo'] = df_last['minutos_ajustados'].fillna(0)
        else:
            df_last['min_trabajo'] = df_last['tiempo_minutos'].fillna(0)

        # --- KPIs ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📋 Registros", int(len(df_last)))
        col2.metric("👷 Empleados", int(df_last['nombre'].nunique()))
        col3.metric("🏗️ Proyectos", int(df_last['proyecto'].nunique()))
        col4.metric("🧩 Piezas", int(df_last['piezas'].fillna(0).sum()))
        col5.metric("⏱️ Tiempo (horas)", round(df_last['min_trabajo'].sum()/60, 2))

        # --- Tabla por empleado (piezas y minutos)
        st.subheader("👷 Trabajo por empleado (último día)")
        tabla_emp = (
            df_last.groupby('nombre', as_index=False)
                   .agg(piezas=('piezas','sum'),
                        minutos=('min_trabajo','sum'),
                        proyectos_distintos=('proyecto','nunique'))
                   .sort_values(['piezas','minutos'], ascending=False)
        )
        st.dataframe(tabla_emp, use_container_width=True)

        # --- Pareto (piezas por empleado del último día)
        st.subheader("📈 Pareto de piezas por empleado (último día)")
        if not tabla_emp.empty:
            pareto = tabla_emp[['nombre','piezas']].sort_values('piezas', ascending=False).reset_index(drop=True)
            pareto['acum'] = pareto['piezas'].cumsum()
            total = pareto['piezas'].sum()
            pareto['acum_pct'] = (pareto['acum'] / total * 100).round(2) if total > 0 else 0

            st.vega_lite_chart(
                {
                    "data": {"values": pareto.to_dict(orient="records")},
                    "layer": [
                        {   # Barras (piezas)
                            "mark": {"type": "bar"},
                            "encoding": {
                                "x": {"field": "nombre", "type": "nominal", "sort": None, "title": "Empleado"},
                                "y": {"field": "piezas", "type": "quantitative", "title": "Piezas"}
                            }
                        },
                        {   # Línea acumulada (%)
                            "mark": {"type": "line", "point": True},
                            "encoding": {
                                "x": {"field": "nombre", "type": "nominal", "sort": None},
                                "y": {"field": "acum_pct", "type": "quantitative", "title": "Acumulado %", "axis": {"grid": False}},
                                "color": {"value": "black"}
                            }
                        }
                    ],
                    "resolve": {"scale": {"y": "independent"}},
                },
                use_container_width=True
            )
            st.caption("Consejo: si prefieres Pareto por proyecto, cambia el agrupamiento a 'proyecto' en lugar de 'nombre'.")

        # --- Gráfico estadístico: Boxplot de tiempos por proceso
        st.subheader("📦 Distribución de tiempos por proceso (boxplot)")
        if not df_last.empty:
            st.vega_lite_chart(
                {
                    "data": {"values": df_last[['proceso','min_trabajo']].dropna().to_dict(orient="records")},
                    "mark": {"type": "boxplot"},
                    "encoding": {
                        "x": {"field": "proceso", "type": "nominal", "title": "Proceso"},
                        "y": {"field": "min_trabajo", "type": "quantitative", "title": "Minutos de trabajo"}
                    }
                },
                use_container_width=True
            )

        # --- Tabla detalle del día y descarga
        st.subheader("📄 Detalle del último día")
        st.dataframe(df_last, use_container_width=True)
        st.download_button(
            "⬇️ Descargar registros del último día (CSV)",
            data=df_last.to_csv(index=False).encode('utf-8'),
            file_name=f"registros_{ultima_fecha}.csv",
            mime="text/csv"
        )