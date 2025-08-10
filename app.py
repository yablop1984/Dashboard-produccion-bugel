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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Vista General", 
    "🔍 Análisis por Empleado", 
    "📤 Exportar Datos", 
    "🕒 Último día",
    "🤖 ML / Proyecciones"
])
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
            st.caption("Consejo Ronald: si prefiere Pareto por proyecto, cambie el agrupamiento a 'proyecto' en lugar de 'nombre'.")

       
        # --- Tabla detalle del día y descarga
        st.subheader("📄 Detalle del último día")
        st.dataframe(df_last, use_container_width=True)
        st.download_button(
            "⬇️ Descargar registros del último día (CSV)",
            data=df_last.to_csv(index=False).encode('utf-8'),
            file_name=f"registros_{ultima_fecha}.csv",
            mime="text/csv"
        )

# ------------------------------
# PESTAÑA 5: ML / Proyecciones
# ------------------------------
with tab5:
    st.subheader("🤖 Análisis y Proyección mensual por empleado")

    # Columna de minutos a usar
    min_col = 'minutos_ajustados' if 'minutos_ajustados' in df.columns else 'tiempo_minutos'
    if min_col not in df.columns:
        st.warning("No encuentro columnas de minutos ('minutos_ajustados' o 'tiempo_minutos').")
    else:
        # Mes de referencia = mes de la última fecha disponible
        fecha_max = df['fecha_inicio_dt'].max()
        if pd.isna(fecha_max):
            st.info("No hay fechas válidas en el dataset.")
        else:
            mes_ref = fecha_max.to_period("M")
            df_mes = df[df['fecha_inicio_dt'].dt.to_period("M") == mes_ref].copy()
            df_mes['fecha'] = df_mes['fecha_inicio_dt'].dt.date

            # Parámetro: días laborados objetivo (26 por defecto)
            dias_obj = st.number_input("Días laborados objetivo para la proyección", min_value=1, max_value=31, value=26)

            # Agregación por empleado (mes en curso)
            agg = (
                df_mes.groupby('nombre', as_index=False)
                      .agg(
                          piezas_actual=('piezas','sum'),
                          minutos_actual=(min_col,'sum'),
                          dias_trabajados=('fecha','nunique'),
                          proyectos=('proyecto','nunique')
                      )
            )

            # Cálculos derivados
            agg['min_dia_prom'] = agg.apply(
                lambda r: r['minutos_actual'] / r['dias_trabajados'] if r['dias_trabajados'] > 0 else 0, axis=1
            )
            agg['piezas_por_min'] = agg.apply(
                lambda r: (r['piezas_actual'] / r['minutos_actual']) if r['minutos_actual'] > 0 else 0, axis=1
            )
            agg['dias_restantes'] = (dias_obj - agg['dias_trabajados']).clip(lower=0)
            agg['min_futuros'] = agg['min_dia_prom'] * agg['dias_restantes']

            # Proyecciones a fin de mes (26 días por defecto)
            agg['piezas_proyectadas_mes'] = agg['piezas_actual'] + agg['piezas_por_min'] * agg['min_futuros']
            agg['minutos_proyectados_mes'] = agg['minutos_actual'] + agg['min_futuros']

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👷 Empleados (mes)", int(agg['nombre'].nunique()))
            c2.metric("🧩 Piezas actuales (mes)", int(agg['piezas_actual'].sum()))
            c3.metric("⏱️ Minutos actuales (mes)", int(agg['minutos_actual'].sum()))
            c4.metric("📅 Mes analizado", str(mes_ref))

            # --- TOP / BOTTOM por minutos (mes actual)
            st.markdown("### ⏱️ Empleados con **más y menos** minutos (mes en curso)")
            colA, colB = st.columns(2)
            top_min = agg.sort_values('minutos_actual', ascending=False).head(10)
            bottom_min = agg.sort_values('minutos_actual', ascending=True).head(10)
            with colA:
                st.caption("Top 10 por **minutos**")
                st.dataframe(top_min[['nombre','minutos_actual','dias_trabajados','min_dia_prom']], use_container_width=True)
            with colB:
                st.caption("Bottom 10 por **minutos**")
                st.dataframe(bottom_min[['nombre','minutos_actual','dias_trabajados','min_dia_prom']], use_container_width=True)

            # --- TOP / BOTTOM por piezas (mes actual)
            st.markdown("### 🧩 Empleados con **más y menos** piezas (mes en curso)")
            colC, colD = st.columns(2)
            top_pzs = agg.sort_values('piezas_actual', ascending=False).head(10)
            bottom_pzs = agg.sort_values('piezas_actual', ascending=True).head(10)
            with colC:
                st.caption("Top 10 por **piezas**")
                st.dataframe(top_pzs[['nombre','piezas_actual','piezas_por_min','proyectos']], use_container_width=True)
            with colD:
                st.caption("Bottom 10 por **piezas**")
                st.dataframe(bottom_pzs[['nombre','piezas_actual','piezas_por_min','proyectos']], use_container_width=True)

            # --- Pareto de piezas proyectadas a fin de mes
            st.markdown("### 📈 Pareto de **piezas proyectadas** (a fin de mes)")
            pareto = agg[['nombre','piezas_proyectadas_mes']].sort_values('piezas_proyectadas_mes', ascending=False).reset_index(drop=True)
            total_p = pareto['piezas_proyectadas_mes'].sum()
            pareto['acum'] = pareto['piezas_proyectadas_mes'].cumsum()
            pareto['acum_pct'] = (pareto['acum'] / total_p * 100).round(2) if total_p > 0 else 0

            st.vega_lite_chart(
                {
                    "data": {"values": pareto.to_dict(orient="records")},
                    "layer": [
                        {   # Barras
                            "mark": {"type": "bar"},
                            "encoding": {
                                "x": {"field": "nombre", "type": "nominal", "sort": None, "title": "Empleado"},
                                "y": {"field": "piezas_proyectadas_mes", "type": "quantitative", "title": "Piezas (proy. mes)"}
                            }
                        },
                        {   # Línea % acumulado
                            "mark": {"type": "line", "point": True},
                            "encoding": {
                                "x": {"field": "nombre", "type": "nominal", "sort": None},
                                "y": {"field": "acum_pct", "type": "quantitative", "title": "Acumulado %", "axis": {"grid": False}},
                                "color": {"value": "black"}
                            }
                        }
                    ],
                    "resolve": {"scale": {"y": "independent"}}
                },
                use_container_width=True
            )

            # --- Tabla resumen con análisis y proyección
            st.markdown("### 📄 Resumen por empleado (mes en curso y proyección)")
            cols_show = [
                'nombre','proyectos','dias_trabajados',
                'minutos_actual','min_dia_prom','minutos_proyectados_mes',
                'piezas_actual','piezas_por_min','piezas_proyectadas_mes'
            ]
            resumen = agg[cols_show].sort_values('piezas_proyectadas_mes', ascending=False)
            st.dataframe(resumen, use_container_width=True)

            st.download_button(
                "⬇️ Descargar resumen (CSV)",
                data=resumen.to_csv(index=False).encode('utf-8'),
                file_name=f"resumen_ml_proyecciones_{mes_ref}.csv",
                mime="text/csv"
            )

            st.caption(
                "Metodología: Proyectamos con **26 días laborados**. "
                "Estimamos minutos futuros = promedio de minutos/día × días restantes; "
                "y piezas futuras = tasa (piezas/min) × minutos futuros."
            )


# ==========================================================
# 🔮 Pronóstico diario (con minutos como predictor)
#     Empleado / Proyecto / Proceso  —  BLOQUE AUTOSUFICIENTE
# ==========================================================
st.markdown("### 🔮 Pronóstico diario (con minutos como predictor)")

import numpy as np
from sklearn.linear_model import Ridge

# --- Bootstrap de datos para este bloque ---
# Columna de minutos disponible o la calculamos
min_col = 'minutos_ajustados' if 'minutos_ajustados' in df.columns else (
    'tiempo_minutos' if 'tiempo_minutos' in df.columns else None
)
if min_col is None:
    if {'fecha_inicio_dt','fecha_fin_dt'}.issubset(df.columns):
        df['tiempo_minutos'] = (df['fecha_fin_dt'] - df['fecha_inicio_dt']).dt.total_seconds()/60
        min_col = 'tiempo_minutos'
    else:
        st.error("No encuentro minutos ni puedo calcularlos (faltan fecha_inicio_dt/fecha_fin_dt).")
        st.stop()

df_ml = df.copy()
df_ml['piezas'] = pd.to_numeric(df_ml['piezas'], errors='coerce').fillna(0)
df_ml[min_col] = pd.to_numeric(df_ml[min_col], errors='coerce').fillna(0)
df_ml['fecha'] = df_ml['fecha_inicio_dt'].dt.date

# --- Configuración de nivel ---
nivel = st.selectbox("Nivel de pronóstico", ["Empleado", "Proyecto", "Proceso"])
col_key = {"Empleado":"nombre", "Proyecto":"proyecto", "Proceso":"proceso"}[nivel]
if col_key not in df_ml.columns:
    st.error(f"Falta la columna '{col_key}' en los datos.")
    st.stop()

keys = sorted(df_ml[col_key].dropna().unique().tolist())
if not keys:
    st.info(f"No hay datos para {nivel.lower()}.")
    st.stop()

sel_key = st.selectbox(nivel, keys)

# Horizonte
modo_hz = st.radio("Horizonte", ["Hasta fin de mes", "N días"], horizontal=True)
n_dias = st.number_input("N días de pronóstico", 1, 60, 14) if modo_hz == "N días" else None

# Serie diaria
serie = (df_ml[df_ml[col_key] == sel_key]
            .groupby('fecha', as_index=False)
            .agg(piezas=('piezas','sum'),
                 minutos=(min_col,'sum'))).sort_values('fecha')

if serie.empty:
    st.info("Sin datos para esa selección.")
    st.stop()

s = serie.set_index(pd.to_datetime(serie['fecha']))
# Features
s['lag1'] = s['piezas'].shift(1).fillna(0)
s['lag7'] = s['piezas'].shift(7).fillna(0)
s['dow']  = s.index.dayofweek
s['min']      = s['minutos']
s['min_lag1'] = s['minutos'].shift(1).fillna(s['minutos'].mean())
s['pzs_ma7']  = s['piezas'].rolling(7, min_periods=1).mean()

# Train/Test
test_days = min(7, max(1, len(s)//5))
train = s.iloc[:-test_days] if len(s) > test_days else s
test  = s.iloc[-test_days:] if len(s) > test_days else s.iloc[0:0]

Xtr = train[['lag1','lag7','dow','min','min_lag1','pzs_ma7']]
ytr = train['piezas']
Xte = test[['lag1','lag7','dow','min','min_lag1','pzs_ma7']]
yte = test['piezas']

model = Ridge(alpha=1.0, random_state=42)
model.fit(Xtr, ytr)
if len(Xte) > 0:
    pred_te = model.predict(Xte)
    mae = float(np.mean(np.abs(pred_te - yte)))
    st.metric("MAE (últimos días)", round(mae, 2))

# Horizonte a pronosticar
last_date = s.index.max()
if modo_hz == "Hasta fin de mes":
    end_of_month = last_date.to_period('M').asfreq('M').to_timestamp()
    horizon = max((end_of_month.date() - last_date.date()).days, 0)
else:
    horizon = int(n_dias)

# Minutos futuros supuestos = promedio últimos 7 días (fallback al global)
min_prom = s['minutos'].tail(7).mean()
if not np.isfinite(min_prom) or min_prom == 0:
    min_prom = max(1.0, s['minutos'].mean())

# Forecast iterativo
hist = s[['piezas','lag1','lag7','dow','min','min_lag1','pzs_ma7']].copy()
preds = []
cur = last_date
for _ in range(horizon):
    cur = cur + pd.Timedelta(days=1)
    dow = cur.dayofweek
    lag1 = hist.iloc[-1]['piezas']
    lag7 = hist.iloc[-7]['piezas'] if len(hist) >= 7 else 0
    min_today = min_prom
    min_lag1 = hist.iloc[-1]['min']
    pzs_ma7  = hist['piezas'].tail(7).mean()

    x = np.array([[lag1, lag7, dow, min_today, min_lag1, pzs_ma7]])
    yhat = max(0.0, float(model.predict(x)))
    preds.append({"fecha": cur.date(), "piezas_pred": yhat, "minutos_supuestos": min_today})

    hist.loc[cur] = [yhat, yhat, lag7, dow, min_today, min_today, (pzs_ma7*6 + yhat)/7 if len(hist)>=6 else yhat]

df_pred = pd.DataFrame(preds)

# Gráfico
chart_df = pd.concat([
    serie[['fecha','piezas']].assign(tipo='real'),
    df_pred.rename(columns={'piezas_pred':'piezas'}).assign(tipo='forecast')
], ignore_index=True)

st.vega_lite_chart(
    {
        "data": {"values": chart_df.to_dict(orient="records")},
        "mark": "line",
        "encoding": {
            "x": {"field": "fecha", "type": "temporal", "title": "Fecha"},
            "y": {"field": "piezas", "type": "quantitative", "title": "Piezas"},
            "color": {"field": "tipo", "type": "nominal"}
        }
    },
    use_container_width=True
)

st.write(f"**Total histórico del mes (hasta {last_date.date()}):** {int(serie['piezas'].sum())} piezas")
st.write(f"**Pronóstico en el horizonte seleccionado:** {int(df_pred['piezas_pred'].sum()) if not df_pred.empty else 0} piezas")
st.dataframe(df_pred, use_container_width=True)