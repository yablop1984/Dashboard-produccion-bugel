import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# ===============================
# Configuración e información
# ===============================
st.set_page_config(page_title="Dashboard Producción", layout="wide")
st.title("📊 Dashboard Producción Bugel")

# Enlace público (ajústalo si cambia tu URL)
PUBLIC_URL = "https://yablop1984-dashboard-produccion-bugel-app-2nqz7v.streamlit.app/"

col_top_a, col_top_b = st.columns([1, 2])
with col_top_a:
    # Algunas versiones antiguas de Streamlit no tienen link_button
    try:
        st.link_button("🔗 Abrir versión pública", PUBLIC_URL)
    except Exception:
        st.markdown(f"[🔗 Abrir versión pública]({PUBLIC_URL})")
with col_top_b:
    st.code(PUBLIC_URL, language="text")

st.sidebar.markdown("### Acceso")
try:
    st.sidebar.link_button("🔗 Abrir versión pública", PUBLIC_URL)
except Exception:
    st.sidebar.markdown(f"[🔗 Abrir versión pública]({PUBLIC_URL})")

with st.expander("📱 Cómo instalar la app en el celular"):
    st.markdown(
        """
**Android (Chrome):** abre la URL → menú **⋮** → **Añadir a pantalla de inicio** → **Instalar**.  
**iPhone/iPad (Safari):** abre la URL → botón **Compartir** → **Añadir a pantalla de inicio**.  
> Crea un acceso directo (no APK).
        """
    )

# ===============================
# Carga y limpieza de datos
# ===============================
url = (
    "https://docs.google.com/spreadsheets/"
    "d/1YtUaTmcVQR7N9FvXb5mYmvrUKkOnufwQV9HoF9Gf75I/"
    "export?format=csv&gid=0"
)
df = pd.read_csv(url)

def limpiar_y_convertir_fecha(serie: pd.Series) -> pd.Series:
    serie = serie.astype(str).str.strip()
    serie = serie.str.replace(',', ' ', regex=False)
    serie = serie.str.replace(r'\s+', ' ', regex=True)
    fechas = pd.to_datetime(serie, errors='coerce', dayfirst=True, infer_datetime_format=True)
    mask_nat = fechas.isna()
    if mask_nat.any():
        fechas[mask_nat] = pd.to_datetime(serie[mask_nat], errors='coerce', dayfirst=False, infer_datetime_format=True)
    return fechas

df['fecha_inicio_dt'] = limpiar_y_convertir_fecha(df['fecha_inicio'])
df['fecha_fin_dt'] = limpiar_y_convertir_fecha(df['fecha_fin'])

valid_inicio = df['fecha_inicio_dt'].notna().sum()
valid_fin = df['fecha_fin_dt'].notna().sum()
st.caption(f"✔ Fechas inicio válidas: {valid_inicio}/{len(df)}, Fechas fin válidas: {valid_fin}/{len(df)}")

# Filtrado mínimo y duración
df = df.dropna(subset=['fecha_inicio_dt']).copy()
df['tiempo_minutos'] = (df['fecha_fin_dt'] - df['fecha_inicio_dt']).dt.total_seconds() / 60

# ===============================
# Pestañas
# ===============================
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
    c1, c2, c3 = st.columns(3)
    c1.metric("🧩 Total piezas", int(pd.to_numeric(df['piezas'], errors='coerce').fillna(0).sum()))
    c2.metric("📋 Total registros", len(df))
    prom_global = pd.to_numeric(df['piezas'], errors='coerce').fillna(0).mean() if not df.empty else 0
    c3.metric("📊 Promedio pzas/registro", round(prom_global, 2))

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

    sel_empleados = st.sidebar.multiselect(
        "Empleado", options=sorted(df['nombre'].dropna().unique()),
        default=list(df['nombre'].dropna().unique())
    )
    df_emp = df[df['nombre'].isin(sel_empleados)]

    sel_proyectos = st.sidebar.multiselect(
        "Proyecto", options=sorted(df_emp['proyecto'].dropna().unique()),
        default=list(df_emp['proyecto'].dropna().unique())
    )
    df_proj = df_emp[df_emp['proyecto'].isin(sel_proyectos)]

    sel_maquinas = st.sidebar.multiselect(
        "Máquina", options=sorted(df_proj['maquina'].dropna().unique()),
        default=list(df_proj['maquina'].dropna().unique())
    )
    sel_procesos = st.sidebar.multiselect(
        "Proceso", options=sorted(df_proj['proceso'].dropna().unique()),
        default=list(df_proj['proceso'].dropna().unique())
    )

    fecha_min = df['fecha_inicio_dt'].min().date()
    fecha_max = df['fecha_inicio_dt'].max().date()
    rango_fechas = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])

    if len(rango_fechas) == 2:
        start_date, end_date = rango_fechas
    else:
        start_date, end_date = fecha_min, fecha_max

    df_filtrado = df_proj[
        (df_proj['maquina'].isin(sel_maquinas)) &
        (df_proj['proceso'].isin(sel_procesos)) &
        (df_proj['fecha_inicio_dt'].dt.date >= start_date) &
        (df_proj['fecha_inicio_dt'].dt.date <= end_date)
    ].copy()

    st.write(f"✅ Total registros filtrados: **{len(df_filtrado)}**")

    st.subheader("👷‍♂️ Piezas por Empleado")
    if not df_filtrado.empty:
        st.bar_chart(df_filtrado.groupby('nombre')['piezas'].sum())

    st.subheader("📊 Piezas por Proyecto (Filtrado)")
    if not df_filtrado.empty:
        st.bar_chart(df_filtrado.groupby('proyecto')['piezas'].sum())

    st.subheader("📈 Evolución por Empleado (Filtrado)")
    if not df_filtrado.empty:
        df_line = df_filtrado.copy()
        df_line['fecha'] = df_line['fecha_inicio_dt'].dt.date
        st.line_chart(df_line.groupby(['fecha', 'nombre'])['piezas'].sum().unstack().fillna(0))

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

    st.subheader("📄 Detalle de registros")
    st.dataframe(df_filtrado)
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
# PESTAÑA 4: Último día
# ------------------------------
with tab4:
    st.subheader("🕒 Trabajo del último día ingresado")

    ultima_fecha = df['fecha_inicio_dt'].dt.date.max()
    if pd.isna(ultima_fecha):
        st.info("No hay fechas válidas en el dataset.")
    else:
        st.caption(f"Última fecha detectada en los datos: **{ultima_fecha}**")

        df_last = df[df['fecha_inicio_dt'].dt.date.eq(ultima_fecha)].copy()

        if 'minutos_ajustados' in df_last.columns:
            df_last['min_trabajo'] = df_last['minutos_ajustados'].fillna(0)
        else:
            df_last['min_trabajo'] = df_last['tiempo_minutos'].fillna(0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📋 Registros", int(len(df_last)))
        c2.metric("👷 Empleados", int(df_last['nombre'].nunique()))
        c3.metric("🏗️ Proyectos", int(df_last['proyecto'].nunique()))
        c4.metric("🧩 Piezas", int(df_last['piezas'].fillna(0).sum()))
        c5.metric("⏱️ Tiempo (horas)", round(df_last['min_trabajo'].sum()/60, 2))

        st.subheader("👷 Trabajo por empleado (último día)")
        tabla_emp = (
            df_last.groupby('nombre', as_index=False)
                   .agg(piezas=('piezas','sum'),
                        minutos=('min_trabajo','sum'),
                        proyectos_distintos=('proyecto','nunique'))
                   .sort_values(['piezas','minutos'], ascending=False)
        )
        st.dataframe(tabla_emp, use_container_width=True)

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
                        {"mark": {"type": "bar"},
                         "encoding": {
                             "x": {"field": "nombre", "type": "nominal", "sort": None, "title": "Empleado"},
                             "y": {"field": "piezas", "type": "quantitative", "title": "Piezas"}
                         }},
                        {"mark": {"type": "line", "point": True},
                         "encoding": {
                             "x": {"field": "nombre", "type": "nominal", "sort": None},
                             "y": {"field": "acum_pct", "type": "quantitative", "title": "Acumulado %", "axis": {"grid": False}},
                             "color": {"value": "black"}
                         }}
                    ],
                    "resolve": {"scale": {"y": "independent"}},
                },
                use_container_width=True
            )
            st.caption("Consejo: si prefiere Pareto por proyecto, cambie el agrupamiento a 'proyecto'.")

        st.subheader("📄 Detalle del último día")
        st.dataframe(df_last, use_container_width=True)
        st.download_button(
            "⬇️ Descargar registros del último día (CSV)",
            data=df_last.to_csv(index=False).encode('utf-8'),
            file_name=f"registros_{ultima_fecha}.csv",
            mime="text/csv"
        )

# ------------------------------
# PESTAÑA 5: ML / Proyecciones (unificada)
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

            # ---------- Agregación EMPLEADOS ----------
            agg = (
                df_mes.groupby('nombre', as_index=False)
                      .agg(
                          piezas_actual=('piezas','sum'),
                          minutos_actual=(min_col,'sum'),
                          dias_trabajados=('fecha','nunique'),
                          proyectos=('proyecto','nunique')
                      )
            )
            agg['min_dia_prom'] = np.where(agg['dias_trabajados']>0, agg['minutos_actual']/agg['dias_trabajados'], 0)
            agg['piezas_por_min'] = np.where(agg['minutos_actual']>0, agg['piezas_actual']/agg['minutos_actual'], 0)
            agg['dias_restantes'] = (dias_obj - agg['dias_trabajados']).clip(lower=0)
            agg['min_futuros'] = agg['min_dia_prom'] * agg['dias_restantes']
            agg['piezas_proyectadas_mes'] = agg['piezas_actual'] + agg['piezas_por_min'] * agg['min_futuros']
            agg['minutos_proyectados_mes'] = agg['minutos_actual'] + agg['min_futuros']

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("👷 Empleados (mes)", int(agg['nombre'].nunique()))
            k2.metric("🧩 Piezas actuales (mes)", int(agg['piezas_actual'].sum()))
            k3.metric("⏱️ Minutos actuales (mes)", int(agg['minutos_actual'].sum()))
            k4.metric("📅 Mes analizado", str(mes_ref))

            # Helpers de ranking 1..10
            def rank_top(df_, col, ascending=False, cols_keep=None, titulo=""):
                t = df_.sort_values(col, ascending=ascending).head(10).reset_index(drop=True)
                t.insert(0, "#", t.index + 1)
                if cols_keep:
                    t = t[['#'] + cols_keep]
                st.caption(titulo)
                st.dataframe(t, use_container_width=True)

            st.markdown("### ⏱️ Empleados con **más y menos** minutos (mes en curso)")
            ca, cb = st.columns(2)
            with ca:
                rank_top(
                    agg, 'minutos_actual', ascending=False,
                    cols_keep=['nombre','minutos_actual','dias_trabajados','min_dia_prom'],
                    titulo="Top 10 por **minutos**"
                )
            with cb:
                rank_top(
                    agg, 'minutos_actual', ascending=True,
                    cols_keep=['nombre','minutos_actual','dias_trabajados','min_dia_prom'],
                    titulo="Bottom 10 por **minutos**"
                )

            st.markdown("### 🧩 Empleados con **más y menos** piezas (mes en curso)")
            cc, cd = st.columns(2)
            with cc:
                rank_top(
                    agg, 'piezas_actual', ascending=False,
                    cols_keep=['nombre','piezas_actual','piezas_por_min','proyectos'],
                    titulo="Top 10 por **piezas**"
                )
            with cd:
                rank_top(
                    agg, 'piezas_actual', ascending=True,
                    cols_keep=['nombre','piezas_actual','piezas_por_min','proyectos'],
                    titulo="Bottom 10 por **piezas**"
                )

            # ---------- Pareto con toggle de Fuente y Dimensión ----------
            st.markdown("### 📈 Pareto (elige Fuente y Dimensión)")
            p1, p2 = st.columns(2)
            with p1:
                fuente_pareto = st.radio("Fuente", ["Actual", "Proyectado"], horizontal=True, index=1)
            with p2:
                dim_pareto = st.radio("Dimensión", ["Empleado", "Proyecto", "Proceso"], horizontal=True, index=0)

            dim_map = {"Empleado": ("nombre", "Empleado"),
                       "Proyecto": ("proyecto", "Proyecto"),
                       "Proceso":  ("proceso",  "Proceso")}
            key_col, key_title = dim_map[dim_pareto]

            daily_dim = (df_mes.groupby([key_col, 'fecha'], as_index=False)
                              .agg(piezas=('piezas','sum'),
                                   minutos=(min_col,'sum')))
            agg_dim = (daily_dim.groupby(key_col, as_index=False)
                               .agg(dias_trabajados=('fecha','nunique'),
                                    piezas_actual=('piezas','sum'),
                                    minutos_actual=('minutos','sum')))
            agg_dim['min_dia_prom'] = np.where(agg_dim['dias_trabajados']>0, agg_dim['minutos_actual']/agg_dim['dias_trabajados'], 0)
            agg_dim['piezas_por_min'] = np.where(agg_dim['minutos_actual']>0, agg_dim['piezas_actual']/agg_dim['minutos_actual'], 0)
            agg_dim['dias_restantes'] = (dias_obj - agg_dim['dias_trabajados']).clip(lower=0)
            agg_dim['min_futuros'] = agg_dim['min_dia_prom'] * agg_dim['dias_restantes']
            agg_dim['piezas_proyectadas_mes'] = agg_dim['piezas_actual'] + agg_dim['piezas_por_min'] * agg_dim['min_futuros']

            if fuente_pareto == "Actual":
                pareto_df = agg_dim[[key_col, 'piezas_actual']].rename(columns={'piezas_actual': 'valor'})
                y_title = f"Piezas (mes actual) por {key_title}"
            else:
                pareto_df = agg_dim[[key_col, 'piezas_proyectadas_mes']].rename(columns={'piezas_proyectadas_mes': 'valor'})
                y_title = f"Piezas (proyectadas) por {key_title}"

            pareto_df = pareto_df.sort_values('valor', ascending=False).reset_index(drop=True)
            total_val = pareto_df['valor'].sum()
            pareto_df['acum'] = pareto_df['valor'].cumsum()
            pareto_df['acum_pct'] = (pareto_df['acum'] / total_val * 100).round(2) if total_val > 0 else 0

            st.vega_lite_chart(
                {
                    "data": {"values": pareto_df.to_dict(orient="records")},
                    "layer": [
                        {"mark": {"type": "bar"},
                         "encoding": {
                             "x": {"field": key_col, "type": "nominal", "sort": None, "title": key_title},
                             "y": {"field": "valor", "type": "quantitative", "title": y_title}
                         }},
                        {"mark": {"type": "line", "point": True},
                         "encoding": {
                             "x": {"field": key_col, "type": "nominal", "sort": None},
                             "y": {"field": "acum_pct", "type": "quantitative", "title": "Acumulado %", "axis": {"grid": False}},
                             "color": {"value": "black"}
                         }}
                    ],
                    "resolve": {"scale": {"y": "independent"}}
                },
                use_container_width=True
            )

            st.download_button(
                "⬇️ Descargar Pareto mostrado (CSV)",
                data=pareto_df.rename(columns={key_col: key_title}).to_csv(index=False).encode('utf-8'),
                file_name=f"pareto_{fuente_pareto.lower()}_{key_col}.csv",
                mime="text/csv"
            )

            st.markdown("### 📄 Resumen por empleado (mes y proyección)")
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
                file_name=f"resumen_proyecciones_{mes_ref}.csv",
                mime="text/csv"
            )

            st.caption(
                "Metodología: Proyección con meta de **26 días laborados**. "
                "Minutos futuros = promedio minutos/día × días faltantes; "
                "Piezas futuras = tasa (piezas/min) × minutos futuros."
            )

            # ----------------------------------------------------------
            # 🔮 PRONÓSTICO DIARIO (minutos + piezas) — con selector mensual
            # ----------------------------------------------------------
            st.markdown("---")
            st.subheader("🔮 Pronóstico diario (con minutos como predictor)")

            df_ml = df.copy()
            df_ml['piezas'] = pd.to_numeric(df_ml['piezas'], errors='coerce').fillna(0)
            df_ml[min_col] = pd.to_numeric(df_ml[min_col], errors='coerce').fillna(0)
            df_ml['fecha'] = df_ml['fecha_inicio_dt'].dt.date

            # --- NIVEL DEL PRONÓSTICO (incluye Proyecto) ---
            nivel = st.selectbox(
                "Nivel de pronóstico",
                ["Empleado", "Proyecto", "Proceso", "Máquina"],
                index=0
            )

            col_key = {
                "Empleado": "nombre",
                "Proyecto": "proyecto",
                "Proceso":  "proceso",
                "Máquina":  "maquina",
            }[nivel]

            keys = sorted(df_ml[col_key].dropna().unique().tolist())

            if not keys:
                st.info(f"No hay datos para {nivel.lower()}.")
            else:
                sel_key = st.selectbox(nivel, keys)
                modo_hz = st.radio("Horizonte", ["Hasta fin de mes", "N días", "Hasta mes objetivo"], horizontal=True, index=0)

                # Meses desde el mes actual hasta diciembre del año en curso
                today = pd.Timestamp.today()
                year = today.year
                months = [pd.Period(f"{year}-{m:02d}") for m in range(today.month, 13)]
                month_labels = [p.strftime("%b %Y") for p in months]
                month_ends = [p.asfreq('M').to_timestamp().date() for p in months]

                if modo_hz == "N días":
                    n_dias = st.number_input("N días de pronóstico", 1, 120, 14)
                    fecha_fin_objetivo = None
                elif modo_hz == "Hasta mes objetivo":
                    if month_labels:
                        sel_label = st.select_slider("Mes objetivo", options=month_labels, value=month_labels[0])
                        fecha_fin_objetivo = month_ends[month_labels.index(sel_label)]
                    else:
                        fecha_fin_objetivo = today.to_period('M').asfreq('M').to_timestamp().date()
                else:
                    fecha_fin_objetivo = None  # fin de mes del último dato

                # Serie diaria agregada por la clave seleccionada
                serie = (df_ml[df_ml[col_key] == sel_key]
                            .groupby('fecha', as_index=False)
                            .agg(piezas=('piezas', 'sum'),
                                 minutos=(min_col, 'sum'))
                         ).sort_values('fecha')

                if serie.empty:
                    st.info("Sin datos para esa selección.")
                else:
                    s = serie.set_index(pd.to_datetime(serie['fecha']))
                    s['lag1'] = s['piezas'].shift(1).fillna(0)
                    s['lag7'] = s['piezas'].shift(7).fillna(0)
                    s['dow'] = s.index.dayofweek
                    s['min'] = s['minutos']
                    s['min_lag1'] = s['minutos'].shift(1).fillna(s['minutos'].mean())
                    s['pzs_ma7'] = s['piezas'].rolling(7, min_periods=1).mean()

                    test_days = min(7, max(1, len(s) // 5))
                    train = s.iloc[:-test_days] if len(s) > test_days else s
                    test = s.iloc[-test_days:] if len(s) > test_days else s.iloc[0:0]

                    Xtr = train[['lag1', 'lag7', 'dow', 'min', 'min_lag1', 'pzs_ma7']]
                    ytr = train['piezas']
                    Xte = test[['lag1', 'lag7', 'dow', 'min', 'min_lag1', 'pzs_ma7']]
                    yte = test['piezas']

                    model = Ridge(alpha=1.0, random_state=42)
                    model.fit(Xtr, ytr)
                    if len(Xte) > 0:
                        pred_te = model.predict(Xte)
                        mae = float(np.mean(np.abs(pred_te - yte)))
                        st.metric("MAE (últimos días)", round(mae, 2))

                    # Horizonte
                    last_date = s.index.max()
                    if modo_hz == "Hasta fin de mes":
                        end_of_month = last_date.to_period('M').asfreq('M').to_timestamp()
                        horizon = max((end_of_month.date() - last_date.date()).days, 0)
                        fecha_fin_forecast = end_of_month.date()
                    elif modo_hz == "N días":
                        horizon = int(n_dias)
                        fecha_fin_forecast = (last_date + pd.Timedelta(days=horizon)).date()
                    else:  # Hasta mes objetivo
                        fecha_fin_forecast = fecha_fin_objetivo
                        horizon = max((pd.Timestamp(fecha_fin_forecast) - last_date).days, 0)

                    st.caption(
                        f"📆 Proyección desde **{last_date.date()}** hasta **{fecha_fin_forecast}** "
                        f"(**{horizon}** día(s) de pronóstico)."
                    )

                    if horizon == 0:
                        st.info("No hay días por proyectar para esta selección en el horizonte elegido.")
                        st.vega_lite_chart(
                            {
                                "data": {"values": serie[['fecha', 'piezas']].assign(tipo='real').to_dict(orient="records")},
                                "mark": "line",
                                "encoding": {
                                    "x": {"field": "fecha", "type": "temporal", "title": "Fecha"},
                                    "y": {"field": "piezas", "type": "quantitative", "title": "Piezas"},
                                    "color": {"field": "tipo", "type": "nominal"}
                                }
                            },
                            use_container_width=True
                        )
                    else:
                        # Minutos futuros supuestos = promedio últimos 7 días (fallback al global/1.0)
                        min_prom = s['minutos'].tail(7).mean()
                        if not np.isfinite(min_prom) or min_prom == 0:
                            min_prom = max(1.0, s['minutos'].mean())

                        # Forecast iterativo
                        hist = s[['piezas', 'lag1', 'lag7', 'dow', 'min', 'min_lag1', 'pzs_ma7']].copy()
                        preds = []
                        cur = last_date
                        for _ in range(horizon):
                            cur = cur + pd.Timedelta(days=1)
                            dow = cur.dayofweek
                            lag1 = hist.iloc[-1]['piezas']
                            lag7 = hist.iloc[-7]['piezas'] if len(hist) >= 7 else 0
                            min_today = float(min_prom)
                            min_lag1 = hist.iloc[-1]['min']
                            pzs_ma7 = hist['piezas'].tail(7).mean()

                            x = np.array([[lag1, lag7, dow, min_today, min_lag1, pzs_ma7]])
                            yhat = max(0.0, float(model.predict(x)))
                            preds.append({"fecha": cur.date(), "piezas_pred": yhat, "minutos_supuestos": min_today})

                            hist.loc[cur] = [
                                yhat, yhat, lag7, dow, min_today, min_today,
                                (pzs_ma7 * 6 + yhat) / 7 if len(hist) >= 6 else yhat
                            ]

                        df_pred = pd.DataFrame(preds)

                        # Gráfico
                        chart_df = pd.concat([
                            serie[['fecha', 'piezas']].assign(tipo='real'),
                            df_pred.rename(columns={'piezas_pred': 'piezas'}).assign(tipo='forecast')
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

                        # Totales del horizonte y resumen mensual del forecast
                        tot_min_pred = int(df_pred['minutos_supuestos'].sum()) if not df_pred.empty else 0
                        tot_pzs_pred = int(df_pred['piezas_pred'].sum()) if not df_pred.empty else 0
                        m1, m2 = st.columns(2)
                        m1.metric("⏱️ Minutos pronosticados (hasta objetivo)", tot_min_pred)
                        m2.metric("🧩 Piezas pronosticadas (hasta objetivo)", tot_pzs_pred)

                        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
                        df_pred['mes'] = df_pred['fecha'].dt.to_period('M')
                        resumen_mes = (df_pred.groupby('mes', as_index=False)
                                                .agg(minutos=('minutos_supuestos', 'sum'),
                                                     piezas=('piezas_pred', 'sum')))
                        st.subheader("📅 Resumen por mes (solo forecast)")
                        st.dataframe(resumen_mes, use_container_width=True)

                        st.write(f"**Total histórico del mes (hasta {last_date.date()}):** {int(serie['piezas'].sum())} piezas")
                        st.write(f"**Pronóstico hasta {fecha_fin_forecast}:** {tot_pzs_pred} piezas, {tot_min_pred} minutos")