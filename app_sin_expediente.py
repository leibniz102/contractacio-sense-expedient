"""
Dashboard de Análisis: Contratación Sin Expediente 2024 vs 2025
Universitat Jaume I - Vicegerència de Recursos Humans
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ============================================================
# INFORMACIÓN DE VERSIÓN
# ============================================================
__version__ = "1.1.0"
__fecha_version__ = "2025-01-18"
__autor__ = "Vicegerència de Recursos Humans - UJI"
__changelog__ = """
v1.1.0 (2025-01-18): Suport Streamlit Cloud
- Afegit file_uploader per pujar Excel des de la interfície
- Compatible amb Streamlit Cloud sense necessitat de fitxers locals

v1.0.0 (2025-01-18): Versió inicial
- Comparativa completa 2024 vs 2025
- Anàlisi detallat de viatges
- Anàlisi detallat de publicacions científiques
- Exportació a Excel
"""

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
RUTA_BASE = Path(__file__).parent
RUTA_DATOS = RUTA_BASE / 'datos'
RUTA_RECURSOS = RUTA_BASE / 'recursos'
RUTA_SALIDAS = RUTA_BASE / 'salidas'
RUTA_PARQUET = RUTA_BASE / 'parquet'
ARCHIVO_DATOS = RUTA_DATOS / 'SIN EXPEDIENTE.xlsx'
ARCHIVO_LOGO = RUTA_RECURSOS / 'logo_uji.png'

# ============================================================
# FUNCIONES DE CACHÉ PARQUET
# ============================================================
def obtener_ruta_parquet(ruta_origen: Path, nombre_hoja: str = None) -> Path:
    """Genera ruta del archivo Parquet en carpeta parquet/."""
    RUTA_PARQUET.mkdir(exist_ok=True)
    nombre_base = ruta_origen.stem
    if nombre_hoja:
        nombre_hoja_clean = nombre_hoja.replace(' ', '_').replace('/', '_')
        return RUTA_PARQUET / f"{nombre_base}_{nombre_hoja_clean}.parquet"
    return RUTA_PARQUET / f"{nombre_base}.parquet"


def necesita_regenerar_parquet(ruta_origen: Path, ruta_parquet: Path) -> bool:
    """Compara timestamps: True si el archivo fuente es más reciente que Parquet."""
    if not ruta_parquet.exists():
        return True
    return ruta_origen.stat().st_mtime > ruta_parquet.stat().st_mtime


def guardar_parquet(df: pd.DataFrame, ruta_parquet: Path) -> None:
    """Guarda DataFrame a Parquet manejando tipos mixtos."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                non_null = df_clean[col].dropna()
                if len(non_null) > 0:
                    types = set(type(x).__name__ for x in non_null)
                    if len(types) > 1:
                        df_clean[col] = df_clean[col].astype(str).replace('nan', pd.NA)
            except Exception:
                df_clean[col] = df_clean[col].astype(str).replace('nan', pd.NA)
    df_clean.to_parquet(ruta_parquet, engine='pyarrow', compression='snappy', index=False)


@st.cache_data(ttl=3600)
def cargar_datos(ruta: Path, nombre_hoja: str) -> pd.DataFrame:
    """Carga datos con caché Parquet."""
    ruta_parquet = obtener_ruta_parquet(ruta, nombre_hoja)

    if necesita_regenerar_parquet(ruta, ruta_parquet):
        df = pd.read_excel(ruta, sheet_name=nombre_hoja)
        guardar_parquet(df, ruta_parquet)
        return df
    else:
        return pd.read_parquet(ruta_parquet)


# ============================================================
# FUNCIONES DE FORMATO
# ============================================================
def formatear_numero(valor: float, decimales: int = 2) -> str:
    """Formatea número al estándar español."""
    if pd.isna(valor):
        return "-"
    formatted = f"{valor:,.{decimales}f}"
    return formatted.replace(',', 'X').replace('.', ',').replace('X', '.')


def formatear_euro(valor: float) -> str:
    """Formatea número como euros."""
    return f"{formatear_numero(valor)} €"


# ============================================================
# FUNCIONES DE ANÁLISIS
# ============================================================
def clasificar_gastos(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica los gastos por categorías."""
    df = df.copy()

    categorias = {
        'Publicacions científiques': [
            'publicación', 'journal', 'article', 'submission', 'publishing',
            'mdpi', 'springer', 'elsevier', 'acs publication', 'open access',
            'copyright clearance', 'wiley', 'frontiers', 'plos'
        ],
        'Inscripcions i congressos': [
            'inscripción', 'inscripció', 'congress', 'conference', 'congreso',
            'seminari', 'workshop', 'symposium', 'jornada', 'registration'
        ],
        'Viatges i transport': [
            'viaje', 'viatge', 'vuelo', 'vol', 'avión', 'taxi', 'transport',
            'billetes', 'holidays', 'tours', 'travel', 'hotel', 'allotjament'
        ],
        'Membresies i quotes': [
            'membresía', 'membership', 'cuota', 'afiliación', 'associació',
            'annual fee', 'suscripción', 'subscripció'
        ],
        'Subministraments (aigua, llum)': [
            'agua', 'facsa', 'regantes', 'electricidad', 'gas', 'energies',
            'suministro', 'subministrament'
        ],
        'Bibliografia i llibres': [
            'bibliogràfic', 'biblioteca', 'libro', 'book', 'adquisició bibliogr'
        ],
        'Material i equipament': [
            'material', 'equip', 'compra', 'adquisició', 'fungible', 'laboratori'
        ],
        'Software i llicències': [
            'software', 'licencia', 'llicència', 'subscription', 'cloud', 'saas'
        ],
    }

    def clasificar(desc):
        if pd.isna(desc):
            return 'Altres'
        desc_lower = str(desc).lower()
        for categoria, keywords in categorias.items():
            if any(kw in desc_lower for kw in keywords):
                return categoria
        return 'Altres'

    df['Categoria'] = df['Desc Gasto'].apply(clasificar)
    return df


def obtener_proveedores_viajes() -> List[str]:
    """Lista de proveedores relacionados con viajes."""
    return [
        'viajes', 'tours', 'holidays', 'travel', 'vueling', 'iberia',
        'ryanair', 'renfe', 'taxi', 'rosselli', 'mago tours', 'mediterráneo',
        'booking', 'amadeus', 'halcón', 'viatge'
    ]


def obtener_proveedores_publicaciones() -> List[str]:
    """Lista de proveedores relacionados con publicaciones."""
    return [
        'springer', 'elsevier', 'wiley', 'acs publication', 'mdpi',
        'frontiers', 'plos', 'nature', 'science', 'taylor & francis',
        'oxford', 'cambridge', 'sage', 'copyright clearance', 'proquest'
    ]


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Contractació Sense Expedient",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data
def cargar_desde_archivo_local():
    """Carga datos desde archivo local (si existe)."""
    df_2025 = cargar_datos(ARCHIVO_DATOS, 'SIN EXPTE 2025')
    df_2024 = cargar_datos(ARCHIVO_DATOS, 'SIN EXPTE 2024')
    return procesar_dataframes(df_2024, df_2025)


@st.cache_data
def cargar_desde_upload(archivo_bytes: bytes):
    """Carga datos desde archivo subido por el usuario."""
    df_2025 = pd.read_excel(io.BytesIO(archivo_bytes), sheet_name='SIN EXPTE 2025')
    df_2024 = pd.read_excel(io.BytesIO(archivo_bytes), sheet_name='SIN EXPTE 2024')
    return procesar_dataframes(df_2024, df_2025)


def procesar_dataframes(df_2024: pd.DataFrame, df_2025: pd.DataFrame):
    """Procesa y limpia los DataFrames."""
    # Limpiar filas de totales
    df_2025 = df_2025[df_2025['Nombre Complet'].notna()].copy()
    df_2024 = df_2024[df_2024['Nombre Complet'].notna()].copy()

    # Clasificar gastos
    df_2025 = clasificar_gastos(df_2025)
    df_2024 = clasificar_gastos(df_2024)

    # Añadir columna de año
    df_2025['Any'] = 2025
    df_2024['Any'] = 2024

    return df_2024, df_2025


# Inicializar variables
datos_cargados = False
df_2024 = None
df_2025 = None
error_msg = ""
archivo_origen = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo
    if ARCHIVO_LOGO.exists():
        st.image(str(ARCHIVO_LOGO), width=150)

    st.title("📊 Contractació Sense Expedient")
    st.caption("Comparativa 2024 vs 2025")
    st.caption("**Universitat Jaume I**")
    st.caption("Vicegerència de Recursos Humans")

    st.divider()

    # ============================================================
    # FILE UPLOADER - Para Streamlit Cloud
    # ============================================================
    st.subheader("📤 Carregar Dades")

    # Verificar si existe archivo local
    archivo_local_existe = ARCHIVO_DATOS.exists()

    if archivo_local_existe:
        st.success("✅ Arxiu local detectat")
        usar_local = st.checkbox("Usar arxiu local", value=True)
    else:
        usar_local = False
        st.info("ℹ️ Puja el fitxer Excel per començar")

    # File uploader
    archivo_subido = st.file_uploader(
        "Puja l'arxiu Excel",
        type=['xlsx', 'xls'],
        help="El fitxer ha de contenir les fulles 'SIN EXPTE 2024' i 'SIN EXPTE 2025'"
    )

    # Lógica de carga de datos
    if archivo_subido is not None:
        # Usuario subió un archivo
        try:
            archivo_bytes = archivo_subido.getvalue()
            df_2024, df_2025 = cargar_desde_upload(archivo_bytes)
            datos_cargados = True
            archivo_origen = archivo_subido.name
        except Exception as e:
            datos_cargados = False
            error_msg = str(e)
    elif usar_local and archivo_local_existe:
        # Usar archivo local
        try:
            df_2024, df_2025 = cargar_desde_archivo_local()
            datos_cargados = True
            archivo_origen = ARCHIVO_DATOS.name
        except Exception as e:
            datos_cargados = False
            error_msg = str(e)

    st.divider()

    if st.button("🔄 Recarregar dades", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Info de datos
    if datos_cargados:
        st.subheader("📂 Dades Carregades")
        st.caption(f"`{archivo_origen}`")
        st.caption(f"2024: {len(df_2024):,} registres")
        st.caption(f"2025: {len(df_2025):,} registres")
        st.divider()

    # Navegación (solo si hay datos)
    if datos_cargados:
        st.subheader("📑 Navegació")
        secciones = [
            "ℹ️ Presentació i Dades",
            "📊 Resum Executiu",
            "📈 Comparativa per Categories",
            "✈️ Anàlisi de Viatges",
            "📚 Anàlisi de Publicacions",
            "🏢 Top Proveïdors",
            "📋 Detall de Registres"
        ]
        seccion = st.radio("Selecciona secció:", secciones, label_visibility="collapsed")
    else:
        seccion = None

    st.divider()

    # Versión
    st.caption(f"📌 Versió {__version__}")
    st.caption(f"📅 {__fecha_version__}")

    with st.expander("📋 Historial de canvis"):
        st.markdown(__changelog__)


# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================
if not datos_cargados:
    st.header("📊 Contractació Sense Expedient")
    st.subheader("Comparativa 2024 vs 2025")

    if error_msg:
        st.error(f"❌ Error carregant les dades: {error_msg}")
    else:
        st.info("""
        ### 👋 Benvingut/da!

        Per començar a utilitzar el dashboard, **puja el fitxer Excel** amb les dades
        de contractació sense expedient a través del panell lateral.

        #### 📋 Requisits del fitxer:
        - Format: `.xlsx` o `.xls`
        - Ha de contenir dues fulles:
          - `SIN EXPTE 2024`
          - `SIN EXPTE 2025`
        - Columnes necessàries: `Nombre Complet`, `N Factura`, `Desc Gasto`, `Base imp`

        #### 🔒 Privacitat:
        Les dades es processen localment en el navegador i **no s'emmagatzemen** al servidor.
        """)

        # Mostrar ejemplo de estructura
        with st.expander("📁 Exemple d'estructura del fitxer"):
            ejemplo = pd.DataFrame({
                'Nombre Complet': ['Proveïdor A', 'Proveïdor B', 'Proveïdor C'],
                'N Factura': ['FAC-001', 'FAC-002', 'FAC-003'],
                'Desc Gasto': ['Descripció del gasto 1', 'Descripció del gasto 2', 'Descripció del gasto 3'],
                'Base imp': [1500.00, 2300.50, 890.25]
            })
            st.dataframe(ejemplo, use_container_width=True, hide_index=True)

    st.stop()

# Métricas globales
total_2024 = df_2024['Base imp'].sum()
total_2025 = df_2025['Base imp'].sum()
diferencia = total_2025 - total_2024
incremento_pct = (diferencia / total_2024) * 100


# ============================================================
# SECCIÓN: PRESENTACIÓN Y DATOS
# ============================================================
if seccion == "ℹ️ Presentació i Dades":
    st.header("ℹ️ Presentació i Dades")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🎯 Objectiu

        Aquesta aplicació proporciona una **anàlisi comparativa exhaustiva** de la contractació
        sense expedient entre els exercicis **2024 i 2025**.

        ### ⚠️ Context

        S'ha detectat un **increment molt significatiu** en la contractació sense expedient,
        superant el **mig milió d'euros** de diferència entre ambdós anys. Aquesta eina permet:

        - 📊 Visualitzar l'evolució dels imports i registres
        - 🏷️ Classificar automàticament els gastos per categories
        - ✈️ Analitzar en detall el sector de viatges
        - 📚 Analitzar en detall les publicacions científiques
        - 🔍 Identificar nous proveïdors i tendències
        """)

    with col2:
        st.markdown("### 📈 Dades actuals")
        st.metric("📋 Registres 2024", f"{len(df_2024):,}")
        st.metric("📋 Registres 2025", f"{len(df_2025):,}")
        st.metric("💰 Increment", formatear_euro(diferencia), f"{incremento_pct:+.1f}%")

    st.divider()

    st.subheader("📊 Mòduls Disponibles")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        **📊 Resum Executiu**
        - Mètriques principals
        - Indicadors clau
        - Visió general
        """)
        st.markdown("""
        **📈 Comparativa per Categories**
        - Classificació automàtica
        - Evolució per tipus de gasto
        """)

    with cols[1]:
        st.markdown("""
        **✈️ Anàlisi de Viatges**
        - Proveïdors de viatges
        - Agències i transport
        - Detall de factures
        """)
        st.markdown("""
        **📚 Anàlisi de Publicacions**
        - Editorials científiques
        - Costos de publicació
        - Open Access
        """)

    with cols[2]:
        st.markdown("""
        **🏢 Top Proveïdors**
        - Rànquing per import
        - Nous vs repetits
        - Increments significatius
        """)
        st.markdown("""
        **📋 Detall de Registres**
        - Taula completa filtrable
        - Exportació a Excel
        """)

    st.divider()

    st.subheader("📁 Fonts de Dades")

    info_fuentes = pd.DataFrame({
        'Arxiu': [archivo_origen if archivo_origen else 'No carregat'],
        'Fulles': ['SIN EXPTE 2024, SIN EXPTE 2025'],
        'Registres 2024': [len(df_2024)],
        'Registres 2025': [len(df_2025)]
    })
    st.dataframe(info_fuentes, use_container_width=True, hide_index=True)


# ============================================================
# SECCIÓN: RESUMEN EJECUTIVO
# ============================================================
elif seccion == "📊 Resum Executiu":
    st.header("📊 Resum Executiu")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Total 2024",
            formatear_euro(total_2024)
        )

    with col2:
        st.metric(
            "💰 Total 2025",
            formatear_euro(total_2025)
        )

    with col3:
        st.metric(
            "📈 Increment Absolut",
            formatear_euro(diferencia),
            f"{incremento_pct:+.1f}%"
        )

    with col4:
        regs_adicionales = len(df_2025) - len(df_2024)
        st.metric(
            "📋 Registres Addicionals",
            f"{regs_adicionales:+,}",
            f"{(regs_adicionales/len(df_2024))*100:+.1f}%"
        )

    st.divider()

    # Gráfico de barras comparativo
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.subheader("📊 Comparativa d'Imports")

        fig_barras = go.Figure()
        fig_barras.add_trace(go.Bar(
            name='2024',
            x=['Import Total', 'Import Mitjà'],
            y=[total_2024, total_2024/len(df_2024)],
            marker_color='#3498db',
            text=[formatear_euro(total_2024), formatear_euro(total_2024/len(df_2024))],
            textposition='outside'
        ))
        fig_barras.add_trace(go.Bar(
            name='2025',
            x=['Import Total', 'Import Mitjà'],
            y=[total_2025, total_2025/len(df_2025)],
            marker_color='#e74c3c',
            text=[formatear_euro(total_2025), formatear_euro(total_2025/len(df_2025))],
            textposition='outside'
        ))
        fig_barras.update_layout(
            barmode='group',
            template='plotly_white',
            separators=',.',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_barras, use_container_width=True)

    with col_graf2:
        st.subheader("📈 Distribució per Rangs d'Import")

        rangos = [0, 100, 500, 1000, 3000, 5000, 10000, 50000, float('inf')]
        etiquetas = ['0-100€', '100-500€', '500-1K€', '1K-3K€', '3K-5K€', '5K-10K€', '10K-50K€', '>50K€']

        df_2024['Rango'] = pd.cut(df_2024['Base imp'], bins=rangos, labels=etiquetas)
        df_2025['Rango'] = pd.cut(df_2025['Base imp'], bins=rangos, labels=etiquetas)

        conteo_2024 = df_2024['Rango'].value_counts().reindex(etiquetas).fillna(0)
        conteo_2025 = df_2025['Rango'].value_counts().reindex(etiquetas).fillna(0)

        fig_rangos = go.Figure()
        fig_rangos.add_trace(go.Bar(
            name='2024',
            x=etiquetas,
            y=conteo_2024.values,
            marker_color='#3498db'
        ))
        fig_rangos.add_trace(go.Bar(
            name='2025',
            x=etiquetas,
            y=conteo_2025.values,
            marker_color='#e74c3c'
        ))
        fig_rangos.update_layout(
            barmode='group',
            template='plotly_white',
            height=400,
            xaxis_title="Rang d'import",
            yaxis_title="Nombre de registres",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_rangos, use_container_width=True)

    st.divider()

    # Análisis de proveedores
    st.subheader("🏢 Anàlisi de Proveïdors")

    proveedores_2024 = set(df_2024['Nombre Complet'].unique())
    proveedores_2025 = set(df_2025['Nombre Complet'].unique())
    nuevos = proveedores_2025 - proveedores_2024
    repetidos = proveedores_2025 & proveedores_2024

    importe_nuevos = df_2025[df_2025['Nombre Complet'].isin(nuevos)]['Base imp'].sum()
    importe_repetidos = df_2025[df_2025['Nombre Complet'].isin(repetidos)]['Base imp'].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Proveïdors 2024", len(proveedores_2024))
    with col2:
        st.metric("Proveïdors 2025", len(proveedores_2025))
    with col3:
        st.metric("🆕 Nous en 2025", len(nuevos))
    with col4:
        st.metric("🔄 Repetits", len(repetidos))

    # Gráfico de pastel
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Proveïdors nous', 'Proveïdors repetits'],
        values=[importe_nuevos, importe_repetidos],
        hole=0.4,
        marker_colors=['#e74c3c', '#3498db'],
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_pie.update_layout(
        title="Distribució import 2025 per tipus de proveïdor",
        template='plotly_white',
        height=350
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.info(f"""
    ⚠️ **Dada crítica**: Els **{len(nuevos)} proveïdors nous** de 2025 aporten
    **{formatear_euro(importe_nuevos)}** ({(importe_nuevos/total_2025)*100:.1f}% del total).
    """)


# ============================================================
# SECCIÓN: COMPARATIVA POR CATEGORÍAS
# ============================================================
elif seccion == "📈 Comparativa per Categories":
    st.header("📈 Comparativa per Categories")

    # Calcular totales por categoría
    cat_2024 = df_2024.groupby('Categoria')['Base imp'].agg(['sum', 'count']).reset_index()
    cat_2024.columns = ['Categoria', 'Import_2024', 'Registres_2024']

    cat_2025 = df_2025.groupby('Categoria')['Base imp'].agg(['sum', 'count']).reset_index()
    cat_2025.columns = ['Categoria', 'Import_2025', 'Registres_2025']

    # Merge
    df_cat = pd.merge(cat_2024, cat_2025, on='Categoria', how='outer').fillna(0)
    df_cat['Diferencia'] = df_cat['Import_2025'] - df_cat['Import_2024']
    df_cat['Variacio_pct'] = np.where(
        df_cat['Import_2024'] > 0,
        (df_cat['Diferencia'] / df_cat['Import_2024']) * 100,
        100
    )
    df_cat = df_cat.sort_values('Import_2025', ascending=False)

    # Gráfico de barras horizontales
    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        name='2024',
        y=df_cat['Categoria'],
        x=df_cat['Import_2024'],
        orientation='h',
        marker_color='#3498db'
    ))
    fig_cat.add_trace(go.Bar(
        name='2025',
        y=df_cat['Categoria'],
        x=df_cat['Import_2025'],
        orientation='h',
        marker_color='#e74c3c'
    ))
    fig_cat.update_layout(
        barmode='group',
        template='plotly_white',
        height=500,
        xaxis_title="Import (€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # Tabla de detalle
    st.subheader("📋 Detall per Categoria")

    df_cat_display = df_cat.copy()
    df_cat_display['Import_2024'] = df_cat_display['Import_2024'].apply(formatear_euro)
    df_cat_display['Import_2025'] = df_cat_display['Import_2025'].apply(formatear_euro)
    df_cat_display['Diferencia'] = df_cat_display['Diferencia'].apply(formatear_euro)
    df_cat_display['Variacio_pct'] = df_cat_display['Variacio_pct'].apply(lambda x: f"{x:+.1f}%")

    df_cat_display = df_cat_display.rename(columns={
        'Import_2024': 'Import 2024',
        'Import_2025': 'Import 2025',
        'Registres_2024': 'Reg. 2024',
        'Registres_2025': 'Reg. 2025',
        'Diferencia': 'Diferència',
        'Variacio_pct': 'Variació %'
    })

    st.dataframe(df_cat_display, use_container_width=True, hide_index=True)


# ============================================================
# SECCIÓN: ANÁLISIS DE VIAJES
# ============================================================
elif seccion == "✈️ Anàlisi de Viatges":
    st.header("✈️ Anàlisi de Viatges i Transport")

    # Filtrar viajes
    keywords_viajes = obtener_proveedores_viajes()
    pattern_viajes = '|'.join(keywords_viajes)

    df_viajes_2024 = df_2024[
        (df_2024['Nombre Complet'].str.lower().str.contains(pattern_viajes, na=False)) |
        (df_2024['Desc Gasto'].str.lower().str.contains(pattern_viajes, na=False)) |
        (df_2024['Categoria'] == 'Viatges i transport')
    ].copy()

    df_viajes_2025 = df_2025[
        (df_2025['Nombre Complet'].str.lower().str.contains(pattern_viajes, na=False)) |
        (df_2025['Desc Gasto'].str.lower().str.contains(pattern_viajes, na=False)) |
        (df_2025['Categoria'] == 'Viatges i transport')
    ].copy()

    total_viajes_2024 = df_viajes_2024['Base imp'].sum()
    total_viajes_2025 = df_viajes_2025['Base imp'].sum()
    dif_viajes = total_viajes_2025 - total_viajes_2024

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✈️ Viatges 2024", formatear_euro(total_viajes_2024))
    with col2:
        st.metric("✈️ Viatges 2025", formatear_euro(total_viajes_2025))
    with col3:
        st.metric("📈 Increment", formatear_euro(dif_viajes))
    with col4:
        st.metric("📋 Registres 2025", len(df_viajes_2025))

    st.divider()

    # Top proveedores de viajes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏢 Top Proveïdors de Viatges 2024")
        top_viajes_2024 = df_viajes_2024.groupby('Nombre Complet')['Base imp'].sum().sort_values(ascending=False).head(10)
        if not top_viajes_2024.empty:
            fig_v24 = px.bar(
                x=top_viajes_2024.values,
                y=[n[:35] + '...' if len(n) > 35 else n for n in top_viajes_2024.index],
                orientation='h',
                color_discrete_sequence=['#3498db']
            )
            fig_v24.update_layout(
                template='plotly_white',
                height=400,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Import (€)",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig_v24, use_container_width=True)
        else:
            st.info("No hi ha dades de viatges en 2024")

    with col2:
        st.subheader("🏢 Top Proveïdors de Viatges 2025")
        top_viajes_2025 = df_viajes_2025.groupby('Nombre Complet')['Base imp'].sum().sort_values(ascending=False).head(10)
        if not top_viajes_2025.empty:
            fig_v25 = px.bar(
                x=top_viajes_2025.values,
                y=[n[:35] + '...' if len(n) > 35 else n for n in top_viajes_2025.index],
                orientation='h',
                color_discrete_sequence=['#e74c3c']
            )
            fig_v25.update_layout(
                template='plotly_white',
                height=400,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Import (€)",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig_v25, use_container_width=True)
        else:
            st.info("No hi ha dades de viatges en 2025")

    st.divider()

    # Nuevas agencias de viajes
    st.subheader("🆕 Noves Agències i Proveïdors de Viatges en 2025")

    proveedores_viajes_2024 = set(df_viajes_2024['Nombre Complet'].unique())
    nuevos_viajes = df_viajes_2025[~df_viajes_2025['Nombre Complet'].isin(proveedores_viajes_2024)]

    if not nuevos_viajes.empty:
        nuevos_agrupados = nuevos_viajes.groupby('Nombre Complet').agg({
            'Base imp': ['sum', 'count']
        }).reset_index()
        nuevos_agrupados.columns = ['Proveïdor', 'Import Total', 'Registres']
        nuevos_agrupados = nuevos_agrupados.sort_values('Import Total', ascending=False)

        # Formatear
        nuevos_display = nuevos_agrupados.copy()
        nuevos_display['Import Total'] = nuevos_display['Import Total'].apply(formatear_euro)

        st.dataframe(nuevos_display.head(15), use_container_width=True, hide_index=True)

        st.warning(f"""
        ⚠️ **Alerta**: Hi ha **{len(nuevos_agrupados)} nous proveïdors de viatges** en 2025
        que no existien en 2024, amb un import total de **{formatear_euro(nuevos_viajes['Base imp'].sum())}**
        """)
    else:
        st.success("No hi ha nous proveïdors de viatges en 2025")

    # Detalle de facturas de viajes
    with st.expander("📋 Detall de totes les factures de viatges 2025"):
        cols_mostrar = ['Nombre Complet', 'N Factura', 'Desc Gasto', 'Base imp']
        cols_disponibles = [c for c in cols_mostrar if c in df_viajes_2025.columns]
        st.dataframe(
            df_viajes_2025[cols_disponibles].sort_values('Base imp', ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SECCIÓN: ANÁLISIS DE PUBLICACIONES
# ============================================================
elif seccion == "📚 Anàlisi de Publicacions":
    st.header("📚 Anàlisi de Publicacions Científiques")

    # Filtrar publicaciones
    keywords_pub = obtener_proveedores_publicaciones()
    pattern_pub = '|'.join(keywords_pub)

    df_pub_2024 = df_2024[
        (df_2024['Nombre Complet'].str.lower().str.contains(pattern_pub, na=False)) |
        (df_2024['Desc Gasto'].str.lower().str.contains('publicación|submission|article|open access|journal', na=False, regex=True)) |
        (df_2024['Categoria'] == 'Publicacions científiques')
    ].copy()

    df_pub_2025 = df_2025[
        (df_2025['Nombre Complet'].str.lower().str.contains(pattern_pub, na=False)) |
        (df_2025['Desc Gasto'].str.lower().str.contains('publicación|submission|article|open access|journal', na=False, regex=True)) |
        (df_2025['Categoria'] == 'Publicacions científiques')
    ].copy()

    total_pub_2024 = df_pub_2024['Base imp'].sum()
    total_pub_2025 = df_pub_2025['Base imp'].sum()
    dif_pub = total_pub_2025 - total_pub_2024

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Publicacions 2024", formatear_euro(total_pub_2024))
    with col2:
        st.metric("📚 Publicacions 2025", formatear_euro(total_pub_2025))
    with col3:
        if total_pub_2024 > 0:
            pct_pub = ((dif_pub / total_pub_2024) * 100)
            st.metric("📈 Increment", formatear_euro(dif_pub), f"{pct_pub:+.1f}%")
        else:
            st.metric("📈 Increment", formatear_euro(dif_pub))
    with col4:
        st.metric("📋 Registres 2025", len(df_pub_2025))

    st.divider()

    # Top editoriales
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Top Editorials 2024")
        top_pub_2024 = df_pub_2024.groupby('Nombre Complet')['Base imp'].sum().sort_values(ascending=False).head(10)
        if not top_pub_2024.empty:
            fig_p24 = px.bar(
                x=top_pub_2024.values,
                y=[n[:40] + '...' if len(n) > 40 else n for n in top_pub_2024.index],
                orientation='h',
                color_discrete_sequence=['#3498db']
            )
            fig_p24.update_layout(
                template='plotly_white',
                height=400,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Import (€)",
                yaxis_title=""
            )
            st.plotly_chart(fig_p24, use_container_width=True)

    with col2:
        st.subheader("📚 Top Editorials 2025")
        top_pub_2025 = df_pub_2025.groupby('Nombre Complet')['Base imp'].sum().sort_values(ascending=False).head(10)
        if not top_pub_2025.empty:
            fig_p25 = px.bar(
                x=top_pub_2025.values,
                y=[n[:40] + '...' if len(n) > 40 else n for n in top_pub_2025.index],
                orientation='h',
                color_discrete_sequence=['#e74c3c']
            )
            fig_p25.update_layout(
                template='plotly_white',
                height=400,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Import (€)",
                yaxis_title=""
            )
            st.plotly_chart(fig_p25, use_container_width=True)

    st.divider()

    # Comparativa de editoriales principales
    st.subheader("📊 Evolució per Editorial")

    editoriales_principales = ['Springer', 'Elsevier', 'MDPI', 'Wiley', 'ACS', 'Frontiers', 'Copyright Clearance']

    comparativa_ed = []
    for ed in editoriales_principales:
        imp_2024 = df_pub_2024[df_pub_2024['Nombre Complet'].str.lower().str.contains(ed.lower(), na=False)]['Base imp'].sum()
        imp_2025 = df_pub_2025[df_pub_2025['Nombre Complet'].str.lower().str.contains(ed.lower(), na=False)]['Base imp'].sum()
        if imp_2024 > 0 or imp_2025 > 0:
            comparativa_ed.append({
                'Editorial': ed,
                'Import 2024': imp_2024,
                'Import 2025': imp_2025,
                'Diferència': imp_2025 - imp_2024
            })

    if comparativa_ed:
        df_ed = pd.DataFrame(comparativa_ed).sort_values('Import 2025', ascending=False)

        fig_ed = go.Figure()
        fig_ed.add_trace(go.Bar(
            name='2024',
            x=df_ed['Editorial'],
            y=df_ed['Import 2024'],
            marker_color='#3498db'
        ))
        fig_ed.add_trace(go.Bar(
            name='2025',
            x=df_ed['Editorial'],
            y=df_ed['Import 2025'],
            marker_color='#e74c3c'
        ))
        fig_ed.update_layout(
            barmode='group',
            template='plotly_white',
            height=400,
            xaxis_title="Editorial",
            yaxis_title="Import (€)"
        )
        st.plotly_chart(fig_ed, use_container_width=True)

        # Tabla
        df_ed_display = df_ed.copy()
        df_ed_display['Import 2024'] = df_ed_display['Import 2024'].apply(formatear_euro)
        df_ed_display['Import 2025'] = df_ed_display['Import 2025'].apply(formatear_euro)
        df_ed_display['Diferència'] = df_ed_display['Diferència'].apply(formatear_euro)
        st.dataframe(df_ed_display, use_container_width=True, hide_index=True)

    # Detalle de publicaciones
    with st.expander("📋 Detall de factures de publicacions 2025"):
        cols_mostrar = ['Nombre Complet', 'N Factura', 'Desc Gasto', 'Base imp']
        cols_disponibles = [c for c in cols_mostrar if c in df_pub_2025.columns]
        st.dataframe(
            df_pub_2025[cols_disponibles].sort_values('Base imp', ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SECCIÓN: TOP PROVEEDORES
# ============================================================
elif seccion == "🏢 Top Proveïdors":
    st.header("🏢 Top Proveïdors")

    # Top general 2025
    st.subheader("🏆 Top 20 Proveïdors 2025")

    top_2025 = df_2025.groupby('Nombre Complet').agg({
        'Base imp': ['sum', 'count']
    }).reset_index()
    top_2025.columns = ['Proveïdor', 'Import Total', 'Registres']
    top_2025 = top_2025.sort_values('Import Total', ascending=False).head(20)

    fig_top = px.bar(
        top_2025,
        x='Import Total',
        y='Proveïdor',
        orientation='h',
        color='Import Total',
        color_continuous_scale='Reds'
    )
    fig_top.update_layout(
        template='plotly_white',
        height=600,
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # Proveedores con mayor incremento
    st.subheader("📈 Proveïdors amb Major Increment (2024→2025)")

    proveedores_2024 = set(df_2024['Nombre Complet'].unique())
    proveedores_2025 = set(df_2025['Nombre Complet'].unique())
    repetidos = proveedores_2025 & proveedores_2024

    comparativa = []
    for prov in repetidos:
        imp_2024 = df_2024[df_2024['Nombre Complet'] == prov]['Base imp'].sum()
        imp_2025 = df_2025[df_2025['Nombre Complet'] == prov]['Base imp'].sum()
        dif = imp_2025 - imp_2024
        comparativa.append({
            'Proveïdor': prov,
            'Import 2024': imp_2024,
            'Import 2025': imp_2025,
            'Diferència': dif
        })

    df_comp = pd.DataFrame(comparativa).sort_values('Diferència', ascending=False)

    # Top incrementos
    top_incrementos = df_comp.head(15)

    fig_inc = px.bar(
        top_incrementos,
        x='Diferència',
        y='Proveïdor',
        orientation='h',
        color='Diferència',
        color_continuous_scale='RdYlGn_r'
    )
    fig_inc.update_layout(
        template='plotly_white',
        height=500,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Increment (€)"
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    # Tabla
    with st.expander("📋 Taula completa d'increments"):
        df_comp_display = df_comp.copy()
        df_comp_display['Import 2024'] = df_comp_display['Import 2024'].apply(formatear_euro)
        df_comp_display['Import 2025'] = df_comp_display['Import 2025'].apply(formatear_euro)
        df_comp_display['Diferència'] = df_comp_display['Diferència'].apply(formatear_euro)
        st.dataframe(df_comp_display, use_container_width=True, hide_index=True)

    st.divider()

    # Nuevos proveedores
    st.subheader("🆕 Top Nous Proveïdors en 2025")

    nuevos = proveedores_2025 - proveedores_2024
    df_nuevos = df_2025[df_2025['Nombre Complet'].isin(nuevos)]

    top_nuevos = df_nuevos.groupby('Nombre Complet').agg({
        'Base imp': ['sum', 'count']
    }).reset_index()
    top_nuevos.columns = ['Proveïdor', 'Import Total', 'Registres']
    top_nuevos = top_nuevos.sort_values('Import Total', ascending=False).head(15)

    top_nuevos_display = top_nuevos.copy()
    top_nuevos_display['Import Total'] = top_nuevos_display['Import Total'].apply(formatear_euro)

    st.dataframe(top_nuevos_display, use_container_width=True, hide_index=True)

    st.info(f"""
    📊 **Resum nous proveïdors**: {len(nuevos)} nous proveïdors en 2025 amb un import total
    de **{formatear_euro(df_nuevos['Base imp'].sum())}** ({(df_nuevos['Base imp'].sum()/total_2025)*100:.1f}% del total)
    """)


# ============================================================
# SECCIÓN: DETALLE DE REGISTROS
# ============================================================
elif seccion == "📋 Detall de Registres":
    st.header("📋 Detall de Registres")

    # Selector de año
    any_seleccionat = st.radio("Selecciona l'any:", [2024, 2025], horizontal=True)

    df_seleccionado = df_2024 if any_seleccionat == 2024 else df_2025

    # Filtros
    col1, col2, col3 = st.columns(3)

    with col1:
        categorias_disponibles = ['Totes'] + sorted(df_seleccionado['Categoria'].unique().tolist())
        categoria_filtro = st.selectbox("Categoria:", categorias_disponibles)

    with col2:
        proveedores_disponibles = ['Tots'] + sorted(df_seleccionado['Nombre Complet'].unique().tolist())
        proveedor_filtro = st.selectbox("Proveïdor:", proveedores_disponibles)

    with col3:
        importe_min = st.number_input("Import mínim (€):", min_value=0.0, value=0.0)

    # Aplicar filtros
    df_filtrado = df_seleccionado.copy()

    if categoria_filtro != 'Totes':
        df_filtrado = df_filtrado[df_filtrado['Categoria'] == categoria_filtro]

    if proveedor_filtro != 'Tots':
        df_filtrado = df_filtrado[df_filtrado['Nombre Complet'] == proveedor_filtro]

    if importe_min > 0:
        df_filtrado = df_filtrado[df_filtrado['Base imp'] >= importe_min]

    # Métricas del filtro
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registres filtrats", len(df_filtrado))
    with col2:
        st.metric("Import total filtrat", formatear_euro(df_filtrado['Base imp'].sum()))
    with col3:
        st.metric("Import mitjà", formatear_euro(df_filtrado['Base imp'].mean() if len(df_filtrado) > 0 else 0))

    st.divider()

    # Tabla de datos
    cols_mostrar = ['Nombre Complet', 'N Factura', 'Desc Gasto', 'Base imp', 'Categoria']
    cols_disponibles = [c for c in cols_mostrar if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado[cols_disponibles].sort_values('Base imp', ascending=False),
        use_container_width=True,
        hide_index=True,
        height=500
    )

    # Exportar a Excel
    st.divider()

    if st.button("📥 Exportar a Excel", type="primary"):
        RUTA_SALIDAS.mkdir(exist_ok=True)
        archivo_salida = RUTA_SALIDAS / f"contractacio_sense_expedient_{any_seleccionat}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        df_filtrado.to_excel(archivo_salida, index=False)
        st.success(f"✅ Arxiu exportat: `{archivo_salida.name}`")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("📊 Dashboard de Contractació Sense Expedient | Universitat Jaume I | Vicegerència de Recursos Humans")
