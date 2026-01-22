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
import json

# ============================================================
# INFORMACIÓN DE VERSIÓN
# ============================================================
__version__ = "1.5.3"
__fecha_version__ = "2026-01-22"
__autor__ = "Vicegerència de Recursos Humans - UJI"
# Historial completo de cambios en: CHANGELOG.md

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
        # Publicaciones científicas
        'Publicacions científiques': [
            'publicación', 'journal', 'article', 'submission', 'publishing',
            'mdpi', 'springer', 'elsevier', 'acs publication', 'open access',
            'copyright clearance', 'wiley', 'frontiers', 'plos', 'sage publications',
            'taylor & francis', 'nature', 'science direct', 'ieee', 'acm digital'
        ],
        # Derechos reprográficos (CEDRO)
        'Drets reprogràfics (CEDRO)': [
            'cedro', 'derechos reprograficos', 'centro español de derechos',
            'reprografia', 'drets reprograf'
        ],
        # Inscripciones y congresos
        'Inscripcions i congressos': [
            'inscripción', 'inscripció', 'congress', 'conference', 'congreso',
            'seminari', 'workshop', 'symposium', 'jornada', 'registration',
            'full pass', 'annual meeting'
        ],
        # Formación y cursos (diferente de congresos)
        'Formació i cursos': [
            'curso', 'curs ', 'formación', 'formació', 'capacitación',
            'taller ', 'máster', 'master ', 'doctorado', 'certificado',
            'técnicas de', 'tècniques de'
        ],
        # Colaboradores docentes y prácticas
        'Col·laboradors docents': [
            'colaborador docente', 'col·laborador docent', 'supervisor práctic',
            'prácticas externas', 'pràctiques externes', 'tutor extern',
            'professorat col·laborador'
        ],
        # Viajes y transporte
        'Viatges i transport': [
            'viaje', 'viatge', 'vuelo', 'vol ', 'avión', 'taxi', 'transport',
            'billetes', 'holidays', 'tours', 'travel', 'hotel', 'allotjament',
            'alojamiento', 'desplazamiento', 'mago tours', 'mediterráneo holiday'
        ],
        # Membresías y cuotas
        'Membresies i quotes': [
            'membresía', 'membership', 'cuota', 'quota', 'afiliación', 'associació',
            'annual fee', 'suscripción', 'subscripció', 'aneca', 'crue', 'ruvid'
        ],
        # Programa Pisos Solidaris
        'Programa Pisos Solidaris': [
            'pisos solidaris', 'pisos-solidaris', 'pis solidari',
            'programa pisos', 'bloc 2', 'bloc 3', 'bloc 4'
        ],
        # Suministros (agua, luz, gas)
        'Subministraments (aigua, llum)': [
            'agua', 'aigua', 'facsa', 'regantes', 'electricidad', 'llum',
            'gas natural', 'energies', 'suministro', 'subministrament',
            'consum elèctric', 'totalenergies'
        ],
        # Reprografía y fotocopiadoras
        'Reprografia i fotocopiadores': [
            'fotocopiadora', 'fotocopias', 'fotocòpies', 'impresora',
            'contador', 'manteniment fotocopiadora', 'multifunción',
            'copistería', 'format, s.l', 'impressió'
        ],
        # Mensajería y envíos
        'Missatgeria i enviaments': [
            'envíos', 'enviaments', 'envios', 'mensajería', 'missatgeria',
            'urgent', 'azahar urgent', 'dachser', 'correos', 'paquetería',
            'enviaments de llibres'
        ],
        # Restauración y catering
        'Restauració i càtering': [
            'comida', 'dinar', 'menú', 'restauració', 'catering', 'càtering',
            'almuerzo', 'cena', 'dietes', 'servei de restauració', 'coffee break'
        ],
        # Servicios legales y asesoría
        'Servicis legals i assessoria': [
            'abogacía', 'procura', 'honorarios', 'letrado', 'jurídic',
            'asesoría', 'assessoria', 'notaría', 'notaria', 'tasas judiciales'
        ],
        # Mantenimiento e infraestructura
        'Manteniment i infraestructura': [
            'manteniment', 'mantenimiento', 'reparación', 'reparació',
            'servidor', 'nube', 'hosting', 'infraestructura'
        ],
        # Prensa y comunicación
        'Premsa i comunicació': [
            'radio', 'prensa', 'periódico', 'publicitat', 'publicidad',
            'cope', 'mediterráneo', 'comunicación', 'multimedia', 'eco3'
        ],
        # Bibliografía y libros
        'Bibliografia i llibres': [
            'bibliogràfic', 'biblioteca', 'libro', 'llibre', 'book',
            'adquisició bibliogr', 'adquisicions bibliogràfiques', 'proquest'
        ],
        # Material y equipamiento
        'Material i equipament': [
            'material', 'equip', 'compra', 'adquisició', 'fungible',
            'laboratori', 'reactiu', 'químic', 'omron', 'instrumental'
        ],
        # Software y licencias
        'Software i llicències': [
            'software', 'licencia', 'llicència', 'subscription', 'cloud', 'saas',
            'aplicación', 'plataforma digital'
        ],
        # Servicios universitarios externos
        'Servicis universitaris externs': [
            'universität', 'university', 'università', 'fundació universitat',
            'institut joan lluís vives', 'crue', 'consorci'
        ],
    }

    # Clasificación adicional por nombre de proveedor
    proveedores_categoria = {
        'Publicacions científiques': [
            'springer', 'elsevier', 'wiley', 'mdpi', 'frontiers', 'plos',
            'acs publication', 'optical publishing', 'sage publication',
            'taylor & francis', 'oxford university press', 'cambridge university'
        ],
        'Drets reprogràfics (CEDRO)': [
            'centro español de derechos reprograficos', 'cedro'
        ],
        'Viatges i transport': [
            'viajes el corte', 'halcon viajes', 'viajes mago', 'mediterráneo holiday',
            'rosselli', 'vueling', 'iberia', 'ryanair', 'renfe', 'civis hoteles'
        ],
        'Missatgeria i enviaments': [
            'dhl express', 'ups', 'fedex', 'seur', 'correos express',
            'azahar urgent', 'dachser'
        ],
        'Reprografia i fotocopiadores': [
            'copistería format', 'copisteria', 'reprografia'
        ],
        'Premsa i comunicació': [
            'eco3 multimedia', 'radio popular', 'cope', 'onda cero',
            'el mediterráneo', 'uniprex'
        ],
        'Servicis universitaris externs': [
            'institut joan lluís vives', 'fundació universitat jaume',
            'universität', 'university', 'università', 'consorci'
        ],
        'Servicis legals i assessoria': [
            'abogacía general', 'procura', 'notaría', 'notario'
        ],
        'Membresies i quotes': [
            'aneca', 'crue', 'ruvid', 'aecr', 'european association'
        ],
        'Restauració i càtering': [
            'panificadora', 'catering', 'restaurante', 'tanatorio'
        ],
    }

    def clasificar(row):
        desc = row.get('Desc Gasto', '')
        nombre = row.get('Nombre Complet', '')

        if pd.isna(desc):
            desc = ''
        if pd.isna(nombre):
            nombre = ''

        desc_lower = str(desc).lower()
        nombre_lower = str(nombre).lower()
        texto_completo = desc_lower + ' ' + nombre_lower

        # Primero buscar por descripción
        for categoria, keywords in categorias.items():
            if any(kw in desc_lower for kw in keywords):
                return categoria

        # Luego buscar por nombre de proveedor
        for categoria, keywords in proveedores_categoria.items():
            if any(kw in nombre_lower for kw in keywords):
                return categoria

        return 'Altres'

    df['Categoria'] = df.apply(clasificar, axis=1)
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
# SESSION STATE PARA GESTIÓN DE CATEGORÍAS
# ============================================================
# Categorías base del sistema
CATEGORIAS_BASE = [
    'Publicacions científiques',
    'Drets reprogràfics (CEDRO)',
    'Inscripcions i congressos',
    'Formació i cursos',
    'Col·laboradors docents',
    'Viatges i transport',
    'Membresies i quotes',
    'Programa Pisos Solidaris',
    'Subministraments (aigua, llum)',
    'Reprografia i fotocopiadores',
    'Missatgeria i enviaments',
    'Restauració i càtering',
    'Servicis legals i assessoria',
    'Manteniment i infraestructura',
    'Premsa i comunicació',
    'Bibliografia i llibres',
    'Material i equipament',
    'Software i llicències',
    'Servicis universitaris externs',
    'Altres'
]

# Inicializar session_state
if 'categorias_personalizadas' not in st.session_state:
    st.session_state.categorias_personalizadas = []  # Nuevas categorías creadas por el usuario

if 'asignaciones_manuales' not in st.session_state:
    st.session_state.asignaciones_manuales = {}  # {(proveedor, factura): nueva_categoria}

if 'reglas_proveedor' not in st.session_state:
    st.session_state.reglas_proveedor = {}  # {proveedor: categoria} - reglas por proveedor


def obtener_todas_categorias() -> List[str]:
    """Devuelve todas las categorías disponibles (base + personalizadas)."""
    todas = CATEGORIAS_BASE.copy()
    for cat in st.session_state.categorias_personalizadas:
        if cat not in todas:
            todas.append(cat)
    return sorted(todas)


def aplicar_asignaciones_manuales(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las asignaciones manuales sobre el DataFrame."""
    df = df.copy()

    # Primero aplicar reglas por proveedor
    for proveedor, categoria in st.session_state.reglas_proveedor.items():
        mask = df['Nombre Complet'] == proveedor
        df.loc[mask, 'Categoria'] = categoria

    # Luego aplicar asignaciones individuales (tienen prioridad)
    for (proveedor, factura), categoria in st.session_state.asignaciones_manuales.items():
        mask = (df['Nombre Complet'] == proveedor) & (df['N Factura'] == factura)
        df.loc[mask, 'Categoria'] = categoria

    return df


# ============================================================
# NORMALIZACIÓN DE COLUMNAS
# ============================================================
import unicodedata
import re


def normalizar_texto_columna(texto: str) -> str:
    """Normaliza texto eliminando acentos, espacios extra y caracteres especiales.

    Args:
        texto: Nombre de columna original

    Returns:
        Texto normalizado para comparación
    """
    if not isinstance(texto, str):
        return str(texto).lower().strip()
    # Convertir a minúsculas y eliminar espacios extra
    texto = texto.lower().strip()
    # Normalizar Unicode (NFD) y eliminar marcas diacríticas (acentos)
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    # Reemplazar caracteres especiales por espacio y eliminar espacios múltiples
    texto = re.sub(r'[_\-\.]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


# Mapeo de nombres alternativos a nombres esperados (en minúsculas y sin acentos)
COLUMNAS_MAPEO = {
    'Nombre Complet': [
        'nombre complet', 'nom complet', 'nombre completo', 'nom complet proveidor',
        'proveedor', 'proveidor', 'nombre', 'nom', 'nombre proveedor', 'nom proveidor',
        'supplier', 'vendor', 'razon social', 'rao social', 'tercero', 'tercer',
        'acreedor', 'creditor', 'empresa', 'entitat', 'entidad', 'denominacion',
        'denominacio', 'nom empresa', 'nombre empresa', 'titular', 'beneficiario',
        'beneficiari', 'nom del proveidor', 'nombre del proveedor'
    ],
    'N Factura': [
        'n factura', 'factura', 'num factura', 'no factura', 'numero factura',
        'num factura', 'invoice', 'num factura', 'n factura', 'nfactura',
        'ref factura', 'referencia', 'documento', 'doc', 'num doc', 'n doc',
        'num documento', 'numero documento', 'factura num', 'invoice number',
        'factura no', 'numero', 'ref'
    ],
    'Desc Gasto': [
        'desc gasto', 'descripcion', 'descripcio', 'concepto', 'detalle',
        'description', 'desc gasto', 'descgasto', 'descripcion gasto',
        'text', 'texto', 'observaciones', 'observacions', 'motivo', 'motiu',
        'concepto gasto', 'objeto', 'objecte', 'desc', 'descripcion factura',
        'descripcio factura', 'detall', 'comentario', 'comentari'
    ],
    'Base imp': [
        'base imp', 'importe', 'import', 'base imponible', 'base',
        'amount', 'total', 'base imp', 'baseimp', 'importe base',
        'base imposable', 'importe neto', 'import net', 'subtotal',
        'importe sin iva', 'import sense iva', 'cantidad', 'quantitat',
        'euros', 'eur', 'valor', 'monto', 'coste', 'cost', 'precio', 'preu'
    ]
}

COLUMNAS_REQUERIDAS = ['Nombre Complet', 'N Factura', 'Desc Gasto', 'Base imp']


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de columnas al formato esperado.

    Usa normalización de texto (elimina acentos, espacios extra, etc.)
    para hacer la comparación más robusta.

    Args:
        df: DataFrame con columnas originales

    Returns:
        DataFrame con columnas renombradas al formato esperado
    """
    df = df.copy()
    # Crear diccionario de columnas normalizadas -> nombre original
    columnas_normalizadas = {
        normalizar_texto_columna(col): col for col in df.columns
    }

    renombrar = {}
    for col_esperada, alternativas in COLUMNAS_MAPEO.items():
        # Si ya existe la columna esperada exacta, no hacer nada
        if col_esperada in df.columns:
            continue

        # Buscar coincidencia normalizada de la columna esperada
        col_esperada_norm = normalizar_texto_columna(col_esperada)
        if col_esperada_norm in columnas_normalizadas:
            col_original = columnas_normalizadas[col_esperada_norm]
            if col_original != col_esperada:
                renombrar[col_original] = col_esperada
            continue

        # Buscar en alternativas
        encontrada = False
        for alt in alternativas:
            alt_norm = normalizar_texto_columna(alt)
            if alt_norm in columnas_normalizadas:
                col_original = columnas_normalizadas[alt_norm]
                renombrar[col_original] = col_esperada
                encontrada = True
                break

        # Si no se encontró, buscar coincidencia parcial (la columna contiene el patrón)
        if not encontrada:
            for alt in alternativas[:5]:  # Solo las primeras alternativas más comunes
                alt_norm = normalizar_texto_columna(alt)
                for col_norm, col_orig in columnas_normalizadas.items():
                    if alt_norm in col_norm or col_norm in alt_norm:
                        renombrar[col_orig] = col_esperada
                        encontrada = True
                        break
                if encontrada:
                    break

    if renombrar:
        df = df.rename(columns=renombrar)

    return df


def validar_columnas(df: pd.DataFrame, nombre_hoja: str) -> tuple:
    """Valida que el DataFrame tenga las columnas requeridas.

    Args:
        df: DataFrame a validar
        nombre_hoja: Nombre de la hoja para mensajes de error

    Returns:
        Tuple (es_valido, mensaje_error)
    """
    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]

    if columnas_faltantes:
        # Mostrar TODAS las columnas disponibles para ayudar al usuario
        columnas_lista = df.columns.tolist()
        columnas_formateadas = '\n'.join([f"  - `{col}`" for col in columnas_lista])

        # Sugerencias de mapeo
        sugerencias = []
        for col_faltante in columnas_faltantes:
            alternativas = COLUMNAS_MAPEO.get(col_faltante, [])
            if alternativas:
                alts_texto = ', '.join(alternativas[:5])
                sugerencias.append(f"- **{col_faltante}**: busquem `{alts_texto}`...")

        sugerencias_texto = '\n'.join(sugerencias) if sugerencias else ""

        msg = f"""**⚠️ Error en la fulla '{nombre_hoja}'**

---

### ❌ Columnes requerides que no s'han trobat:
{', '.join([f'`{c}`' for c in columnas_faltantes])}

---

### 📋 Columnes disponibles en el fitxer ({len(columnas_lista)}):
{columnas_formateadas}

---

### 🔍 Noms de columna que busquem:
{sugerencias_texto}

---

### 💡 Solució:
Renombra les columnes del fitxer Excel perquè coincidisquen amb:
- `Nombre Complet` → nom del proveïdor
- `N Factura` → número de factura
- `Desc Gasto` → descripció del gasto
- `Base imp` → import base imponible
"""
        return False, msg

    return True, ""


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
    # Normalizar nombres de columnas
    df_2024 = normalizar_columnas(df_2024)
    df_2025 = normalizar_columnas(df_2025)

    # Validar columnas requeridas
    valido_2024, error_2024 = validar_columnas(df_2024, 'SIN EXPTE 2024')
    valido_2025, error_2025 = validar_columnas(df_2025, 'SIN EXPTE 2025')

    if not valido_2024:
        raise ValueError(error_2024)
    if not valido_2025:
        raise ValueError(error_2025)

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

    # Aplicar asignaciones manuales (fuera del caché para reflejar cambios)
    if datos_cargados:
        df_2024 = aplicar_asignaciones_manuales(df_2024)
        df_2025 = aplicar_asignaciones_manuales(df_2025)

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
            "📋 Detall de Registres",
            "⚙️ Gestió de Categories"
        ]
        seccion = st.radio("Selecciona secció:", secciones, label_visibility="collapsed")
    else:
        seccion = None

    st.divider()

    # Versión (historial completo en CHANGELOG.md)
    st.caption(f"📌 Versió {__version__}")
    st.caption(f"📅 {__fecha_version__}")


# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================
if not datos_cargados:
    st.header("📊 Contractació Sense Expedient")
    st.subheader("Comparativa 2024 vs 2025")

    if error_msg:
        st.error("❌ Error carregant les dades")
        # Si el error tiene formato markdown (de validación de columnas), mostrarlo con st.markdown
        if '**' in error_msg:
            st.markdown(error_msg)
        else:
            st.code(error_msg)
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=40, t=80, b=40)  # Margen superior para etiquetas
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
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)  # Márgenes para etiquetas exteriores
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
    df_cat['Diferencia_abs'] = df_cat['Diferencia'].abs()
    df_cat['Import_Total'] = df_cat['Import_2024'] + df_cat['Import_2025']

    # Ordenar por diferencia absoluta y tomar top 10
    df_cat_full = df_cat.copy()  # Guardar todas las categorías para la tabla
    df_cat_top10 = df_cat.nlargest(10, 'Diferencia_abs').sort_values('Diferencia_abs', ascending=True)

    st.subheader("🔝 Top 10 Categories amb Major Variació")
    st.caption("Categories ordenades per variació absoluta d'import entre 2024 i 2025")

    # Gráfico de barras horizontales con variación
    # Colores según si la variación es positiva o negativa
    colors = ['#e74c3c' if x > 0 else '#27ae60' for x in df_cat_top10['Diferencia']]

    fig_var = go.Figure()

    # Barras de variación (diferencia)
    fig_var.add_trace(go.Bar(
        y=df_cat_top10['Categoria'],
        x=df_cat_top10['Diferencia'],
        orientation='h',
        marker_color=colors,
        text=[f"{formatear_euro(x)}" for x in df_cat_top10['Diferencia']],
        textposition='outside',
        textfont=dict(size=11),
        name='Variació',
        hovertemplate='<b>%{y}</b><br>' +
                      'Variació: %{x:,.2f} €<br>' +
                      '<extra></extra>'
    ))

    fig_var.update_layout(
        template='plotly_white',
        height=450,
        xaxis_title="Variació Import (€)",
        xaxis=dict(
            tickformat=',.0f',
            zeroline=True,
            zerolinecolor='#7f8c8d',
            zerolinewidth=2
        ),
        showlegend=False,
        margin=dict(l=20, r=150, t=60, b=40)  # r=150 para etiquetas exteriores
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # Leyenda explicativa
    col_leg1, col_leg2 = st.columns(2)
    with col_leg1:
        st.markdown("🔴 **Roig**: Increment de gasto en 2025")
    with col_leg2:
        st.markdown("🟢 **Verd**: Reducció de gasto en 2025")

    st.divider()

    # Gráfico comparativo 2024 vs 2025 para el top 10
    st.subheader("📊 Comparativa Import 2024 vs 2025 (Top 10)")

    df_cat_top10_sorted = df_cat_top10.sort_values('Import_Total', ascending=True)

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        name='2024',
        y=df_cat_top10_sorted['Categoria'],
        x=df_cat_top10_sorted['Import_2024'],
        orientation='h',
        marker_color='#3498db',
        text=[formatear_euro(x) for x in df_cat_top10_sorted['Import_2024']],
        textposition='inside',
        textfont=dict(size=10, color='white')
    ))
    fig_cat.add_trace(go.Bar(
        name='2025',
        y=df_cat_top10_sorted['Categoria'],
        x=df_cat_top10_sorted['Import_2025'],
        orientation='h',
        marker_color='#e74c3c',
        text=[formatear_euro(x) for x in df_cat_top10_sorted['Import_2025']],
        textposition='inside',
        textfont=dict(size=10, color='white')
    ))
    fig_cat.update_layout(
        barmode='group',
        template='plotly_white',
        height=450,
        xaxis_title="Import (€)",
        xaxis=dict(tickformat=',.0f'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # Tabla de detalle completa (todas las categorías)
    with st.expander("📋 Detall per Categoria (Totes)", expanded=False):
        df_cat_display = df_cat_full.sort_values('Diferencia_abs', ascending=False).copy()
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

        # Eliminar columnas auxiliares
        df_cat_display = df_cat_display.drop(columns=['Diferencia_abs', 'Import_Total'])

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

        st.warning(f"""
        ⚠️ **Alerta**: Hi ha **{len(nuevos_agrupados)} nous proveïdors de viatges** en 2025
        que no existien en 2024, amb un import total de **{formatear_euro(nuevos_viajes['Base imp'].sum())}**
        """)

        with st.expander(f"📋 Llistat de {len(nuevos_agrupados)} nous proveïdors", expanded=False):
            # Formatear
            nuevos_display = nuevos_agrupados.copy()
            nuevos_display['Import Total'] = nuevos_display['Import Total'].apply(formatear_euro)
            st.dataframe(nuevos_display, use_container_width=True, hide_index=True)
    else:
        st.success("No hi ha nous proveïdors de viatges en 2025")

    # Detalle de facturas de viajes
    with st.expander("📋 Detall de totes les factures de viatges 2025", expanded=False):
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
        with st.expander("📋 Comparativa per editorial", expanded=False):
            df_ed_display = df_ed.copy()
            df_ed_display['Import 2024'] = df_ed_display['Import 2024'].apply(formatear_euro)
            df_ed_display['Import 2025'] = df_ed_display['Import 2025'].apply(formatear_euro)
            df_ed_display['Diferència'] = df_ed_display['Diferència'].apply(formatear_euro)
            st.dataframe(df_ed_display, use_container_width=True, hide_index=True)

    # Detalle de publicaciones
    with st.expander("📋 Detall de factures de publicacions 2025", expanded=False):
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
    with st.expander("📋 Taula completa d'increments", expanded=False):
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

    st.info(f"""
    📊 **Resum nous proveïdors**: {len(nuevos)} nous proveïdors en 2025 amb un import total
    de **{formatear_euro(df_nuevos['Base imp'].sum())}** ({(df_nuevos['Base imp'].sum()/total_2025)*100:.1f}% del total)
    """)

    with st.expander("📋 Top 15 nous proveïdors", expanded=False):
        top_nuevos_display = top_nuevos.copy()
        top_nuevos_display['Import Total'] = top_nuevos_display['Import Total'].apply(formatear_euro)
        st.dataframe(top_nuevos_display, use_container_width=True, hide_index=True)


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
# SECCIÓN: GESTIÓN DE CATEGORÍAS
# ============================================================
elif seccion == "⚙️ Gestió de Categories":
    st.header("⚙️ Gestió de Categories")

    st.markdown("""
    Aquesta secció permet personalitzar la classificació de gastos:
    - **Reassignar categories** a registres individuals o proveïdors
    - **Crear noves categories** personalitzades
    - **Veure i gestionar** les regles actives
    """)

    # Combinar datos para el formulario
    df_combinado = pd.concat([df_2024, df_2025], ignore_index=True)

    # Tabs para organizar la sección
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Reassignar Registres",
        "🏢 Regles per Proveïdor",
        "➕ Nova Categoria",
        "📊 Regles Actives"
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: Reasignar registros individuales
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("📋 Reassignar Categoria a Registres")

        st.info("Selecciona un any i filtra per trobar el registre que vols reassignar.")

        col1, col2 = st.columns(2)

        with col1:
            any_asignar = st.radio("Any:", [2024, 2025], horizontal=True, key="any_asignar")
            df_trabajo = df_2024 if any_asignar == 2024 else df_2025

        with col2:
            # Filtro por categoría actual
            cat_actual_filter = st.selectbox(
                "Filtrar per categoria actual:",
                ['Totes'] + sorted(df_trabajo['Categoria'].unique().tolist()),
                key="cat_filter_asignar"
            )

        # Aplicar filtro
        df_mostrar = df_trabajo.copy()
        if cat_actual_filter != 'Totes':
            df_mostrar = df_mostrar[df_mostrar['Categoria'] == cat_actual_filter]

        # Buscar por texto
        busqueda = st.text_input("🔍 Buscar per proveïdor o descripció:", key="busqueda_registre")
        if busqueda:
            mask = (
                df_mostrar['Nombre Complet'].str.lower().str.contains(busqueda.lower(), na=False) |
                df_mostrar['Desc Gasto'].str.lower().str.contains(busqueda.lower(), na=False)
            )
            df_mostrar = df_mostrar[mask]

        st.caption(f"Mostrant {len(df_mostrar)} registres")

        # Mostrar registros con checkbox para selección
        if len(df_mostrar) > 0:
            # Crear identificador único
            df_mostrar = df_mostrar.copy()
            df_mostrar['_id'] = df_mostrar['Nombre Complet'] + ' | ' + df_mostrar['N Factura'].astype(str)

            # Selector de registro
            registro_seleccionado = st.selectbox(
                "Selecciona el registre a reassignar:",
                df_mostrar['_id'].tolist(),
                key="registro_seleccionado"
            )

            if registro_seleccionado:
                # Mostrar detalle del registro
                registro = df_mostrar[df_mostrar['_id'] == registro_seleccionado].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Proveïdor:**")
                    st.write(registro['Nombre Complet'])
                    st.markdown("**Factura:**")
                    st.write(registro['N Factura'])
                with col2:
                    st.markdown("**Descripció:**")
                    st.write(registro['Desc Gasto'])
                    st.markdown("**Import:**")
                    st.write(formatear_euro(registro['Base imp']))

                st.markdown(f"**Categoria actual:** `{registro['Categoria']}`")

                # Selector de nueva categoría
                todas_cats = obtener_todas_categorias()
                nueva_cat = st.selectbox(
                    "Nova categoria:",
                    todas_cats,
                    key="nueva_cat_registro"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Aplicar canvi", type="primary", key="btn_aplicar_registro"):
                        clave = (registro['Nombre Complet'], registro['N Factura'])
                        st.session_state.asignaciones_manuales[clave] = nueva_cat
                        st.success(f"✅ Registre reassignat a '{nueva_cat}'")
                        st.rerun()

                with col2:
                    # Verificar si tiene asignación manual
                    clave = (registro['Nombre Complet'], registro['N Factura'])
                    if clave in st.session_state.asignaciones_manuales:
                        if st.button("🗑️ Eliminar assignació manual", key="btn_eliminar_registro"):
                            del st.session_state.asignaciones_manuales[clave]
                            st.success("Assignació manual eliminada")
                            st.rerun()
        else:
            st.warning("No s'han trobat registres amb els filtres seleccionats.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Reglas por proveedor
    # ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🏢 Assignar Categoria a Tots els Registres d'un Proveïdor")

        st.info("""
        Crea una regla per assignar automàticament una categoria a **tots els registres**
        d'un proveïdor (passat i futur).
        """)

        # Lista de proveedores únicos
        proveedores_unicos = sorted(df_combinado['Nombre Complet'].unique().tolist())

        proveedor_seleccionado = st.selectbox(
            "Selecciona proveïdor:",
            proveedores_unicos,
            key="proveedor_regla"
        )

        if proveedor_seleccionado:
            # Mostrar info del proveedor
            registros_prov = df_combinado[df_combinado['Nombre Complet'] == proveedor_seleccionado]
            cats_actuales = registros_prov['Categoria'].unique()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Registres totals", len(registros_prov))
            with col2:
                st.metric("Import total", formatear_euro(registros_prov['Base imp'].sum()))
            with col3:
                st.metric("Categories actuals", len(cats_actuales))

            st.markdown(f"**Categories actuals:** {', '.join(cats_actuales)}")

            # Verificar si ya tiene regla
            if proveedor_seleccionado in st.session_state.reglas_proveedor:
                st.warning(f"⚠️ Aquest proveïdor ja té una regla activa: `{st.session_state.reglas_proveedor[proveedor_seleccionado]}`")

            # Selector de categoría
            todas_cats = obtener_todas_categorias()
            nueva_cat_prov = st.selectbox(
                "Assignar categoria:",
                todas_cats,
                key="nueva_cat_proveedor"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Crear regla", type="primary", key="btn_crear_regla"):
                    st.session_state.reglas_proveedor[proveedor_seleccionado] = nueva_cat_prov
                    st.success(f"✅ Regla creada: '{proveedor_seleccionado}' → '{nueva_cat_prov}'")
                    st.rerun()

            with col2:
                if proveedor_seleccionado in st.session_state.reglas_proveedor:
                    if st.button("🗑️ Eliminar regla", key="btn_eliminar_regla"):
                        del st.session_state.reglas_proveedor[proveedor_seleccionado]
                        st.success("Regla eliminada")
                        st.rerun()

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Crear nueva categoría
    # ─────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("➕ Crear Nova Categoria")

        st.info("Crea una nova categoria personalitzada per classificar gastos.")

        # Mostrar categorías existentes
        with st.expander("📋 Categories existents"):
            todas_cats = obtener_todas_categorias()
            col1, col2 = st.columns(2)
            for i, cat in enumerate(todas_cats):
                with col1 if i % 2 == 0 else col2:
                    if cat in st.session_state.categorias_personalizadas:
                        st.markdown(f"🆕 {cat}")
                    else:
                        st.markdown(f"📁 {cat}")

        # Formulario para nueva categoría
        nueva_categoria = st.text_input(
            "Nom de la nova categoria:",
            placeholder="Ex: Servicis d'assessorament extern",
            key="input_nueva_cat"
        )

        if nueva_categoria:
            if nueva_categoria in obtener_todas_categorias():
                st.error("❌ Aquesta categoria ja existeix")
            else:
                if st.button("✅ Crear categoria", type="primary", key="btn_crear_cat"):
                    st.session_state.categorias_personalizadas.append(nueva_categoria)
                    st.success(f"✅ Categoria '{nueva_categoria}' creada correctament")
                    st.balloons()
                    st.rerun()

        # Eliminar categorías personalizadas
        if st.session_state.categorias_personalizadas:
            st.divider()
            st.markdown("**🗑️ Eliminar categories personalitzades:**")

            cat_eliminar = st.selectbox(
                "Selecciona categoria a eliminar:",
                st.session_state.categorias_personalizadas,
                key="cat_eliminar"
            )

            if st.button("🗑️ Eliminar", key="btn_eliminar_cat"):
                # Verificar que no esté en uso
                en_uso = False
                for cat in st.session_state.reglas_proveedor.values():
                    if cat == cat_eliminar:
                        en_uso = True
                        break
                for cat in st.session_state.asignaciones_manuales.values():
                    if cat == cat_eliminar:
                        en_uso = True
                        break

                if en_uso:
                    st.error("❌ No es pot eliminar: la categoria està en ús")
                else:
                    st.session_state.categorias_personalizadas.remove(cat_eliminar)
                    st.success(f"✅ Categoria '{cat_eliminar}' eliminada")
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    # TAB 4: Ver reglas activas
    # ─────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("📊 Regles Actives")

        # Contador de reglas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏢 Regles per proveïdor", len(st.session_state.reglas_proveedor))
        with col2:
            st.metric("📋 Assignacions individuals", len(st.session_state.asignaciones_manuales))
        with col3:
            st.metric("🆕 Categories personalitzades", len(st.session_state.categorias_personalizadas))

        st.divider()

        # Reglas por proveedor
        if st.session_state.reglas_proveedor:
            st.markdown("### 🏢 Regles per Proveïdor")
            reglas_df = pd.DataFrame([
                {'Proveïdor': k, 'Categoria Assignada': v}
                for k, v in st.session_state.reglas_proveedor.items()
            ])
            st.dataframe(reglas_df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Eliminar totes les regles per proveïdor", key="btn_limpiar_reglas"):
                st.session_state.reglas_proveedor = {}
                st.success("Totes les regles eliminades")
                st.rerun()
        else:
            st.info("No hi ha regles per proveïdor actives")

        st.divider()

        # Asignaciones individuales
        if st.session_state.asignaciones_manuales:
            st.markdown("### 📋 Assignacions Individuals")
            asig_df = pd.DataFrame([
                {'Proveïdor': k[0], 'Factura': k[1], 'Categoria Assignada': v}
                for k, v in st.session_state.asignaciones_manuales.items()
            ])
            st.dataframe(asig_df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Eliminar totes les assignacions individuals", key="btn_limpiar_asig"):
                st.session_state.asignaciones_manuales = {}
                st.success("Totes les assignacions eliminades")
                st.rerun()
        else:
            st.info("No hi ha assignacions individuals actives")

        st.divider()

        # Categorías personalizadas
        if st.session_state.categorias_personalizadas:
            st.markdown("### 🆕 Categories Personalitzades")
            for cat in st.session_state.categorias_personalizadas:
                st.markdown(f"- {cat}")
        else:
            st.info("No hi ha categories personalitzades")

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # EXPORTAR / IMPORTAR REGLAS
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 💾 Exportar / Importar Regles")

        col_exp, col_imp = st.columns(2)

        with col_exp:
            st.markdown("**📤 Exportar regles a JSON**")

            # Preparar datos para exportar
            # Convertir tuplas a listas para JSON
            asignaciones_export = {
                f"{k[0]}|||{k[1]}": v
                for k, v in st.session_state.asignaciones_manuales.items()
            }

            export_data = {
                "version": __version__,
                "fecha_exportacion": datetime.now().isoformat(),
                "categorias_personalizadas": st.session_state.categorias_personalizadas,
                "reglas_proveedor": st.session_state.reglas_proveedor,
                "asignaciones_manuales": asignaciones_export
            }

            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📥 Descarregar JSON",
                data=json_str,
                file_name=f"regles_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="btn_exportar_json"
            )

            st.caption(f"📊 {len(st.session_state.reglas_proveedor)} regles, "
                      f"{len(st.session_state.asignaciones_manuales)} assignacions, "
                      f"{len(st.session_state.categorias_personalizadas)} categories")

        with col_imp:
            st.markdown("**📥 Importar regles des de JSON**")

            archivo_json = st.file_uploader(
                "Selecciona fitxer JSON",
                type=['json'],
                key="upload_json_regles"
            )

            if archivo_json is not None:
                try:
                    import_data = json.load(archivo_json)

                    st.info(f"📋 Fitxer versió: {import_data.get('version', 'desconeguda')}")

                    # Mostrar resumen
                    n_cats = len(import_data.get('categorias_personalizadas', []))
                    n_reglas = len(import_data.get('reglas_proveedor', {}))
                    n_asig = len(import_data.get('asignaciones_manuales', {}))

                    st.caption(f"Conté: {n_reglas} regles, {n_asig} assignacions, {n_cats} categories")

                    modo_import = st.radio(
                        "Mode d'importació:",
                        ["Afegir a existents", "Reemplaçar tot"],
                        key="modo_importacion",
                        horizontal=True
                    )

                    if st.button("✅ Importar", type="primary", key="btn_importar_json"):
                        if modo_import == "Reemplaçar tot":
                            st.session_state.categorias_personalizadas = []
                            st.session_state.reglas_proveedor = {}
                            st.session_state.asignaciones_manuales = {}

                        # Importar categorías personalizadas
                        for cat in import_data.get('categorias_personalizadas', []):
                            if cat not in st.session_state.categorias_personalizadas:
                                st.session_state.categorias_personalizadas.append(cat)

                        # Importar reglas por proveedor
                        for prov, cat in import_data.get('reglas_proveedor', {}).items():
                            st.session_state.reglas_proveedor[prov] = cat

                        # Importar asignaciones manuales
                        for key_str, cat in import_data.get('asignaciones_manuales', {}).items():
                            parts = key_str.split('|||')
                            if len(parts) == 2:
                                clave = (parts[0], parts[1])
                                st.session_state.asignaciones_manuales[clave] = cat

                        st.success("✅ Regles importades correctament!")
                        st.rerun()

                except json.JSONDecodeError:
                    st.error("❌ Error: El fitxer no és un JSON vàlid")
                except Exception as e:
                    st.error(f"❌ Error important: {str(e)}")

        st.divider()

        # Botón para limpiar todo
        st.warning("⚠️ **Zona de perill**")
        if st.button("🗑️ ELIMINAR TOTES LES PERSONALITZACIONS", type="secondary", key="btn_reset_todo"):
            st.session_state.reglas_proveedor = {}
            st.session_state.asignaciones_manuales = {}
            st.session_state.categorias_personalizadas = []
            st.success("✅ Totes les personalitzacions han sigut eliminades")
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("📊 Dashboard de Contractació Sense Expedient | Universitat Jaume I | Vicegerència de Recursos Humans")
