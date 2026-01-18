# 📊 Contractació Sense Expedient - Dashboard

Dashboard interactiu per a l'anàlisi comparatiu de la contractació sense expedient entre els exercicis 2024 i 2025.

**Universitat Jaume I** - Vicegerència de Recursos Humans

## 🎯 Funcionalitats

- 📊 **Resum Executiu**: Mètriques principals i indicadors clau
- 📈 **Comparativa per Categories**: Classificació automàtica de gastos
- ✈️ **Anàlisi de Viatges**: Detall de proveïdors de transport i agències
- 📚 **Anàlisi de Publicacions**: Editorials científiques i costos Open Access
- 🏢 **Top Proveïdors**: Rànquings, increments i nous proveïdors
- 📋 **Detall de Registres**: Taula filtrable amb exportació a Excel

## 📁 Estructura de Dades Requerida

Per a executar el dashboard, necessites crear la carpeta `datos/` amb el fitxer:

```
datos/
└── SIN EXPEDIENTE.xlsx
    ├── Fulla: "SIN EXPTE 2024"
    └── Fulla: "SIN EXPTE 2025"
```

### Columnes esperades

| Columna | Descripció |
|---------|------------|
| `Nombre Complet` | Nom del proveïdor |
| `N Factura` | Número de factura |
| `Desc Gasto` | Descripció del gasto |
| `Base imp` | Base imposable (€) |

## 🚀 Execució Local

```bash
# Crear entorn virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instal·lar dependències
pip install -r requirements.txt

# Executar
streamlit run app_sin_expediente.py
```

## ☁️ Desplegament a Streamlit Cloud

1. Fork o clona aquest repositori
2. Connecta amb [Streamlit Cloud](https://streamlit.io/cloud)
3. Puja manualment `datos/SIN EXPEDIENTE.xlsx` a través de la interfície
4. Configura l'app apuntant a `app_sin_expediente.py`

## 📊 Resultats Clau (Exemple)

| Mètrica | 2024 | 2025 | Variació |
|---------|------|------|----------|
| Registres | 1.361 | 2.385 | +75% |
| Import total | 662.574 € | 1.213.342 € | +83% |
| Proveïdors únics | 525 | 895 | +70% |

## 🛠️ Tecnologies

- **Frontend**: Streamlit
- **Visualització**: Plotly
- **Dades**: Pandas, NumPy
- **Caché**: PyArrow (Parquet)

## 📄 Llicència

Projecte intern - Universitat Jaume I

---

*Desenvolupat per la Vicegerència de Recursos Humans - UJI*
