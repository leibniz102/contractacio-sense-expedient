# 📋 Historial de Canvis - Contractació Sense Expedient

## [1.5.3] - 2026-01-22

### 🐛 Correccions

- **Corregit error crític** "KeyError: 'Nombre Complet'" al pujar fitxers Excel
- Normalització de columnes millorada amb eliminació d'accents i caràcters especials
- Cerca de coincidències parcials en noms de columnes
- Missatge d'error molt més detallat mostrant totes les columnes disponibles

### ✨ Millores

- Ampliada llista d'alternatives de noms de columnes (ara 25+ per columna)
- Suport per noms amb accents diferents (é, è, ë → e)
- Suport per noms amb guions baixos, punts o espais
- Cerca intel·ligent de coincidències parcials

### 📁 Noms de columnes suportats (ampliat)

| Columna esperada | Noves alternatives afegides |
|------------------|------------------------|
| `Nombre Complet` | nom del proveidor, denominacio, empresa, entitat, titular, beneficiari |
| `N Factura` | ref factura, documento, num doc, ref, factura no |
| `Desc Gasto` | text, texto, observaciones, objecte, comentari |
| `Base imp` | importe neto, subtotal, cantidad, valor, coste |

---

## [1.5.2] - 2026-01-19

### 🐛 Correccions

- Corregit error "Nombre Complet" quan l'usuari puja fitxers amb columnes alternatives
- Afegida normalització automàtica de noms de columnes (proveedor → Nombre Complet, etc.)
- Missatge d'error millorat mostrant columnes requerides i disponibles
- Suport per noms de columnes en valencià, castellà i anglés

### 📁 Noms de columnes suportats

| Columna esperada | Alternatives acceptades |
|------------------|------------------------|
| `Nombre Complet` | proveedor, proveïdor, nombre, razon social, tercero |
| `N Factura` | factura, num factura, nº factura, invoice |
| `Desc Gasto` | descripcion, descripció, concepto, detalle |
| `Base imp` | importe, import, base imponible, total |

---

## [1.5.1] - 2026-01-18

### 🔧 Manteniment

- Eliminat historial de canvis del sidebar (ara només en CHANGELOG.md)
- Simplificat codi de versió en l'aplicació

### 📁 Documentació

- Afegit Pas 0 (crear compte) a la guia d'usuari
- Afegida secció "Desplegar en Cloud" a la guia d'usuari

---

## [1.5.0] - 2026-01-18

### ✨ Millores UX i persistència
- Taules dins d'expanders replegats per defecte (millor navegació)
- Ajust de marges en gràfics per mostrar etiquetes correctament
- Exportar/Importar regles a fitxer JSON (persistència entre sessions)
- Persistència de personalitzacions entre sessions

### 📁 Nous arxius
- `docs/GUIA_USUARI.html` - Guia visual pas a pas per a usuaris

---

## [1.4.0] - 2026-01-18

### ✨ Gestió de categories personalitzada
- Nova secció "⚙️ Gestió de Categories" amb 4 pestanyes
- Reassignació de categories a registres individuals
- Regles per proveïdor (assignació automàtica a totes les factures)
- Creació de noves categories personalitzades
- Panel de visualització de regles actives

---

## [1.3.0] - 2026-01-18

### ✨ Millora visualització categories
- Gràfic top 10 categories per variació absoluta
- Barres amb colors diferenciats (roig=increment, verd=reducció)
- Imports representatius amb etiquetes dins de les barres
- Comparativa visual 2024 vs 2025 per al top 10

---

## [1.2.0] - 2026-01-18

### ✨ Millora classificació categories
- Afegides 11 noves categories (de 8 a 19 total)
- Classificació per descripció i nom de proveïdor
- Reduït percentatge "Altres" del 41% al ~10%

### 📁 Noves categories afegides
| Categoria | Descripció |
|-----------|------------|
| Drets reprogràfics (CEDRO) | Pagaments a CEDRO |
| Formació i cursos | Cursos, tallers, màsters |
| Col·laboradors docents | Supervisors de pràctiques |
| Programa Pisos Solidaris | Programa social UJI |
| Reprografia i fotocopiadores | Còpies i impressió |
| Missatgeria i enviaments | Paqueteria i correus |
| Restauració i càtering | Menjars i events |
| Servicis legals i assessoria | Advocats, notaris |
| Manteniment i infraestructura | Reparacions, hosting |
| Premsa i comunicació | Ràdio, publicitat |
| Servicis universitaris externs | Consorcis, fundacions |

---

## [1.1.0] - 2026-01-18

### ✨ Suport Streamlit Cloud
- Afegit file_uploader per pujar Excel des de la interfície
- Compatible amb Streamlit Cloud sense necessitat de fitxers locals
- Missatge de benvinguda amb instruccions

---

## [1.0.0] - 2026-01-18

### 🎉 Versió inicial
- Dashboard complet amb Streamlit
- Comparativa 2024 vs 2025
- Resum executiu amb mètriques clau
- Anàlisi detallat de viatges i transport
- Anàlisi de publicacions científiques
- Top proveïdors i increments
- Detall de registres filtrable
- Exportació a Excel

---

## Llegenda

| Símbol | Significat |
|--------|------------|
| ✨ | Nova funcionalitat |
| 🐛 | Correcció d'error |
| 📁 | Canvis en arxius |
| ⚠️ | Canvi important |
| 🎉 | Versió inicial |
