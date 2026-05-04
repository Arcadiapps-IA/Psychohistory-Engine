# Psychohistory Engine

> *"La psicohistoria era la rama de las matemáticas que se ocupaba de las reacciones de los conglomerados humanos ante estímulos sociales y económicos fijos."*
> — Isaac Asimov, *Fundación*

Un framework computacional que combina **física cuántica**, **inteligencia artificial** y **sistemas deterministas** para analizar patrones históricos, predecir trayectorias futuras e identificar los puntos de intervención críticos donde una acción mínima puede alterar el curso de los eventos — inspirado directamente en la psicohistoria de Hari Seldon.

---

## ¿Qué es esto?

El Psychohistory Engine parte de una premisa: aunque el comportamiento individual es impredecible, el comportamiento agregado de grandes poblaciones a lo largo del tiempo sigue patrones estadísticos modelables. Con suficientes datos históricos y los algoritmos correctos, es posible calcular las trayectorias más probables del futuro y encontrar los momentos — las **Crisis Seldon** — donde intervenir tiene el mayor impacto.

El sistema ingiere eventos históricos desde múltiples fuentes (incluyendo Wikipedia y Archive.org), detecta correlaciones y ciclos usando algoritmos clásicos y cuánticos, genera predicciones probabilísticas y señala los puntos de bifurcación críticos con su índice de sensibilidad.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    PsychohistoryEngine                       │
│                      (Orquestador)                           │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
  EventIngester  PatternAnalyzer  TrajectoryPredictor  InterventionDetector
       │          │          │          │          │
       │     ┌────┴────┐     │     ┌────┴────┐     │
       │     │Classical│     │     │  QAOA   │     │
       │     │  + VQC  │     │     │ +Grover │     │
       │     └────┬────┘     │     └────┬────┘     │
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                             │
                      QuantumEngine
                    (Facade cuántico)
                    PennyLane / Qiskit
                    + fallback clásico

Fuentes externas:
  WikipediaConnector ──► ExtractionPipeline ──► Corpus (SQLite/PostgreSQL)
  ArchiveOrgConnector ─►
```

### Subsistemas principales

| Módulo | Responsabilidad |
|--------|----------------|
| `engine.py` | Orquestador principal — API pública del sistema |
| `event_ingester.py` | Ingesta y normalización de eventos históricos (JSON, CSV, texto) |
| `pattern_analyzer.py` | Detección de patrones sociales clásicos y cuánticos |
| `trajectory_predictor.py` | Predicción de trayectorias futuras con QAOA + Grover |
| `intervention_detector.py` | Identificación de Crisis Seldon con entrelazamiento cuántico |
| `quantum_engine.py` | Facade cuántico (VQC, QAOA, Grover, entropía de Von Neumann) |
| `uncertainty.py` | Principio de incertidumbre social (análogo a Heisenberg) |
| `persistence.py` | Repositorio del Corpus con SQLAlchemy |
| `serialization.py` | Export/import de estado con MessagePack + SHA-256 |
| `explainability.py` | Trazabilidad y reportes legibles por humanos |
| `extraction_pipeline.py` | Pipeline de extracción desde fuentes externas |
| `connectors/wikipedia.py` | Conector MediaWiki REST API |
| `connectors/archiveorg.py` | Conector Internet Archive API |

---

## Física cuántica en el sistema

El componente cuántico no es decorativo — está integrado en cuatro aspectos fundamentales del análisis:

### 1. Superposición cuántica en predicción de trayectorias
El espacio de futuros posibles se modela como un **estado cuántico en superposición** antes de ser "colapsado" al momento de la consulta. El algoritmo QAOA (Quantum Approximate Optimization Algorithm) explora este espacio de forma eficiente, y cuando el espacio supera 10,000 combinaciones, el algoritmo de Grover proporciona una ventaja cuadrática en la búsqueda.

### 2. Quantum Machine Learning en detección de patrones
Cuando el espacio de características de los datos históricos supera 50 dimensiones, el sistema delega la detección de correlaciones al **Quantum Engine** mediante **Circuitos Cuánticos Variacionales (VQC)** con arquitectura `AngleEmbedding + StronglyEntanglingLayers`. Esto permite detectar correlaciones que los algoritmos clásicos no encontrarían.

### 3. Entrelazamiento cuántico en Crisis Seldon
El **Sensitivity Index** de cada punto de intervención incorpora la **entropía de Von Neumann** del estado cuántico reducido entre pares de nodos de la trayectoria. Dos eventos históricamente distantes pero con alta entropía de entrelazamiento están correlacionados de forma no local — una intervención en uno afecta al otro.

```
SI(nodo) = 0.6 × D(nodo) + 0.4 × E(nodo)

donde:
  D = divergencia promedio con N=10 perturbaciones ±10%
  E = Entanglement_Metric promedio (entropía de Von Neumann)
  
Crisis Seldon: SI > 0.7
```

### 4. Principio de incertidumbre social
El sistema reconoce un límite teórico fundamental análogo al principio de Heisenberg:

```
σ_estado × σ_momentum ≥ ħ_social (= 0.01 por defecto)
```

No es posible conocer simultáneamente con precisión arbitraria el **estado** de una sociedad y su **momentum de cambio**. Para horizontes temporales superiores a 100 años, la incertidumbre se escala proporcionalmente.

---

## Algoritmos implementados

### Algoritmos cuánticos

| Algoritmo | Uso en el sistema | Límite de qubits |
|-----------|------------------|-----------------|
| **VQC** (Variational Quantum Circuit) | Detección de patrones en alta dimensionalidad | ≤ 50 qubits |
| **QAOA** (Quantum Approximate Optimization) | Exploración del espacio de trayectorias | ≤ 30 qubits |
| **Grover Search** | Búsqueda de trayectorias óptimas (ventaja cuadrática O(√N)) | ≤ 20 qubits |
| **Entropía de Von Neumann** | Cálculo de entrelazamiento entre nodos | Numpy puro |

### Algoritmos clásicos

| Algoritmo | Uso |
|-----------|-----|
| **Co-ocurrencia por períodos** | Detección de patrones en ventanas de 50 años |
| **Análisis de ciclos** | Detección de patrones recurrentes (1–500 años) |
| **Grafos de causalidad dirigidos** | Representación de relaciones entre categorías de eventos |
| **Distancia de Wasserstein** | Cálculo de divergencia entre trayectorias perturbadas |
| **Producto logarítmico** | Confidence Score de trayectorias (evita underflow numérico) |
| **SHA-256** | Integridad de archivos de estado y deduplicación del Corpus |
| **Traza parcial** | Cálculo de matrices de densidad reducidas para entrelazamiento |

---

## Tecnologías

### Core
- **Python 3.11+** — lenguaje principal
- **NumPy** — álgebra lineal, traza parcial, cálculos de probabilidad
- **SciPy** — distancia de Wasserstein para el Sensitivity Index
- **NetworkX** — grafos dirigidos de causalidad entre categorías de eventos

### Computación cuántica
- **PennyLane 0.38+** — backend primario para VQC (diferenciación automática nativa)
- **Qiskit 1.x** — backend secundario para QAOA y Grover, acceso a hardware IBM Quantum
- **pennylane-qiskit** — plugin de integración entre ambos
- **qiskit-aer** — simulador local cuando no hay hardware disponible
- *Fallback automático*: si ningún backend cuántico está disponible, el sistema opera con simulación clásica numpy manteniendo la misma interfaz

### Persistencia
- **SQLAlchemy 2.0** — ORM para la capa de datos
- **SQLite** — base de datos en desarrollo/prototipado
- **PostgreSQL** — base de datos en producción
- **MessagePack** — serialización binaria eficiente del estado del sistema

### Fuentes de datos externas
- **MediaWiki REST API** — extracción de artículos de Wikipedia
- **internetarchive** — biblioteca oficial de Internet Archive / Archive.org
- **requests** — cliente HTTP

### Testing
- **pytest** — framework de tests
- **Hypothesis 6.x** — property-based testing (23 propiedades de corrección)

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/psychohistory-engine.git
cd psychohistory-engine

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -e .

# Instalar dependencias de desarrollo
pip install -e ".[dev]"
```

### Dependencias opcionales

```bash
# Para backend cuántico real (PennyLane)
pip install pennylane>=0.38

# Para hardware IBM Quantum (Qiskit)
pip install qiskit>=1.0 qiskit-aer pennylane-qiskit

# Para extracción desde Archive.org
pip install internetarchive

# Para extracción desde Wikipedia
pip install requests
```

> El sistema funciona sin ninguna de estas dependencias opcionales usando fallback clásico con numpy.

---

## Uso rápido

```python
from psychohistory.engine import PsychohistoryEngine, PredictionParams

# Inicializar el motor
engine = PsychohistoryEngine(
    db_url="sqlite:///mi_corpus.db",
    seed=42
)

# Ingestar eventos históricos
events = [
    {
        "date": "1789-07-14",
        "description": "Toma de la Bastilla. Inicio de la Revolución Francesa.",
        "category": "POLITICAL",
        "actors": ["Tercer Estado", "Asamblea Nacional"],
        "magnitude": 0.95
    },
    # ... más eventos
]
report = engine.ingest_events(events, format="json")
print(f"Aceptados: {report.accepted}, Rechazados: {report.rejected}")

# Predecir trayectorias (requiere ≥ 1000 eventos en el Corpus)
result = engine.predict(
    horizon_years=50,
    params=PredictionParams(seed=42)
)

# Ver las 3 trayectorias más probables
for i, traj in enumerate(result.trajectories):
    print(f"\nTrayectoria {i+1}:")
    print(f"  Confianza: {traj.confidence_score:.4f}")
    print(f"  Incertidumbre: σ_estado={traj.uncertainty_bound.sigma_state:.4f}")
    for node in traj.nodes:
        print(f"  [{node.sequence_index}] {node.predicted_event.description}")

# Ver Crisis Seldon identificadas
for i, crisis_list in enumerate(result.intervention_points):
    for crisis in crisis_list:
        print(f"\nCrisis Seldon en trayectoria {i+1}:")
        print(f"  Sensitivity Index: {crisis.sensitivity_index:.4f}")
        print(f"  Fecha estimada: {crisis.temporal_coordinates.year}")
        print(f"  Acción recomendada: {crisis.recommended_action_type}")

# Obtener explicación detallada
for traj in result.trajectories:
    report = engine.get_explanation(traj.id)
    print(report.uncertainty_description)
```

### Extracción desde Wikipedia

```python
from psychohistory.connectors.wikipedia import WikipediaConnector
from psychohistory.connectors.base import SearchQuery

# Configurar conector
connector = WikipediaConnector()
engine.configure_connector("wikipedia", connector)

# Extraer artículos sobre revoluciones
extraction_report = engine.trigger_extraction(
    "wikipedia",
    SearchQuery(text="French Revolution industrial revolution")
)
print(f"Documentos recuperados: {extraction_report.documents_retrieved}")
print(f"Eventos generados: {extraction_report.events_generated}")
```

### Extracción desde Archive.org

```python
from psychohistory.connectors.archiveorg import ArchiveOrgConnector

connector = ArchiveOrgConnector()
engine.configure_connector("archiveorg", connector)

extraction_report = engine.trigger_extraction(
    "archiveorg",
    SearchQuery(collection="americana", text="historical newspapers 1800")
)
```

### Consultar el Corpus

```python
from psychohistory.persistence import CorpusQuery
from psychohistory.enums import EventCategory

# Filtrar eventos políticos entre 1700 y 1900
from datetime import datetime
results = engine.query_corpus(CorpusQuery(
    date_from=datetime(1700, 1, 1),
    date_to=datetime(1900, 12, 31),
    categories=[EventCategory.POLITICAL, EventCategory.MILITARY],
    operator="OR"
))
print(f"Eventos encontrados: {len(results)}")
```

### Exportar e importar estado

```python
# Guardar estado completo del sistema
engine.export_state("estado_2024.msgpack")

# Restaurar en otra instancia
engine2 = PsychohistoryEngine(db_url="sqlite:///:memory:")
engine2.import_state("estado_2024.msgpack")
```

---

## Categorías de eventos

El sistema clasifica los eventos históricos en seis categorías:

| Categoría | Descripción |
|-----------|-------------|
| `POLITICAL` | Revoluciones, elecciones, cambios de gobierno, tratados |
| `ECONOMIC` | Crisis financieras, burbujas, cambios en sistemas económicos |
| `SOCIAL` | Movimientos sociales, cambios culturales, demografía |
| `MILITARY` | Guerras, batallas, conflictos armados |
| `CULTURAL` | Descubrimientos científicos, movimientos artísticos, religión |
| `NATURAL` | Pandemias, desastres naturales, cambios climáticos |

---

## Propiedades de corrección verificadas

El sistema incluye **23 propiedades de corrección** verificadas con property-based testing (Hypothesis), cubriendo:

- Normalización preserva el esquema canónico de eventos
- Rechazo correcto de eventos con campos obligatorios ausentes
- Unicidad de IDs en ingesta
- Confidence Score de patrones siempre en [0.0, 1.0]
- Patrones con confianza < 0.3 son descartados
- Actualización del Corpus preserva patrones no relacionados
- Predicción retorna exactamente 3 trayectorias
- Confidence Score = producto de probabilidades individuales
- Determinismo con la misma semilla
- Entanglement Metric siempre en [0.0, 1.0]
- Invariante del Uncertainty Bound: σ_estado × σ_momentum ≥ ħ_social
- Monotonía del Uncertainty Bound con el horizonte temporal
- Intervention Points ordenados por Sensitivity Index descendente
- Round-trip de serialización produce estado equivalente
- Integridad SHA-256 del archivo de estado
- Completitud de la traza de razonamiento
- Preservación del documento fuente en el pipeline
- Completitud de metadatos de atribución
- Deduplicación temporal en extracciones incrementales

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Solo tests de propiedades
pytest tests/unit/ -v -k "test_" --hypothesis-seed=42

# Tests de integración end-to-end
pytest tests/integration/ -v
```

---

## Estructura del proyecto

```
psychohistory-engine/
├── psychohistory/
│   ├── __init__.py
│   ├── engine.py                  # Orquestador principal
│   ├── enums.py                   # EventCategory
│   ├── models.py                  # Dataclasses (HistoricalEvent, Trajectory, etc.)
│   ├── exceptions.py              # Jerarquía de excepciones
│   ├── event_ingester.py          # Ingesta y normalización
│   ├── pattern_analyzer.py        # Detección de patrones
│   ├── trajectory_predictor.py    # Predicción de trayectorias
│   ├── intervention_detector.py   # Crisis Seldon
│   ├── quantum_engine.py          # Facade cuántico
│   ├── uncertainty.py             # Principio de incertidumbre
│   ├── persistence.py             # CorpusRepository (SQLAlchemy)
│   ├── serialization.py           # Export/import MessagePack
│   ├── explainability.py          # Reportes de trazabilidad
│   ├── extraction_pipeline.py     # Pipeline de extracción
│   └── connectors/
│       ├── base.py                # DataConnector abstracto
│       ├── wikipedia.py           # MediaWiki REST API
│       └── archiveorg.py          # Internet Archive API
├── tests/
│   ├── unit/                      # 113 tests unitarios
│   └── integration/               # 7 tests end-to-end
├── .kiro/specs/psychohistory-engine/
│   ├── requirements.md            # 10 requisitos formales
│   ├── design.md                  # Diseño técnico completo
│   └── tasks.md                   # Plan de implementación
└── pyproject.toml
```

---

## Posibilidades y extensiones

### Análisis histórico
- Cargar corpus de eventos históricos desde Wikipedia, Archive.org o fuentes propias
- Detectar ciclos históricos recurrentes (guerras cada N años, crisis económicas periódicas)
- Visualizar grafos de causalidad entre categorías de eventos

### Predicción y escenarios
- Generar múltiples escenarios futuros con diferentes horizontes temporales (1–1000 años)
- Comparar trayectorias bajo diferentes condiciones iniciales
- Calcular el impacto diferencial de intervenciones específicas

### Identificación de puntos críticos
- Encontrar los momentos históricos donde una acción mínima tiene el mayor impacto
- Ordenar Crisis Seldon por índice de sensibilidad
- Identificar correlaciones no locales entre eventos distantes (entrelazamiento)

### Extensiones posibles
- **Interfaz web**: API REST sobre el motor para consultas interactivas
- **Visualización**: grafos de trayectorias y Crisis Seldon en tiempo real
- **Más conectores**: Wikidata, GDELT, bases de datos históricas especializadas
- **Hardware cuántico real**: conectar a IBM Quantum para análisis de alta dimensionalidad
- **NLP avanzado**: integrar spaCy o transformers para extracción de entidades más precisa
- **Análisis geoespacial**: incorporar coordenadas para patrones geográficos

---

## Limitaciones conocidas

- El sistema requiere un mínimo de **1,000 eventos** en el Corpus para generar predicciones confiables
- Los backends cuánticos reales (PennyLane/Qiskit) son opcionales; sin ellos el sistema usa simulación clásica numpy
- La extracción de entidades desde texto usa regex simple; para mayor precisión se recomienda integrar spaCy
- Las predicciones a largo plazo (> 100 años) tienen incertidumbre creciente por diseño (principio de incertidumbre social)
- El sistema modela comportamiento agregado de poblaciones, no eventos individuales

---

## Inspiración

Este proyecto es una implementación computacional de la **psicohistoria** de Isaac Asimov, la ciencia ficticia que Hari Seldon desarrolló en la saga *Fundación*. La psicohistoria combinaba historia, sociología y matemáticas para predecir el comportamiento de grandes poblaciones. Este motor reemplaza las matemáticas ficticias de Asimov con algoritmos reales: computación cuántica, machine learning y teoría de grafos.

> *"No puedes predecir lo que hará un individuo, pero puedes predecir con precisión lo que hará una masa de individuos."*

---

## Licencia

MIT License — ver `LICENSE` para detalles.
