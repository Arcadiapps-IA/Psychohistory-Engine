"

# Psychohistory Engine

> *"Psychohistory was the branch of mathematics that dealt with the reactions of human conglomerates to fixed social and economic stimuli."*
> — Isaac Asimov, *Foundation*

A computational framework that combines **quantum physics**, **artificial intelligence**, and **deterministic systems** to analyze historical patterns, predict future trajectories, and identify critical intervention points where minimal action can alter the course of events — directly inspired by the psychohistory of Hari Seldon.

---

## What is this?

The Psychohistory Engine is based on a premise: although individual behavior is unpredictable, the aggregated behavior of large populations over time follows statistical patterns that can be modeled. With enough historical data and the right algorithms, it is possible to compute the most probable trajectories of the future and find the moments — the **Seldon Crises** — where intervention has the greatest impact.

The system ingests historical events from multiple sources (including Wikipedia and Archive.org), detects correlations and cycles using classical and quantum algorithms, generates probabilistic predictions, and identifies critical bifurcation points along with their sensitivity index.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PsychohistoryEngine                       │
│                      (Orchestrator)                          │
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
                    (Quantum facade)
                    PennyLane / Qiskit
                    + classical fallback

External sources:
  WikipediaConnector ──► ExtractionPipeline ──► Corpus (SQLite/PostgreSQL)
  ArchiveOrgConnector ─►
```

### Main subsystems

| Module                     | Responsibility                                                     |
| -------------------------- | ------------------------------------------------------------------ |
| `engine.py`                | Main orchestrator — public API of the system                       |
| `event_ingester.py`        | Ingestion and normalization of historical events (JSON, CSV, text) |
| `pattern_analyzer.py`      | Detection of classical and quantum social patterns                 |
| `trajectory_predictor.py`  | Prediction of future trajectories with QAOA + Grover               |
| `intervention_detector.py` | Identification of Seldon Crises with quantum entanglement          |
| `quantum_engine.py`        | Quantum facade (VQC, QAOA, Grover, Von Neumann entropy)            |
| `uncertainty.py`           | Social uncertainty principle (analogous to Heisenberg)             |
| `persistence.py`           | Corpus repository with SQLAlchemy                                  |
| `serialization.py`         | State export/import with MessagePack + SHA-256                     |
| `explainability.py`        | Traceability and human-readable reports                            |
| `extraction_pipeline.py`   | Extraction pipeline from external sources                          |
| `connectors/wikipedia.py`  | MediaWiki REST API connector                                       |
| `connectors/archiveorg.py` | Internet Archive API connector                                     |

---

## Quantum physics in the system

The quantum component is not decorative — it is integrated into four fundamental aspects of the analysis:

### 1. Quantum superposition in trajectory prediction

The space of possible futures is modeled as a **quantum state in superposition** before being "collapsed" at query time. The QAOA (Quantum Approximate Optimization Algorithm) explores this space efficiently, and when the space exceeds 10,000 combinations, Grover’s algorithm provides a quadratic advantage in search.

### 2. Quantum Machine Learning in pattern detection

When the feature space of historical data exceeds 50 dimensions, the system delegates correlation detection to the **Quantum Engine** using **Variational Quantum Circuits (VQC)** with `AngleEmbedding + StronglyEntanglingLayers` architecture. This enables detection of correlations that classical algorithms would not find.

### 3. Quantum entanglement in Seldon Crises

The **Sensitivity Index** of each intervention point incorporates the **Von Neumann entropy** of the reduced quantum state between pairs of trajectory nodes. Two historically distant events with high entanglement entropy are non-locally correlated — an intervention in one affects the other.

```
SI(node) = 0.6 × D(node) + 0.4 × E(node)

where:
  D = average divergence with N=10 perturbations ±10%
  E = average Entanglement_Metric (Von Neumann entropy)
  
Seldon Crisis: SI > 0.7
```

### 4. Social uncertainty principle

The system recognizes a fundamental theoretical limit analogous to Heisenberg’s principle:

```
σ_state × σ_momentum ≥ ħ_social (= 0.01 by default)
```

It is not possible to simultaneously know with arbitrary precision the **state** of a society and its **momentum of change**. For time horizons beyond 100 years, uncertainty scales proportionally.

---

## Implemented algorithms

### Quantum algorithms

| Algorithm                                   | Use in the system                                     | Qubit limit |
| ------------------------------------------- | ----------------------------------------------------- | ----------- |
| **VQC** (Variational Quantum Circuit)       | Pattern detection in high dimensionality              | ≤ 50 qubits |
| **QAOA** (Quantum Approximate Optimization) | Exploration of trajectory space                       | ≤ 30 qubits |
| **Grover Search**                           | Optimal trajectory search (quadratic advantage O(√N)) | ≤ 20 qubits |
| **Von Neumann Entropy**                     | Entanglement calculation between nodes                | Pure NumPy  |

### Classical algorithms

| Algorithm                     | Use                                                      |
| ----------------------------- | -------------------------------------------------------- |
| **Co-occurrence by periods**  | Pattern detection in 50-year windows                     |
| **Cycle analysis**            | Detection of recurrent patterns (1–500 years)            |
| **Directed causality graphs** | Representation of relationships between event categories |
| **Wasserstein distance**      | Divergence calculation between perturbed trajectories    |
| **Logarithmic product**       | Trajectory Confidence Score (avoids numerical underflow) |
| **SHA-256**                   | Integrity of state files and Corpus deduplication        |
| **Partial trace**             | Reduced density matrix calculation for entanglement      |

---

## Technologies

### Core

* **Python 3.11+** — main language
* **NumPy** — linear algebra, partial trace, probability calculations
* **SciPy** — Wasserstein distance for the Sensitivity Index
* **NetworkX** — directed causality graphs between event categories

### Quantum computing

* **PennyLane 0.38+** — primary backend for VQC (native automatic differentiation)
* **Qiskit 1.x** — secondary backend for QAOA and Grover, access to IBM Quantum hardware
* **pennylane-qiskit** — integration plugin between both
* **qiskit-aer** — local simulator when hardware is not available
* *Automatic fallback*: if no quantum backend is available, the system operates with classical NumPy simulation while maintaining the same interface

### Persistence

* **SQLAlchemy 2.0** — ORM for the data layer
* **SQLite** — database for development/prototyping
* **PostgreSQL** — production database
* **MessagePack** — efficient binary serialization of system state

### External data sources

* **MediaWiki REST API** — extraction of Wikipedia articles
* **internetarchive** — official Internet Archive / Archive.org library
* **requests** — HTTP client

### Testing

* **pytest** — testing framework
* **Hypothesis 6.x** — property-based testing (23 correctness properties)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/tu-usuario/psychohistory-engine.git
cd psychohistory-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# For real quantum backend (PennyLane)
pip install pennylane>=0.38

# For IBM Quantum hardware (Qiskit)
pip install qiskit>=1.0 qiskit-aer pennylane-qiskit

# For extraction from Archive.org
pip install internetarchive

# For extraction from Wikipedia
pip install requests
```

> The system works without any of these optional dependencies using classical NumPy fallback.

---

## Quick start

```python
from psychohistory.engine import PsychohistoryEngine, PredictionParams

# Initialize engine
engine = PsychohistoryEngine(
    db_url="sqlite:///mi_corpus.db",
    seed=42
)

# Ingest historical events
events = [
    {
        "date": "1789-07-14",
        "description": "Storming of the Bastille. Beginning of the French Revolution.",
        "category": "POLITICAL",
        "actors": ["Third Estate", "National Assembly"],
        "magnitude": 0.95
    },
    # ... more events
]
report = engine.ingest_events(events, format="json")
print(f"Accepted: {report.accepted}, Rejected: {report.rejected}")

# Predict trajectories (requires ≥ 1000 events in the Corpus)
result = engine.predict(
    horizon_years=50,
    params=PredictionParams(seed=42)
)

# View the 3 most probable trajectories
for i, traj in enumerate(result.trajectories):
    print(f"\nTrajectory {i+1}:")
    print(f"  Confidence: {traj.confidence_score:.4f}")
    print(f"  Uncertainty: σ_state={traj.uncertainty_bound.sigma_state:.4f}")
    for node in traj.nodes:
        print(f"  [{node.sequence_index}] {node.predicted_event.description}")

# View identified Seldon Crises
for i, crisis_list in enumerate(result.intervention_points):
    for crisis in crisis_list:
        print(f"\nSeldon Crisis in trajectory {i+1}:")
        print(f"  Sensitivity Index: {crisis.sensitivity_index:.4f}")
        print(f"  Estimated date: {crisis.temporal_coordinates.year}")
        print(f"  Recommended action: {crisis.recommended_action_type}")

# Get detailed explanation
for traj in result.trajectories:
    report = engine.get_explanation(traj.id)
    print(report.uncertainty_description)
```

### Extraction from Wikipedia

```python
from psychohistory.connectors.wikipedia import WikipediaConnector
from psychohistory.connectors.base import SearchQuery

# Configure connector
connector = WikipediaConnector()
engine.configure_connector("wikipedia", connector)

# Extract articles about revolutions
extraction_report = engine.trigger_extraction(
    "wikipedia",
    SearchQuery(text="French Revolution industrial revolution")
)
print(f"Documents retrieved: {extraction_report.documents_retrieved}")
print(f"Events generated: {extraction_report.events_generated}")
```

### Extraction from Archive.org

```python
from psychohistory.connectors.archiveorg import ArchiveOrgConnector

connector = ArchiveOrgConnector()
engine.configure_connector("archiveorg", connector)

extraction_report = engine.trigger_extraction(
    "archiveorg",
    SearchQuery(collection="americana", text="historical newspapers 1800")
)
```

### Query the Corpus

```python
from psychohistory.persistence import CorpusQuery
from psychohistory.enums import EventCategory

# Filter political events between 1700 and 1900
from datetime import datetime
results = engine.query_corpus(CorpusQuery(
    date_from=datetime(1700, 1, 1),
    date_to=datetime(1900, 12, 31),
    categories=[EventCategory.POLITICAL, EventCategory.MILITARY],
    operator="OR"
))
print(f"Events found: {len(results)}")
```

### Export and import state

```python
# Save full system state
engine.export_state("estado_2024.msgpack")

# Restore in another instance
engine2 = PsychohistoryEngine(db_url="sqlite:///:memory:")
engine2.import_state("estado_2024.msgpack")
```

---

## Event categories

The system classifies historical events into six categories:

| Category    | Description                                            |
| ----------- | ------------------------------------------------------ |
| `POLITICAL` | Revolutions, elections, government changes, treaties   |
| `ECONOMIC`  | Financial crises, bubbles, changes in economic systems |
| `SOCIAL`    | Social movements, cultural changes, demography         |
| `MILITARY`  | Wars, battles, armed conflicts                         |
| `CULTURAL`  | Scientific discoveries, artistic movements, religion   |
| `NATURAL`   | Pandemics, natural disasters, climate changes          |

---

## Verified correctness properties

The system includes **23 correctness properties** verified with property-based testing (Hypothesis), covering:

* Normalization preserves the canonical event schema
* Correct rejection of events with missing required fields
* ID uniqueness during ingestion
* Pattern Confidence Score always in [0.0, 1.0]
* Patterns with confidence < 0.3 are discarded
* Corpus update preserves unrelated patterns
* Prediction returns exactly 3 trajectories
* Confidence Score = product of individual probabilities
* Determinism with the same seed
* Entanglement Metric always in [0.0, 1.0]
* Uncertainty Bound invariant: σ_state × σ_momentum ≥ ħ_social
* Monotonicity of Uncertainty Bound with time horizon
* Intervention Points sorted by descending Sensitivity Index
* Serialization round-trip produces equivalent state
* SHA-256 integrity of the state file
* Completeness of the reasoning trace
* Preservation of source document in the pipeline
* Completeness of attribution metadata
* Temporal deduplication in incremental extractions

```bash
# Run all tests
pytest tests/ -v

# Property tests only
pytest tests/unit/ -v -k "test_" --hypothesis-seed=42

# End-to-end integration tests
pytest tests/integration/ -v
```

---

## Project structure

```
psychohistory-engine/
├── psychohistory/
│   ├── __init__.py
│   ├── engine.py                  # Main orchestrator
│   ├── enums.py                   # EventCategory
│   ├── models.py                  # Dataclasses (HistoricalEvent, Trajectory, etc.)
│   ├── exceptions.py              # Exception hierarchy
│   ├── event_ingester.py          # Ingestion and normalization
│   ├── pattern_analyzer.py        # Pattern detection
│   ├── trajectory_predictor.py    # Trajectory prediction
│   ├── intervention_detector.py   # Seldon Crises
│   ├── quantum_engine.py          # Quantum facade
│   ├── uncertainty.py             # Uncertainty principle
│   ├── persistence.py             # CorpusRepository (SQLAlchemy)
│   ├── serialization.py           # MessagePack export/import
│   ├── explainability.py          # Traceability reports
│   ├── extraction_pipeline.py     # Extraction pipeline
│   └── connectors/
│       ├── base.py                # Abstract DataConnector
│       ├── wikipedia.py           # MediaWiki REST API
│       └── archiveorg.py          # Internet Archive API
├── tests/
│   ├── unit/                      # 113 unit tests
│   └── integration/               # 7 end-to-end tests
├── .kiro/specs/psychohistory-engine/
│   ├── requirements.md            # 10 formal requirements
│   ├── design.md                  # Complete technical design
│   └── tasks.md                   # Implementation plan
└── pyproject.toml
```

---

## Possibilities and extensions

### Historical analysis

* Load historical event corpora from Wikipedia, Archive.org, or custom sources
* Detect recurring historical cycles (wars every N years, periodic economic crises)
* Visualize causality graphs between event categories

### Prediction and scenarios

* Generate multiple future scenarios with different time horizons (1–1000 years)
* Compare trajectories under different initial conditions
* Calculate differential impact of specific interventions

### Identification of critical points

* Find historical moments where minimal action has maximum impact
* Rank Seldon Crises by sensitivity index
* Identify non-local correlations between distant events (entanglement)

### Possible extensions

* **Web interface**: REST API over the engine for interactive queries
* **Visualization**: real-time graphs of trajectories and Seldon Crises
* **More connectors**: Wikidata, GDELT, specialized historical databases
* **Real quantum hardware**: connect to IBM Quantum for high-dimensional analysis
* **Advanced NLP**: integrate spaCy or transformers for more accurate entity extraction
* **Geospatial analysis**: incorporate coordinates for geographic patterns

---

## Known limitations

* The system requires a minimum of **1,000 events** in the Corpus to generate reliable predictions
* Real quantum backends (PennyLane/Qiskit) are optional; without them the system uses classical NumPy simulation
* Entity extraction from text uses simple regex; for higher accuracy, integrating spaCy is recommended
* Long-term predictions (> 100 years) have increasing uncertainty by design (social uncertainty principle)
* The system models aggregated population behavior, not individual events

---

## Inspiration

This project is a computational implementation of Isaac Asimov’s **psychohistory**, the fictional science developed by Hari Seldon in the *Foundation* saga. Psychohistory combined history, sociology, and mathematics to predict the behavior of large populations. This engine replaces Asimov’s fictional mathematics with real algorithms: quantum computing, machine learning, and graph theory.

> *"You cannot predict what an individual will do, but you can predict with accuracy what a mass of individuals will do."*

---

## License

MIT License — see `LICENSE` for details.

"


-----------------------------------------------------




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
