# Plan de Implementacion: Psychohistory Engine

## Resumen

Plan de implementacion incremental del Psychohistory Engine en Python 3.11+. El sistema combina computacion cuantica (PennyLane/Qiskit), machine learning (spaCy NLP), grafos de causalidad (NetworkX) y conectores externos (Wikipedia, Archive.org). Las tareas siguen el orden de dependencias: infraestructura y modelos de datos, subsistemas nucleares, motor cuantico, pipeline de extraccion, integracion y tests de propiedades.

## Tareas

- [x] 1. Configurar infraestructura del proyecto y modelos de datos base
  - Crear la estructura de directorios del proyecto: `psychohistory/`, `tests/unit/`, `tests/integration/`
  - Crear `pyproject.toml` con dependencias: pennylane>=0.38, qiskit>=1.0, networkx>=3.0, spacy>=3.0, msgpack, hypothesis>=6.0, pytest, sqlalchemy, apscheduler
  - Implementar `psychohistory/models.py` con todos los dataclasses: `HistoricalEvent`, `Location`, `SocialPattern`, `TrajectoryNode`, `PredictedEvent`, `EntanglementCorrelation`, `UncertaintyBound`, `Trajectory`, `InterventionPoint`, `RawSourceDocument`, `SystemState`, `ReasoningTrace`, `ReasoningStep`
  - Implementar `psychohistory/exceptions.py` con la jerarquia de excepciones: `PsychohistoryError`, `ValidationError`, `BatchIngestionError`, `InsufficientDataError`, `InvalidHorizonError`, `QuantumExecutionError`, `ConnectorTimeoutError`, `ConnectorRateLimitError`, `StateIntegrityError`, `StateImportError`
  - Implementar `psychohistory/enums.py` con `EventCategory` (POLITICAL, ECONOMIC, SOCIAL, MILITARY, CULTURAL, NATURAL)
  - Crear `tests/conftest.py` con fixtures compartidas y estrategias Hypothesis: `historical_events()`, `valid_corpus()`, `valid_system_states()`
  - _Requisitos: 1.1, 1.2, 1.4_

- [x] 2. Implementar Event_Ingester y persistencia en Corpus
  - [x] 2.1 Implementar `psychohistory/event_ingester.py` con la clase `EventIngester`
    - Metodo `ingest(raw, format)`: parsea JSON, CSV y texto plano estructurado; normaliza campos al esquema canonico; asigna UUID v4; persiste en Corpus
    - Metodo `ingest_batch(source, format)`: procesa hasta 100,000 eventos; retorna `BatchIngestionReport` con conteos de aceptados/rechazados y motivos
    - Metodo `_normalize(raw)`: mapea campos al dataclass `HistoricalEvent`
    - Metodo `_validate(event)`: verifica presencia de `date` y `description`; lanza `ValidationError` con lista de campos faltantes
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  - [ ]* 2.2 Escribir test de propiedad: Normalizacion preserva esquema canonico (Propiedad 1)
    - **Propiedad 1: Normalizacion preserva el esquema canonico**
    - **Valida: Requisito 1.2**
  - [ ]* 2.3 Escribir test de propiedad: Rechazo de eventos con campos obligatorios ausentes (Propiedad 2)
    - **Propiedad 2: Rechazo de eventos con campos obligatorios ausentes**
    - **Valida: Requisito 1.3**
  - [ ]* 2.4 Escribir test de propiedad: Ingesta produce IDs unicos y persistencia verificable (Propiedad 3)
    - **Propiedad 3: Ingesta produce IDs unicos y persistencia verificable**
    - **Valida: Requisito 1.4**
  - [ ]* 2.5 Escribir test de propiedad: Consistencia del reporte de ingesta por lotes (Propiedad 4)
    - **Propiedad 4: Consistencia del reporte de ingesta por lotes**
    - **Valida: Requisito 1.6**

- [x] 3. Checkpoint - Verificar ingesta y modelos base
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

- [x] 4. Implementar capa de persistencia (SQLite/PostgreSQL)
  - [x] 4.1 Implementar `psychohistory/persistence.py` con `CorpusRepository`
    - Esquema SQLAlchemy para `HistoricalEvent` con indices en `date`, `category`, `location`
    - Metodos: `save(event)`, `save_batch(events)`, `find_by_id(id)`, `query(filters)`, `count()`, `get_snapshot_hash()`
    - Soporte para filtros combinados con AND/OR por rango de fechas, categoria, lugar y actores
    - Respuesta en menos de 2 segundos para corpus de hasta 1,000,000 eventos (indices apropiados)
    - _Requisitos: 1.4, 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 4.2 Escribir test de propiedad: Corrección de filtros en consultas del Corpus (Propiedad 17)
    - **Propiedad 17: Corrección de filtros en consultas del Corpus**
    - **Valida: Requisito 7.4**

- [x] 5. Implementar Quantum_Engine (Facade)
  - [x] 5.1 Implementar `psychohistory/quantum_engine.py` con la clase `QuantumEngine` y submodulos
    - `BackendFactory`: seleccion de backend (hardware IBM Quantum vs simulador `default.qubit`); fallback automatico con timeout de 10 segundos; logging de backend utilizado
    - `VQCModule`: `build_vqc(n_qubits, n_layers, seed)` con `AngleEmbedding` + `StronglyEntanglingLayers`; limite de 50 qubits; entrenamiento con Adam y diferenciacion automatica PennyLane
    - `QAOAModule`: `run_qaoa(cost_operator, n_qubits, p_layers, seed)` con Qiskit QAOA + COBYLA; limite de 30 qubits
    - `GroverModule`: `run_grover(oracle, n_qubits, iterations, seed)` con Qiskit Grover; limite de 20 qubits
    - `EntanglementModule`: `von_neumann_entropy(state_vector, subsystem_qubits, total_qubits)` con traza parcial y normalizacion a [0.0, 1.0]
    - Metodo `_log_execution(circuit_type, n_qubits, iterations, ms, backend, seed)` para auditoria
    - Validacion de parametros fuera de rango con `QuantumExecutionError`
    - _Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_
  - [ ]* 5.2 Escribir test de propiedad: Entanglement_Metric esta en rango valido (Propiedad 13)
    - **Propiedad 13: Entanglement_Metric esta en rango valido [0.0, 1.0]**
    - **Valida: Requisito 4.4**
  - [ ]* 5.3 Escribir test de propiedad: Determinismo de predicciones con la misma semilla (Propiedad 11, parte cuantica)
    - **Propiedad 11: Determinismo de predicciones con la misma semilla**
    - **Valida: Requisitos 3.7, 5.9**

- [x] 6. Checkpoint - Verificar Quantum_Engine
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

- [x] 7. Implementar Pattern_Analyzer
  - [x] 7.1 Implementar `psychohistory/pattern_analyzer.py` con la clase `PatternAnalyzer`
    - Metodo `analyze(corpus)`: correlacion clasica para dimensionalidad <= 50; delegacion a `QuantumEngine.train_vqc()` para dimensionalidad > 50
    - Metodo `recalculate_affected(new_events)`: recalcula solo patrones afectados por categorias de los nuevos eventos; preserva patrones no relacionados
    - Metodo `_classical_correlation(events)`: calcula correlaciones estadisticas y construye `SocialPattern`
    - Metodo `_quantum_correlation(feature_matrix)`: invoca VQC y mapea resultado a `SocialPattern` con `is_quantum_detected=True` y `qubits_used`
    - Metodo `_build_causality_graph(pattern)`: construye `nx.DiGraph` de causalidad entre categorias de eventos
    - Descarte de patrones con `Confidence_Score < 0.3` con registro en log
    - Deteccion de patrones ciclicos con periodos entre 1 y 500 anos
    - _Requisitos: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_
  - [x]* 7.2 Escribir test de propiedad: Confidence_Score de patron esta en rango valido (Propiedad 5)
    - **Propiedad 5: Confidence_Score de patron esta en rango valido [0.0, 1.0]**
    - **Valida: Requisito 2.2**
  - [x]* 7.3 Escribir test de propiedad: Patrones con Confidence_Score bajo son descartados (Propiedad 6)
    - **Propiedad 6: Patrones con Confidence_Score bajo son descartados**
    - **Valida: Requisito 2.3**
  - [x]* 7.4 Escribir test de propiedad: Actualizacion del Corpus preserva patrones no relacionados (Propiedad 7)
    - **Propiedad 7: Actualizacion del Corpus preserva patrones no relacionados**
    - **Valida: Requisito 2.5**
  - [x]* 7.5 Escribir test de propiedad: Delegacion cuantica por dimensionalidad (Propiedad 8)
    - **Propiedad 8: Delegacion cuantica por dimensionalidad (mock del QuantumEngine)**
    - **Valida: Requisito 2.7**

- [x] 8. Implementar calculo del Uncertainty_Bound
  - [x] 8.1 Implementar `psychohistory/uncertainty.py` con la funcion `compute_uncertainty_bound(sigma_state, sigma_momentum, h_social, horizon_years)`
    - Escalar incertidumbres proporcionalmente cuando `horizon_years > 100`
    - Ajuste proporcional con `sqrt(h_social / product)` cuando el producto viola la restriccion
    - Retornar `UncertaintyBound` con `was_adjusted=True` y `adjustment_reason` cuando se aplica ajuste
    - Advertencia al usuario cuando `sigma_state < 0.05`
    - _Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_
  - [ ]* 8.2 Escribir test de propiedad: Invariante del Uncertainty_Bound (Propiedad 15)
    - **Propiedad 15: Invariante del Uncertainty_Bound (product >= h_social)**
    - **Valida: Requisitos 6.1, 6.2, 6.3**
  - [ ]* 8.3 Escribir test de propiedad: Monotonia del Uncertainty_Bound con el horizonte temporal (Propiedad 16)
    - **Propiedad 16: Monotonia del Uncertainty_Bound con el horizonte temporal**
    - **Valida: Requisito 6.7**

- [x] 9. Implementar Trajectory_Predictor
  - [x] 9.1 Implementar `psychohistory/trajectory_predictor.py` con la clase `TrajectoryPredictor`
    - Metodo `predict(patterns, horizon, seed)`: verifica corpus >= 1000 eventos (lanza `InsufficientDataError`); valida horizonte en [1, 1000] (lanza `InvalidHorizonError`); solicita `Quantum_Trajectory_State` via QAOA; aplica Grover si espacio > 10,000 combinaciones; colapsa a 3 trayectorias de mayor amplitud
    - Metodo `_apply_uncertainty_bound(trajectory)`: invoca `compute_uncertainty_bound()` e incluye resultado en la trayectoria; escala por horizonte > 100 anos
    - Metodo `_invalidate_stale(updated_patterns)`: recalcula `Confidence_Score` y descarta trayectorias con variacion > 0.05
    - Metodo `_compute_confidence_score(trajectory)`: producto de probabilidades individuales en espacio logaritmico para evitar underflow
    - Generacion de `ReasoningTrace` con `ReasoningStep` por cada nodo vinculando patrones y eventos historicos
    - Garantia de determinismo con semilla aleatoria fija
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11_
  - [ ]* 9.2 Escribir test de propiedad: Prediccion retorna exactamente tres trayectorias (Propiedad 9)
    - **Propiedad 9: Prediccion retorna exactamente tres trayectorias**
    - **Valida: Requisito 3.1**
  - [ ]* 9.3 Escribir test de propiedad: Confidence_Score de trayectoria es el producto de probabilidades individuales (Propiedad 10)
    - **Propiedad 10: Confidence_Score de trayectoria es el producto de probabilidades individuales**
    - **Valida: Requisito 3.2**
  - [ ]* 9.4 Escribir test de propiedad: Determinismo de predicciones con la misma semilla (Propiedad 11)
    - **Propiedad 11: Determinismo de predicciones con la misma semilla**
    - **Valida: Requisitos 3.7, 5.9**

- [x] 10. Checkpoint - Verificar prediccion de trayectorias
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

- [x] 11. Implementar Intervention_Detector
  - [x] 11.1 Implementar `psychohistory/intervention_detector.py` con la clase `InterventionDetector`
    - Metodo `detect(trajectory)`: analiza cada nodo; clasifica como `Seldon_Crisis` si `Sensitivity_Index > 0.7`; retorna lista vacia con indicador de estabilidad si no hay crisis
    - Metodo `_compute_sensitivity_index(node, trajectory)`: formula `SI = 0.6 * D + 0.4 * E`; calcula D con N=10 perturbaciones +-10% y distancia de Wasserstein; calcula E como `Entanglement_Metric` promedio via `QuantumEngine.von_neumann_entropy()`
    - Metodo `_compute_differential_impact(point, trajectory)`: divergencia entre trayectoria original y post-intervencion
    - Generacion de `InterventionPoint` con coordenadas temporales, actores, accion recomendada, `Sensitivity_Index`, nodos entrelazados (Entanglement_Metric > 0.6) y patrones de soporte
    - Ordenamiento de `InterventionPoints` de mayor a menor `Sensitivity_Index`
    - Registro de correlaciones no locales entre nodos con `Entanglement_Metric > 0.6`
    - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_
  - [ ]* 11.2 Escribir test de propiedad: Corrección del Sensitivity_Index y clasificacion de Crisis Seldon (Propiedad 12)
    - **Propiedad 12: Corrección del Sensitivity_Index y clasificacion de Crisis Seldon**
    - **Valida: Requisitos 4.2, 4.3**
  - [ ]* 11.3 Escribir test de propiedad: Intervention_Points ordenados por Sensitivity_Index descendente (Propiedad 14)
    - **Propiedad 14: Intervention_Points ordenados por Sensitivity_Index descendente**
    - **Valida: Requisito 4.7**

- [x] 12. Implementar serializacion y persistencia del estado del sistema
  - [x] 12.1 Implementar `psychohistory/serialization.py` con funciones `export_state(engine_state, path)` e `import_state(path)`
    - Serializacion a MessagePack con hash de integridad SHA-256 (excluye el propio campo `integrity_hash`)
    - Verificacion del hash en importacion; lanza `StateIntegrityError` si no coincide
    - Lanza `StateImportError` si el formato es invalido
    - Soporte para round-trip completo: Corpus + SocialPatterns + Trajectories activas + ConnectorConfigs
    - _Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5_
  - [ ]* 12.2 Escribir test de propiedad: Round-trip de serializacion del estado del sistema (Propiedad 18)
    - **Propiedad 18: Round-trip de serializacion del estado del sistema**
    - **Valida: Requisito 8.3**
  - [ ]* 12.3 Escribir test de propiedad: Integridad del archivo de estado exportado (Propiedad 19)
    - **Propiedad 19: Integridad del archivo de estado exportado (SHA-256)**
    - **Valida: Requisito 8.5**

- [x] 13. Implementar trazabilidad y explicabilidad
  - [x] 13.1 Implementar `psychohistory/explainability.py` con la clase `ExplainabilityReporter`
    - Metodo `get_explanation(trajectory_id)`: recupera `ReasoningTrace` de la trayectoria; genera reporte legible por humanos con cadena de evidencia evento -> patron -> prediccion
    - Inclusion del `Uncertainty_Bound` en el reporte con descripcion legible de la restriccion aplicada
    - Inclusion de `Social_Patterns` que elevan el `Sensitivity_Index` en cada `InterventionPoint`
    - _Requisitos: 9.1, 9.2, 9.3, 9.4_
  - [ ]* 13.2 Escribir test de propiedad: Completitud de la traza de razonamiento (Propiedad 20)
    - **Propiedad 20: Completitud de la traza de razonamiento**
    - **Valida: Requisito 9.1**

- [x] 14. Checkpoint - Verificar subsistemas nucleares integrados
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

- [x] 15. Implementar Data_Connectors y Extraction_Pipeline
  - [x] 15.1 Implementar `psychohistory/connectors/base.py` con la clase abstracta `DataConnector`
    - Metodos abstractos: `search(query)`, `fetch(identifier)`
    - Metodo `_handle_rate_limit(retry_after)`: pausa exactamente `retry_after` segundos ante HTTP 429
    - Metodo `_handle_timeout()`: registra fallo con timestamp y numero de reintentos; lanza `ConnectorTimeoutError`
    - Politica de reintentos con backoff exponencial (1s, 2s, 4s, maximo 3 reintentos) para errores de red transitorios
    - _Requisitos: 10.7, 10.8_
  - [x] 15.2 Implementar `psychohistory/connectors/wikipedia.py` con `WikipediaConnector`
    - Busqueda por titulo, categoria y texto libre via MediaWiki REST API (`https://en.wikipedia.org/w/api.php`)
    - Timeout de 30 segundos; manejo de HTTP 429 con cabecera `Retry-After`
    - Mapeo de respuesta a `RawSourceDocument` con `connector_name="wikipedia"` y `extraction_timestamp`
    - _Requisitos: 10.1, 10.7, 10.8_
  - [x] 15.3 Implementar `psychohistory/connectors/archiveorg.py` con `ArchiveOrgConnector`
    - Busqueda por coleccion, rango de fechas y texto libre via biblioteca `internetarchive`
    - Mapeo de resultado a `RawSourceDocument` con `connector_name="archiveorg"` y `extraction_timestamp`
    - _Requisitos: 10.2, 10.7, 10.8_
  - [x] 15.4 Implementar `psychohistory/extraction_pipeline.py` con la clase `ExtractionPipeline`
    - Metodo `run(connector, query)`: recupera documentos; persiste `RawSourceDocument` sin modificar; transforma via spaCy NER; deduplica; ingesta en Corpus; retorna `ExtractionReport`
    - Metodo `_transform(doc)`: usa spaCy `en_core_web_trf` para extraer DATE/TIME -> `date`, GPE/LOC -> `location`, PERSON/ORG/NORP -> `actors`
    - Metodo `_deduplicate(events)`: hash SHA-256 de `(date, description)` normalizado; filtra duplicados existentes en Corpus
    - Metodo `_schedule(connector_id, frequency_hours)`: valida rango [1, 8760]; registra tarea en APScheduler
    - Descarte silencioso de documentos sin campos obligatorios con registro en log (ID + motivo)
    - Inclusion de `source_url`, `connector_name`, `extraction_timestamp` en metadatos de cada `HistoricalEvent`
    - Extraccion incremental: filtra documentos con fecha anterior al ultimo checkpoint del conector
    - _Requisitos: 10.3, 10.4, 10.5, 10.6, 10.9, 10.10, 10.11, 10.12_
  - [ ]* 15.5 Escribir test de propiedad: Preservacion del documento fuente en el pipeline (Propiedad 21)
    - **Propiedad 21: Preservacion del documento fuente en el pipeline**
    - **Valida: Requisito 10.3**
  - [ ]* 15.6 Escribir test de propiedad: Completitud de metadatos de atribucion en el pipeline (Propiedad 22)
    - **Propiedad 22: Completitud de metadatos de atribucion en el pipeline**
    - **Valida: Requisitos 10.4, 10.6**
  - [ ]* 15.7 Escribir test de propiedad: Deduplicacion temporal en extracciones incrementales (Propiedad 23)
    - **Propiedad 23: Deduplicacion temporal en extracciones incrementales**
    - **Valida: Requisito 10.11**

- [x] 16. Checkpoint - Verificar pipeline de extraccion
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

- [x] 17. Implementar Psychohistory_Engine (orquestador) y API publica
  - [x] 17.1 Implementar `psychohistory/engine.py` con la clase `PsychohistoryEngine`
    - Metodo `ingest_events(events, format)`: delega a `EventIngester.ingest_batch()`; retorna `IngestionReport`
    - Metodo `predict(horizon_years, params)`: verifica corpus >= 1000; obtiene patrones activos; delega a `TrajectoryPredictor`; detecta crisis via `InterventionDetector`; retorna `PredictionResult`
    - Metodo `query_corpus(filters)`: delega a `CorpusRepository.query()`; retorna lista vacia con mensaje si no hay resultados; valida formato de filtros
    - Metodo `export_state(path)`: delega a `export_state()` de serializacion
    - Metodo `import_state(path)`: delega a `import_state()` de serializacion; no modifica estado actual si falla
    - Metodo `get_explanation(trajectory_id)`: delega a `ExplainabilityReporter`
    - Metodo `configure_connector(connector_id, config)`: configura y valida el conector especificado
    - Metodo `trigger_extraction(connector_id)`: ejecuta extraccion manual via `ExtractionPipeline`
    - Advertencia al usuario cuando `sigma_state < 0.05` en solicitudes de prediccion
    - _Requisitos: 1.1, 3.3, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.4, 9.2, 9.4, 10.10_

- [x] 18. Conectar todos los subsistemas e integracion final
  - [x] 18.1 Cablear `PsychohistoryEngine` con todos los subsistemas
    - Inyeccion de dependencias: `EventIngester`, `PatternAnalyzer`, `TrajectoryPredictor`, `InterventionDetector`, `QuantumEngine`, `ExtractionPipeline`, `CorpusRepository`, `ExplainabilityReporter`
    - Flujo completo: ingesta -> analisis de patrones -> prediccion cuantica -> deteccion de crisis -> explicabilidad
    - Invalidacion automatica de trayectorias al actualizar el Corpus (variacion > 0.05 en `Confidence_Score`)
    - _Requisitos: 2.5, 3.6, 5.1_
  - [x]* 18.2 Escribir tests de integracion del pipeline completo
    - Test end-to-end: ingesta de corpus sintetico -> analisis -> prediccion -> deteccion de crisis -> exportacion/importacion de estado
    - Verificar que el flujo completo produce resultados deterministas con la misma semilla
    - _Requisitos: 3.7, 8.3_

- [x] 19. Checkpoint final - Verificar integracion completa
  - Asegurar que todos los tests pasen, consultar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP mas rapido
- Cada tarea referencia requisitos especificos para trazabilidad
- Los checkpoints garantizan validacion incremental del sistema
- Los tests de propiedades validan invariantes universales con Hypothesis (minimo 100 iteraciones)
- Los tests unitarios validan ejemplos especificos y casos de borde
- Los tests de integracion con APIs reales se marcan con `@pytest.mark.integration` y requieren conectividad
- Los tests con hardware cuantico real se marcan con `@pytest.mark.quantum_hardware`
