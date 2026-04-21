# evaluacion-final-llm-chatbots-agentes
Evaluación final del curso de postgrado LLMs, chatbots y agentes


Este repositorio sirve como **punto de partida común** para la evaluación del curso.

Todos los estudiantes deben:

1. Hacer **fork** de este repositorio
2. Elegir **UNO de los tres proyectos propuestos**
3. Implementar una aplicación funcional basada en **Ollama + LangGraph + Streamlit**

---

# 🎯 Objetivo

Construir una aplicación basada en **agentes LLM** que:

* Utilice al menos **2-3 agentes con roles diferenciados**
* Implemente **orquestación con LangGraph**
* Tenga una **interfaz funcional en Streamlit**

---

# 🧩 Proyectos disponibles (elige uno)

## 1. Asistente Académico

* Planificación de estudio
* Explicación de conceptos
* Generación de resúmenes

## 2. Sistema de Soporte Interno de una empresa

* Respuesta a preguntas frecuentes sobre la empresa
* Uso de base de conocimiento de la empresa
* Router de servicios

## 3. Planificador viajes

* Reserva de viaje
* Conocimiento del destino
* Itinerario

---

**Los elementos planteados son solo tips, pueden modificarse o extenderse**


# 📁 Estructura del repositorio

```
.
├── app.py 
├── env.example
│
├── agents/
├── prompts/
├── tools/
├── utils/
├── data/
│
└── README.md
```

---

# 📂 Descripción de carpetas

## `app.py`

Punto de entrada de la aplicación.

* Implementa la interfaz con Streamlit
* Captura input del usuario
* Llama al grafo principal

---

## `.env.example`

Configuración global:

* Constantes del sistema
* Cambiar nombre a solo .env y tratar de utilizarlo y no poner constantes en el código

---

## `agents/`

Contiene los agentes del sistema.

Cada archivo representa un agente con una responsabilidad clara.

Ejemplos:

* `router.py`
* `planner.py`
* `tutor.py`
* `validator.py`

Cada agente debería:

* Tener un prompt definido
* Recibir estado
* Devolver salida estructurada

---

## `prompts/`

Prompts separados del código.

Ventajas:

* Facilita iteración
* Mejora claridad
* Permite evaluar diseño de prompting

Ejemplo:

* `router.txt`
* `planner.txt`

---

## `tools/`

Herramientas externas usadas por agentes.

Ejemplos:

* Retriever (búsqueda en documentos)
* APIs simuladas
* Funciones auxiliares

---


## `utils/`

Código auxiliar reutilizable.

Ejemplos:

* Cliente de Ollama
* Parsear texto
* Funciones auxiliares

---

## `data/`

Datos usados por la aplicación.

Ejemplos:

* Base de conocimiento
* Documentos (PDF, txt)
* APIs externas

**Pueden inventarse o generarse mediante algún modelo de lenguaje**

---


# ⚙️ Estado actual del repositorio

Este repositorio incluye:

* Estructura de carpetas
* Orientación en el README

⚠️ **NO es una solución completa**

Su trabajo es:

* Diseñar múltiples agentes
* Extender el sistema
* Implementar orquestación 

---

# 📌 Requisitos mínimos

Tu proyecto debe incluir:

* Al menos **2 agentes** 
* Uso de **LangGraph**
* Separación clara de responsabilidades entre agentes
* Interfaz funcional en Streamlit (no tiene que quedar muy bonita, pero si funcionar)
* Código estructurado (Tratar de respetar la estructura de carpetas)
* Documentación. Comentar todo lo posible el código

---


# 🧠 ENTREGA

Debes entregar:

* Repositorio funcional
* README actualizado con:

  * Caso de uso elegido
  * Arquitectura
  * Decisiones de diseño

---
