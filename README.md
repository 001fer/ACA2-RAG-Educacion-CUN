# Sistema RAG de Orientación Académica y Permanencia Estudiantil (ACA 2 - CUN)

Proyecto desarrollado para la asignatura **Computación en la Nube** (Especialización en Inteligencia Artificial - CUN). Implementa un sistema **Retrieval-Augmented Generation (RAG)** de extremo a extremo sobre el pilar de **Educación**.

## Arquitectura Tecnológica
* **Entorno Cloud:** Google Colab
* **Base de Datos Vectorial:** ChromaDB (recuperación semántica Top-2)
* **Modelo de Lenguaje (LLM):** `gemma-4-26b-a4b-it` vía Google GenAI SDK (`temperature=0.2` para mitigación de alucinaciones)
* **Interfaz Gráfica (UI):** Gradio (Módulo de Chat con historial + Módulo de Gestión Documental/Vectorial)

## Ejecución Local o en la Nube
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
