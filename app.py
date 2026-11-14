import os
import time
import chromadb
import gradio as gr
import pandas as pd
from google import genai
from google.genai import types

# 1. Gestión segura de credenciales mediante variable de entorno
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Error: Configura la variable de entorno GEMINI_API_KEY antes de ejecutar.")

client = genai.Client(api_key=api_key)
MODELO_LLM = "gemma-4-26b-a4b-it"

# 2. Base Vectorial ChromaDB y Corpus Institucional CUN (Pilar: Educación)
chroma_client = chromadb.Client()
coleccion = chroma_client.get_or_create_collection(name="corpus_educacion_cun")

documentos_cun = [
    "Artículo 10 - Calificaciones: Para aprobar una asignatura de pregrado o posgrado en la CUN, la nota mínima definitiva debe ser de 3.0 sobre una escala de 0.0 a 5.0.",
    "Artículo 24 - Asistencia y Permanencia: Si un estudiante acumula inasistencias superiores al 20% del total de horas programadas en una asignatura sin excusa médica o laboral debidamente validada, perderá la materia por fallas con nota de 0.0.",
    "Artículo 35 - Homologación Externa: La solicitud de homologación de estudios previos debe radicarse ante Registro y Control mínimo 15 días hábiles antes del inicio del periodo académico, adjuntando certificados de notas y contenidos programáticos.",
    "Guía de Bienestar y Permanencia (Alerta Temprana): Todo estudiante con promedio acumulado inferior a 3.2 o con más de 2 asignaturas reprobadas entra automáticamente en ruta de acompañamiento psicopedagógico y tutorías académicas gratuitas para prevenir la deserción.",
    "Política de Grados y Opción de Grado: Para optar al título en la Especialización en Inteligencia Artificial, el estudiante debe haber aprobado el 100% de los créditos académicos, estar a paz y salvo financiero y aprobar el proyecto aplicado o seminario de profundización.",
    "Reglamento de Cancelación de Semestre: El aplazamiento o cancelación voluntaria del periodo académico sin afectación del promedio solo puede solicitarse hasta la semana 6 del calendario académico vigente."
]

metadatos_cun = [
    {"fuente": "Reglamento Estudiantil CUN - Art. 10", "categoria": "Evaluación", "anio": 2026},
    {"fuente": "Reglamento Estudiantil CUN - Art. 24", "categoria": "Asistencia", "anio": 2026},
    {"fuente": "Manual de Admisiones - Art. 35", "categoria": "Homologación", "anio": 2026},
    {"fuente": "Guía Bienestar Institucional - Permanencia", "categoria": "Retención Estudiantil", "anio": 2026},
    {"fuente": "Lineamientos de Grado Posgrados CUN", "categoria": "Graduación", "anio": 2026},
    {"fuente": "Reglamento Estudiantil CUN - Calendario", "categoria": "Trámites", "anio": 2026}
]

ids_cun = [f"doc_{i+1}" for i in range(len(documentos_cun))]
coleccion.upsert(documents=documentos_cun, metadatas=metadatos_cun, ids=ids_cun)

# 3. Pipeline RAG
def consultar_chatbot_rag(pregunta, top_k=2):
    inicio = time.time()
    resultados = coleccion.query(query_texts=[pregunta], n_results=top_k)
    fragmentos = resultados['documents'][0]
    fuentes = [m['fuente'] for m in resultados['metadatas'][0]]
    contexto = "\n- ".join(fragmentos)
    
    prompt = f"""Eres el Asistente Virtual Institucional de la Corporación Unificada Nacional (CUN).
Responde la consulta del estudiante usando ÚNICAMENTE la información del contexto proporcionado.
Reglas estrictas:
1. No inventes datos ni uses conocimiento externo.
2. Si la respuesta no está en el contexto, responde exactamente: "No encuentro información sobre ese tema en los documentos institucionales cargados."
3. Al final de tu respuesta, incluye una sección llamada "Fuentes consultadas:" citando los documentos utilizados.

Contexto documental:
- {contexto}

Fuentes disponibles: {fuentes}

Pregunta del estudiante: {pregunta}"""

    respuesta = client.models.generate_content(
        model=MODELO_LLM,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    latencia = round(time.time() - inicio, 2)
    return respuesta.text, fuentes, latencia

def responder_chat_ui(mensaje, historial):
    respuesta_texto, _, latencia = consultar_chatbot_rag(mensaje, top_k=2)
    return f"{respuesta_texto}\n\n---\n⏱️ *Latencia: {latencia}s* | 📚 *Top-2 ChromaDB*"

def ver_base_vectorial():
    datos = coleccion.get()
    return pd.DataFrame([
        {
            "ID Vector": datos['ids'][i],
            "Fuente": datos['metadatas'][i].get('fuente', 'N/A'),
            "Categoría": datos['metadatas'][i].get('categoria', 'N/A'),
            "Fragmento Indexado": datos['documents'][i]
        }
        for i in range(len(datos['ids']))
    ])

def agregar_documento_ui(nuevo_texto, nombre_fuente, categoria):
    if not nuevo_texto.strip() or not nombre_fuente.strip():
        return "⚠️ Completa el texto y el nombre de la fuente.", ver_base_vectorial()
    nuevo_id = f"doc_{coleccion.count() + 1}"
    coleccion.upsert(
        documents=[nuevo_texto.strip()],
        metadatas=[{"fuente": nombre_fuente.strip(), "categoria": categoria.strip() or "General", "anio": 2026}],
        ids=[nuevo_id]
    )
    return f"✅ Indexado '{nombre_fuente}' ({nuevo_id}).", ver_base_vectorial()

with gr.Blocks(title="Asistente RAG Institucional - CUN") as demo:
    gr.Markdown("# 🎓 Sistema RAG de Orientación Académica y Permanencia - CUN")
    with gr.Tabs():
        with gr.TabItem("💬 Módulo de Chat RAG"):
            gr.ChatInterface(
                fn=responder_chat_ui,
                examples=[
                    "¿Cuál es la nota mínima para aprobar una materia?",
                    "¿Qué pasa si falto a más del 20% de las clases?",
                    "¿Hasta qué semana puedo cancelar el semestre sin afectar mi promedio?"
                ]
            )
        with gr.TabItem("🗂️ Gestión Documental y Base Vectorial"):
            btn_refrescar = gr.Button("🔄 Actualizar tabla de vectores")
            tabla_docs = gr.Dataframe(value=ver_base_vectorial, interactive=False, wrap=True)
            btn_refrescar.click(fn=ver_base_vectorial, outputs=tabla_docs)
            txt_fuente = gr.Textbox(label="Nombre de la Fuente")
            txt_cat = gr.Textbox(label="Categoría")
            txt_contenido = gr.Textbox(label="Contenido del Artículo", lines=3)
            btn_agregar = gr.Button("📥 Vectorizar e Indexar", variant="primary")
            estado_carga = gr.Textbox(label="Estado", interactive=False)
            btn_agregar.click(fn=agregar_documento_ui, inputs=[txt_contenido, txt_fuente, txt_cat], outputs=[estado_carga, tabla_docs])

if __name__ == "__main__":
    demo.launch()
