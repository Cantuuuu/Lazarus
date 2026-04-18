# Lazarus — Asistente Visual para Personas con Discapacidad Visual

![License](https://img.shields.io/badge/Licencia-MIT-green)

---

## Descripción

**Lazarus** es un asistente visual en tiempo real diseñado para personas con discapacidad visual. Usa la cámara de un dispositivo portátil para capturar el entorno, lo interpreta con un LLM multimodal y devuelve una descripción en audio, además de poder disparar acciones como alertar a contactos de emergencia o programar recordatorios.

El sistema puede activarse de tres formas:

- **Manualmente** mediante un atajo de teclado global o botón físico.
- **Automáticamente** cuando YOLOv8 detecta objetos relevantes en la escena (personas, vehículos, semáforos, puertas, etc.).
- **Por voz** pronunciando la palabra de activación "Lazarus" seguida de un comando.

El objetivo es que una persona ciega pueda llevar el dispositivo encima y recibir descripciones auditivas de su entorno sin necesidad de interactuar con una pantalla.

---

## Las dos rutas cliente

El repositorio contiene **dos implementaciones independientes y complementarias** del cliente, cada una con su propio backend y su propio modelo de cómputo. No dependen una de la otra: son dos caminos para el mismo producto.

```
┌──────────────────────────────┐              ┌──────────────────────────────┐
│   vision_assistant/          │              │   edge_computing_OPI5MX/     │
│   (prototipo PC)             │              │   (wearable edge)            │
│                              │              │                              │
│   Windows / Linux + webcam   │              │   Orange Pi 5 Max (NPU)      │
│                              │              │   + Raspberry Pi Zero W (cam)│
│   Captura + triggers         │              │   + auriculares Bluetooth    │
│   (YOLO/voz/manual)          │              │                              │
│              │               │              │   Pipeline completo local:   │
│              ▼               │              │   YOLO → NPU (RKNN)          │
│   POST a webhook n8n         │              │   Whisper STT local (ES)     │
│              │               │              │   Llamada directa Gemini     │
│              ▼               │              │   ElevenLabs con cache LRU   │
│   Agente n8n (backend) ──────┼─────┐        │   FastAPI + modos P/R        │
│   + MongoDB Atlas (memoria)  │     │        │                              │
│   + Twilio / WhatsApp (SOS)  │     │        │                              │
│              │               │     │        │                              │
│              ▼               │     │        │                              │
│   Respuesta → TTS            │     │        │                              │
└──────────────────────────────┘     │        └──────────────┬───────────────┘
                                     │                       │
                                     ▼                       ▼
                             ┌────────────────────────────────────────┐
                             │  Servicios cloud compartidos           │
                             │  · Gemini 3 Flash  (LLM + visión)      │
                             │  · ElevenLabs      (TTS)               │
                             └────────────────────────────────────────┘
```

| Carpeta | Plataforma | Dónde vive la inteligencia | Estado |
|---------|-----------|---------------------------|--------|
| **[vision_assistant/](vision_assistant/README.md)** | PC (Windows / Linux) con webcam | En un agente **n8n** en la nube que orquesta Gemini, memoria en Mongo Atlas y acciones (WhatsApp, llamadas). | Prototipo de iteración rápida |
| **[edge_computing_OPI5MX/](edge_computing_OPI5MX/README.md)** | Orange Pi 5 Max + Raspberry Pi Zero W + auriculares Bluetooth | En el propio dispositivo: NPU RKNN a 33 FPS, Whisper local, llamadas directas a Gemini / ElevenLabs. **No usa n8n.** | Wearable de producción |

La elección entre rutas depende del objetivo:

- **`vision_assistant`** es ideal para iterar sobre el comportamiento del agente: todo el prompt, memoria, herramientas y flujos de emergencia se editan visualmente en n8n sin tocar el cliente.
- **`edge_computing_OPI5MX`** es la ruta "real": autónoma, baja latencia, sin dependencia de un servidor intermedio más allá de las APIs de Gemini y ElevenLabs.

Cada carpeta tiene su propio README con instalación, configuración y uso.

---

## Servicios cloud compartidos

Ambas rutas consumen los mismos dos servicios externos de terceros:

| Servicio | Rol | Consumido por |
|----------|-----|--------------|
| **Gemini 3 Flash** (Google) | LLM multimodal con visión: interpreta la imagen más el comando del usuario y decide qué responder o qué acción tomar. | `vision_assistant` (a través de n8n) y `edge_computing_OPI5MX` (llamada directa desde `services/gemini_client.py`). |
| **ElevenLabs** | Síntesis de voz neuronal en español para convertir la respuesta textual del agente en audio natural. | `edge_computing_OPI5MX` en producción (con cache LRU para frases repetidas). En `vision_assistant` la reproducción local está en el roadmap. |

---

## Backend del prototipo PC: agente n8n (solo `vision_assistant`)

> **Nota:** esta sección aplica únicamente a la ruta `vision_assistant`. La ruta edge no la necesita.

Para que el prototipo de escritorio pueda iterar rápido sobre el comportamiento del agente sin redeployar código Python, toda la lógica conversacional vive en una instancia propia de **n8n** desplegada en la nube.

### Topología

```
┌──────────────────────────────────────────────────────────────────┐
│                        Google Cloud (VM)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                        Coolify                             │  │
│  │   (PaaS self-hosted: gestiona contenedores, dominios,      │  │
│  │    SSL automático, backups y despliegues declarativos)     │  │
│  │                                                            │  │
│  │   ┌──────────────────────────────────────────────────┐     │  │
│  │   │                     n8n                          │     │  │
│  │   │   Workflows en n8n_workflows/ :                  │     │  │
│  │   │   · lazarus_assistant   (agente principal)       │     │  │
│  │   │   · send_message_to_me       (WhatsApp al user)  │     │  │
│  │   │   · send_message_to_emergency (contactos SOS)    │     │  │
│  │   │   · call_emergency_contact    (llamada Twilio)   │     │  │
│  │   │   · create_reminder / reminer_cron               │     │  │
│  │   └──────────────────┬───────────────────────────────┘     │  │
│  └──────────────────────┼─────────────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────────────┘
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
        Gemini 3      MongoDB      Twilio + Evolution API
        Flash         Atlas        (llamadas + WhatsApp)
```

### Por qué cada pieza

| Pieza | Qué es | Para qué se usa |
|-------|--------|-----------------|
| **Google Cloud VM** | Máquina virtual de propósito general en GCP. | Host con IP fija y dominio propio para exponer los webhooks sin depender de un túnel local. |
| **Coolify** | Plataforma open-source tipo Heroku/Vercel que se auto-hospeda (alternativa a AWS Amplify o Railway). Corre sobre Docker y gestiona certificados Let's Encrypt, variables de entorno, despliegue por git y reverse proxy con Traefik. | Montar n8n como servicio con HTTPS y respaldos, sin tener que escribir `docker-compose` a mano ni mantener nginx/certbot. |
| **n8n** | Orquestador visual de workflows (low-code). Cada workflow es un grafo de nodos HTTP, LLM, base de datos, mensajería, etc. | Recibe el POST del cliente Python, decide qué hacer con un agente LLM, llama a subflows y responde con el texto final. |
| **MongoDB Atlas** | Base de datos documental administrada. | Memoria conversacional del agente (`MongoDB Chat Memory` → `lazarusdb`, ventana de 15 mensajes). Al ser Atlas, vive fuera de la VM y se conecta por URI con TLS. |
| **Twilio + Evolution API** | Telefonía programable y gateway de WhatsApp. | Subflows de emergencia: `call_emergency_contact` llama por teléfono, `send_message_to_emergency` / `send_message_to_me` mandan WhatsApp con texto e imágenes. |

### Flujo de una petición

1. El cliente `vision_assistant` empaqueta `input` (texto/transcripción), `id` (user_id) e `imgs` (frames en base64) y hace `POST` al webhook de n8n.
2. El nodo **Webhook** pasa la request al **AI Agent** (Gemini 3 Flash) junto con el historial recuperado de **MongoDB Atlas**.
3. El agente decide la respuesta: si hay peligro dispara los subflows de emergencia (Twilio/WhatsApp); si solo es una pregunta, devuelve `response_text`.
4. El **Structured Output Parser** fuerza la salida a un JSON `{actions, response_text}` válido.
5. La respuesta viaja de vuelta al cliente, que (en el roadmap) la entregará a ElevenLabs para reproducirla como audio.

---

## Puesta en marcha

### Ruta PC (`vision_assistant` + backend n8n)

1. **Desplegar el backend** (una sola vez): provisionar la VM en GCP, instalar Coolify, levantar n8n desde Coolify, crear el cluster en MongoDB Atlas e importar los workflows de `n8n_workflows/` con sus credenciales (Gemini, Twilio, Evolution API).
2. **Correr el cliente:** seguir el [README de `vision_assistant/`](vision_assistant/README.md).

### Ruta edge (`edge_computing_OPI5MX`)

1. Flashear la Orange Pi 5 Max con Ubuntu ARM64 y la Raspberry Pi Zero W con el stream MJPEG.
2. Convertir el modelo YOLOv8n a RKNN (desde un host x86 vía Docker) y copiarlo a la Orange Pi.
3. Configurar `.env` con `GEMINI_API_KEY`, `ELEVENLABS_*` y `PI_ZERO_STREAM_URL`.
4. Seguir el [README de `edge_computing_OPI5MX/`](edge_computing_OPI5MX/README.md) para el resto (Bluetooth, FastAPI, tests).

---

## Roadmap general

- [x] Cliente PC con triggers manual, YOLO y voz ([vision_assistant](vision_assistant/README.md))
- [x] Backend n8n con agente conversacional, memoria por usuario y acciones de emergencia (WhatsApp, llamada)
- [x] Cliente edge sobre Orange Pi 5 Max con NPU RKNN, modos proactivo y reactivo, STT local y TTS con cache ([edge_computing_OPI5MX](edge_computing_OPI5MX/README.md))
- [ ] Integración n8n en tiempo real dentro de `vision_assistant` (hoy solo se usa en `process_video.py`)
- [ ] Respuesta en audio con ElevenLabs TTS reproducida desde `vision_assistant`
- [ ] Ruta edge opcionalmente enrutada al mismo agente n8n cuando haya conexión, con fallback local

---

## Licencia

MIT.
