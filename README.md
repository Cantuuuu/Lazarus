# Lazarus — Asistente Visual para Personas con Discapacidad Visual

![License](https://img.shields.io/badge/Licencia-MIT-green)

---

## Descripción

**Lazarus** es un asistente visual en tiempo real diseñado para personas con discapacidad visual. Usa la cámara de un dispositivo portátil para capturar el entorno, lo interpreta con un LLM multimodal y devuelve una descripción en audio, además de poder disparar acciones como alertar a contactos de emergencia o programar recordatorios.

El sistema puede activarse de tres formas:

- **Manualmente** mediante un atajo de teclado global.
- **Automáticamente** cuando YOLOv8 detecta objetos relevantes en la escena (personas, vehículos, semáforos, puertas, etc.).
- **Por voz** pronunciando la palabra de activación "Lazarus" seguida de un comando.

El objetivo es que una persona ciega pueda llevar el dispositivo encima y recibir descripciones auditivas de su entorno sin necesidad de interactuar con una pantalla.

---

## Estructura del repositorio

El repositorio contiene **dos implementaciones cliente complementarias** más la definición de la infraestructura en la nube:

```
Lazarus/
├── vision_assistant/         # Cliente Python de escritorio (prototipo sobre PC)
├── edge_computing_OPI5MX/    # Wearable edge sobre Orange Pi 5 Max + Raspberry Pi Zero W
└── n8n_workflows/            # Workflows del agente, memoria y acciones (usados por vision_assistant)
```

### Las dos rutas de cliente

| Carpeta | Plataforma | Modelo de cómputo | Estado |
|---------|-----------|-------------------|--------|
| **[vision_assistant/](vision_assistant/README.md)** | Windows / Linux (desarrollo) | Triggers locales (YOLO + voz) → webhook → **agente n8n en la nube** → respuesta | Prototipo funcional |
| **[edge_computing_OPI5MX/](edge_computing_OPI5MX/README.md)** | Orange Pi 5 Max (NPU RK3588) + Raspberry Pi Zero W como cámara MJPEG + auriculares Bluetooth | Todo en local: YOLO en NPU, STT Whisper local, llamadas directas a Gemini / ElevenLabs, modos proactivo y reactivo con FastAPI | Wearable de producción |

Son dos caminos que comparten la misma idea de producto pero difieren en *dónde* vive la inteligencia:

- **`vision_assistant`** externaliza toda la lógica conversacional a **n8n** (ver sección siguiente). El cliente solo captura y envía. Útil para iterar rápido en el agente sin tocar código.
- **`edge_computing_OPI5MX`** ejecuta el pipeline completo en el dispositivo, llama a Gemini/ElevenLabs por API y **no usa n8n**. Está optimizado para latencia y autonomía del dispositivo (alertas a 33 FPS con la NPU, sin depender de un workflow externo).

---

## Infraestructura y servicios externos

Esta sección aplica a **`vision_assistant/`**. La ruta edge (`edge_computing_OPI5MX/`) consume Gemini y ElevenLabs directamente por su API y no depende de la VM ni de n8n.

El cliente Python de escritorio únicamente captura, detecta y empaqueta el contexto visual y de voz. Toda la inteligencia conversacional, la memoria y las acciones sobre el mundo exterior (enviar WhatsApp, llamar por teléfono, programar recordatorios) se ejecutan en una instancia propia de **n8n** alojada en la nube.

### Topología de despliegue

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
│  │   │   (orquestador de workflows — expone webhooks    │     │  │
│  │   │    HTTP que el cliente Python consume)           │     │  │
│  │   │                                                  │     │  │
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
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   Gemini 3 Flash     MongoDB Atlas      ElevenLabs
   (LLM + visión)     (memoria chat)     (TTS, en curso)
```

### ¿Por qué cada pieza?

| Pieza | Qué es | Para qué se usa aquí |
|-------|--------|----------------------|
| **Google Cloud VM** | Máquina virtual de propósito general en GCP | Host de largo plazo con IP fija y dominio propio para exponer los webhooks sin depender de un túnel local. |
| **Coolify** | Plataforma open-source tipo Heroku/Vercel que se auto-hospeda (alternativa a AWS Amplify o Railway). Corre sobre Docker y gestiona certificados Let's Encrypt, variables de entorno, despliegue por git y reverse proxy con Traefik. | Montar n8n como servicio con HTTPS y respaldos, sin tener que escribir `docker-compose` a mano ni mantener nginx/certbot. |
| **n8n** | Orquestador visual de workflows (low-code). Cada workflow es un grafo de nodos HTTP, LLM, base de datos, mensajería, etc. | Recibe el POST del cliente Python con la imagen y el comando, decide qué hacer con un agente LLM, llama a subflows para mensajería/emergencias, y responde con el texto final. |
| **Gemini 3 Flash** (Google) | LLM multimodal con visión | Cerebro del workflow `lazarus_assistant`: interpreta la imagen + comando de voz, decide si hay riesgo y qué acción tomar (`take_photo`, `send_emergency_message`, …) y redacta la `response_text`. También actúa como parser estructurado para forzar la salida JSON. También es el LLM que llama directamente `edge_computing_OPI5MX` para la descripción de escena en modo reactivo. |
| **MongoDB Atlas** | Base de datos documental administrada en la nube | Memoria conversacional del agente (nodo `MongoDB Chat Memory` → cluster Atlas, base `lazarusdb`, ventana de 15 mensajes). Persiste el historial por `user_id` entre peticiones para que Lazarus recuerde lo que el usuario ya preguntó o lo que ya describió. Al ser Atlas, vive fuera de la VM y se conecta por URI con TLS. |
| **ElevenLabs** | Síntesis de voz neuronal en español | Convertir la `response_text` en audio natural. En `vision_assistant` la integración está en desarrollo; en `edge_computing_OPI5MX` ya está en producción con cache LRU para frases repetidas. |
| **Twilio + Evolution API** | Telefonía programable y gateway de WhatsApp | Subflows de emergencia: `call_emergency_contact` llama por teléfono, `send_message_to_emergency` / `send_message_to_me` mandan WhatsApp con texto e imágenes. |

### Flujo end-to-end de una petición (vision_assistant)

1. El cliente Python empaqueta `input` (texto/transcripción), `id` (user_id para la memoria) e `imgs` (frames en base64) y hace `POST` al webhook de n8n.
2. El nodo **Webhook** recibe la request y la pasa al **AI Agent** (Gemini 3 Flash) junto con el historial recuperado de **MongoDB Atlas**.
3. El agente decide la respuesta: si hay peligro dispara los subflows de emergencia (Twilio/WhatsApp); si el usuario solo pregunta, devuelve `response_text`.
4. El **Structured Output Parser** (otra instancia de Gemini) garantiza que la salida sea un JSON `{actions, response_text}` válido.
5. La respuesta viaja de vuelta al cliente, que (en el roadmap) la entregará a **ElevenLabs** para reproducirla como audio.

---

## Puesta en marcha

Dependiendo de la ruta, la puesta en marcha es distinta:

### Ruta PC (vision_assistant + n8n)

1. **Desplegar la infraestructura** (una sola vez): provisionar la VM en GCP, instalar Coolify, levantar n8n desde Coolify, crear el cluster en MongoDB Atlas e importar los workflows de `n8n_workflows/` en n8n con sus credenciales correspondientes (Gemini, ElevenLabs, Twilio, Evolution API).
2. **Correr el cliente:** seguir el [README de `vision_assistant/`](vision_assistant/README.md) para instalación, configuración y uso.

### Ruta edge (edge_computing_OPI5MX)

1. Flashear la Orange Pi 5 Max con Ubuntu ARM64 y la Raspberry Pi Zero W con el stream MJPEG.
2. Convertir el modelo YOLOv8n a RKNN (desde un host x86 vía Docker) y copiarlo a la Orange Pi.
3. Configurar `.env` con `GEMINI_API_KEY`, `ELEVENLABS_*` y `PI_ZERO_STREAM_URL`.
4. Seguir el [README de `edge_computing_OPI5MX/`](edge_computing_OPI5MX/README.md) para el resto (Bluetooth, FastAPI, tests).

---

## Roadmap general

- [x] Infraestructura n8n + Coolify + MongoDB Atlas desplegada y accesible vía webhook
- [x] Agente conversacional (`lazarus_assistant`) con memoria por usuario y acciones de emergencia (WhatsApp, llamada)
- [x] Cliente PC con triggers manual, YOLO y voz ([vision_assistant](vision_assistant/README.md))
- [x] Cliente edge sobre Orange Pi 5 Max con NPU RKNN, modos proactivo y reactivo, STT/TTS local+cloud ([edge_computing_OPI5MX](edge_computing_OPI5MX/README.md))
- [ ] Integración n8n en tiempo real dentro de `vision_assistant` (hoy solo se usa en `process_video.py`)
- [ ] Respuesta en audio con ElevenLabs TTS reproducida desde `vision_assistant`
- [ ] Unificar el agente: que la ruta edge también consuma el webhook de n8n cuando haya conexión, manteniendo fallback local

---

## Licencia

MIT.
