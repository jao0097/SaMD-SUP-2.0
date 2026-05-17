# SaMD-SUP 2.0 — Sistema de Apoio à Decisão Clínica (Dr. Ajuda)

> **Retrieval-Augmented Generation (RAG) para profissionais de saúde** | 🏥 Clínico | 🤖 IA | 📚 Baseado em Evidência

---

## 📋 Visão Geral

**SaMD-SUP 2.0** é um sistema de **apoio à decisão clínica** que combina:
- **RAG (Retrieval-Augmented Generation)**: recupera fragmentos de conhecimento médico relevante de uma base de dados vetorial
- **LLM potente**: Groq API (modelos Mixtral/LLaMA 3) para gerar respostas contextadas
- **API segura**: autenticação Bearer token, rate limiting, conformidade LGPD
- **Interface web**: consultas reais ou via SSE (streaming)

O sistema processa **artigos científicos**, **transcriçõess de vídeos YouTube** e **documentação médica**, criando um índice vetorial otimizado para recuperação semântica.

> ⚠️ **Restricão Legal**: Uso exclusivo por **profissionais de saúde habilitados**. Não substitui diagnóstico clínico.

---

## 🎯 Caso de Uso

```
Médico/Enfermeiro/Farmacêutico → Faz pergunta clínica
                                ↓
                    Sistema RAG localiza evidências relevantes
                                ↓
                    LLM sintetiza resposta fundamentada
                                ↓
                    Retorna contexto + recomendação
```

**Exemplos:**
- "Qual é o protocolo atual para tratamento de pneumonia adquirida na comunidade?"
- "Contraindicações da metformina em pacientes com insuficiência renal?"
- "Efeitos colaterais e interações do atorvastatina?"

---

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────────────┐
│          FastAPI Web Server (web_api.py)        │
│  ├─ /health          → Liveness probe           │
│  ├─ /status          → Estatísticas da base     │
│  ├─ /consultar       → Q&A síncrono (JSON)     │
│  └─ /consultar/stream → Q&A streaming (SSE)     │
└─────────────────────────────────────────────────┘
           ↓ (run_in_executor + asyncio.Semaphore)
┌─────────────────────────────────────────────────┐
│         RAG Pipeline Core (super.py)            │
│  ├─ VectorDB (ChromaDB)                        │
│  ├─ Embeddings (SentenceTransformer)           │
│  ├─ Retrieval: semantic search (cosine sim.)   │
│  ├─ Ranking: classify relevância (LLM)         │
│  └─ Generation: síntese de resposta (Groq LLM) │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│    Indexing Scripts (utilitários CLI)           │
│  ├─ reindexar_pdfs.py → Ingestion de PDFs      │
│  ├─ limpar_transcricao_yt_paralelo.py          │
│  └─ atualizar_jsons_apos_limpeza.py            │
└─────────────────────────────────────────────────┘
```

### Fluxo de uma Consulta (Request → Response)

```
1. Cliente envia: POST /consultar { "pergunta": "..." } + Bearer token
2. FastAPI valida: autenticação, rate limit, tamanho da pergunta
3. Semáforo Groq: aguarda slot disponível (máx. 5 paralelos)
4. RAG Pipeline:
   a) Embedding da pergunta (SentenceTransformer)
   b) Busca vetorial em ChromaDB (top-k chunks)
   c) Classificação de relevância (LLM rápido)
   d) Prompt contextado + LLM potente (Groq)
5. Resposta Markdown retornada + metadados (modelo, latência)
6. Logging estruturado (sem conteúdo sensível — LGPD)
```

---

## 🚀 Início Rápido

### Pré-requisitos
- **Docker & Docker Compose** (recomendado)
- Ou: **Python 3.11+**, **CUDA/CPU PyTorch**, **ChromaDB**
- **Groq API key** (gratuita em https://console.groq.com)

### 1️⃣ Clone e Configure

```bash
git clone https://github.com/jao0097/SaMD-SUP-2.0.git
cd SaMD-SUP-2.0

# Copie o .env template
cp .env.example .env
```

### 2️⃣ Configure Variáveis de Ambiente

```bash
# .env
WEB_TOKEN=seu-token-super-secreto-aqui
GROQ_API_KEY=seu-groq-key-aqui
WEB_HOST=0.0.0.0
WEB_PORT=8000
LOG_FORMAT=text  # ou "json" para Datadog/observabilidade
SENTRY_DSN=      # opcional — para erro tracking
```

### 3️⃣ Inicie com Docker Compose

```bash
docker-compose up -d
```

O sistema:
- ✅ Baixa imagem Python + dependências
- ✅ Inicializa ChromaDB (vectorstore)
- ✅ Carrega modelo SentenceTransformer (primeira vez: ~500MB)
- ✅ Sobe API em `http://localhost:8000`

**Verifique saúde:**
```bash
curl http://localhost:8000/health
# {"status": "ok", "timestamp": "2026-05-17T22:00:00Z"}
```

### 4️⃣ Primeira Consulta

#### Via cURL (JSON)
```bash
curl -X POST http://localhost:8000/consultar \
  -H "Authorization: Bearer seu-token-super-secreto-aqui" \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Qual é o tratamento padrão para diabetes tipo 2?"}'
```

#### Via JavaScript (SSE Streaming)
```javascript
const eventSource = new EventSource(
  `/consultar/stream?pergunta=Qual é o tratamento para diabetes?&token=seu-token`
);

eventSource.addEventListener('message', (e) => {
  const data = JSON.parse(e.data);
  console.log(data.tipo, data.resposta || data.mensagem);
});
```

---

## 📊 Estrutura de Arquivos

```
SaMD-SUP-2.0/
├── web_api.py                      # 🌐 FastAPI app (main entry)
├── super.py                        # 🧠 RAG core (185KB — pipeline completo)
│
├── reindexar_pdfs.py              # 📄 Ingest PDFs → ChromaDB
├── limpar_transcricao_yt_paralelo.py # 🎬 Baixa + limpa transcrições
├── atualizar_jsons_apos_limpeza.py   # 🔄 Pós-processamento
├── test_api.py                     # ✅ Testes de integração
│
├── docker-compose.yml              # 🐳 Orquestração local
├── Dockerfile                      # 🐳 Build da imagem
├── requirements_web.txt            # 📦 Dependências Python
│
├── templates/                      # 🎨 HTML/CSS/JS
│   ├── index.html                  # Interface web (app)
│   └── landing.html                # Landing page pública
├── static/                         # 📦 Assets (CSS/JS)
│
└── README.md (este arquivo)
```

---

## 🔌 API Reference

### Endpoints Públicos

#### `GET /health`
Liveness probe (sem autenticação).
```json
{
  "status": "ok",
  "timestamp": "2026-05-17T22:00:00Z"
}
```

---

### Endpoints Autenticados

#### `GET /status`
Retorna estatísticas da base de conhecimento.

**Header required:** `Authorization: Bearer <token>`

```json
{
  "api_version": "1.0.0",
  "status": "online",
  "banco": {
    "pasta_chroma": "chroma_db",
    "chunks_totais": 1847,
    "urls_processadas": 42,
    "artigos": 38,
    "videos_youtube": 4
  },
  "modelos_groq": {
    "resposta": "mixtral-8x7b-32768",
    "classificacao": "llama2-7b-chat"
  },
  "timestamp": "2026-05-17T22:00:00Z"
}
```

---

#### `POST /consultar` (JSON)
Consulta síncrona. Aguarda resposta completa.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Body:**
```json
{
  "pergunta": "Quais são os critérios de diagnóstico para sepse?"
}
```

**Response:**
```json
{
  "resposta": "# Diagnóstico de Sepse\n\nSegundo a definição SIRS/qSOFA...",
  "modelo_usado": "mixtral-8x7b-32768",
  "tempo_resposta_s": 12.5
}
```

**Rate Limit:** 10 requisições por minuto por IP.

---

#### `GET /consultar/stream` (Server-Sent Events)
Streaming de resposta em tempo real.

**Query Parameters:**
```
?pergunta=<string>&token=<bearer_token>
```

**Resposta (SSE):**
```
data: {"tipo": "inicio"}
data: {"tipo": "resposta", "resposta": "...", "modelo_usado": "...", "tempo_resposta_s": 15.3}
data: {"tipo": "fim"}
```

---

### Endpoints de Interface

#### `GET /`
Landing page pública.

#### `GET /app`
Interface web (requer `templates/index.html`).

---

## 🔐 Segurança & Conformidade

### LGPD (Lei Geral de Proteção de Dados)

- ✅ **Nunca loga conteúdo de perguntas/respostas** — apenas metadados (IP, tamanho, latência)
- ✅ **Token timing-safe** — usa `secrets.compare_digest()` para evitar timing attacks
- ✅ **Sanitização Sentry** — Remove request bodies antes de enviar eventos de erro
- ✅ **ChromaDB local** — dados médicos ficam within customer infrastructure

### Autenticação
- Bearer token estático em `WEB_TOKEN`
- Headers: `Authorization: Bearer <token>`
- Sem token = 401 Unauthorized

### Rate Limiting
- 10 requisições/minuto por IP na endpoint `/consultar`
- Semáforo interno: máx. 5 chamadas paralelas ao Groq (evita rate limit da API)

### Observabilidade Segura
```python
# Sentry integrado (opcional)
SENTRY_DSN=https://xxxxx@sentry.io/123456
```

---

## 📦 Dependências Principais

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| **FastAPI** | 0.111.0 | Framework web assíncrono |
| **Uvicorn** | 0.30.1 | ASGI server |
| **ChromaDB** | 1.5.8 | Vector database (embeddings) |
| **SentenceTransformers** | 3.0.1 | Embedding model (multilingue) |
| **Groq** | 0.9.0 | LLM API client |
| **PyTorch** | 2.3.0 | ML backend (CPU) |
| **requests** | 2.31.0 | HTTP client |
| **beautifulsoup4** | 4.12.3 | Web scraping |
| **slowapi** | 0.1.9 | Rate limiting |

> **PyTorch CPU-only** para reduzir tamanho da imagem Docker (evita 1.5GB de CUDA).

---

## 🧠 RAG Pipeline (super.py)

### Fases

1. **Ingestion** (scripts CLI)
   - PDFs → OCR/texto
   - Transcripts YouTube → limpar formatação
   - Normalize + chunk (overlap semanticamente)

2. **Embedding**
   - SentenceTransformer: `all-MiniLM-L6-v2` ou `distiluse-base-multilingual-cased-v2`
   - Dimensionalidade: 384 ou 512
   - Armazenado em ChromaDB

3. **Retrieval**
   - Query embedding + cosine similarity search
   - Top-K chunks com score > threshold

4. **Ranking**
   - LLM rápido classifica relevância: "Muito Relevante" / "Relevante" / "Não Relevante"
   - Filtra ruído semanticamente

5. **Generation**
   - Prompt contextado (chunks + pergunta)
   - LLM potente sintetiza resposta
   - Markdown format com citations

### Parâmetros Tuneáveis

```python
# Em super.py
MODELO_POTENTE = "mixtral-8x7b-32768"  # Resposta (mais preciso)
MODELO_RAPIDO = "llama2-7b-chat"        # Classificação (rápido)
TOP_K_CHUNKS = 5                        # Quantos chunks recuperar
THRESHOLD_RELEVANCIA = 0.6              # Score mínimo
```

---

## 🛠️ Utilitários (Indexing Scripts)

### `reindexar_pdfs.py`
```bash
python reindexar_pdfs.py --pasta ./PDFs --output ./indexed
```
- Lê PDFs da pasta
- Extrai texto (OCR se necessário)
- Cria chunks com overlap
- Indexa em ChromaDB

### `limpar_transcricao_yt_paralelo.py`
```bash
python limpar_transcricao_yt_paralelo.py --urls urls.txt --workers 4
```
- Baixa transcripts de vídeos YouTube
- Remove timestamps, normaliza pontuação
- Salva em JSON estruturado

### `atualizar_jsons_apos_limpeza.py`
- Pós-processa JSONs após limpeza
- Valida schema, adiciona metadados

### `test_api.py`
```bash
python test_api.py --endpoint http://localhost:8000 --token seu-token
```
- Suite de testes de integração
- Valida endpoints, latência, formato de resposta

---

## 🐳 Docker

### Build Manual
```bash
docker build -t sup:latest .
docker run -p 8000:8000 \
  -e WEB_TOKEN=seu-token \
  -e GROQ_API_KEY=sua-key \
  sup:latest
```

### Docker Compose (recomendado)
```bash
docker-compose up -d       # Inicia background
docker-compose logs -f     # Ver logs em tempo real
docker-compose down        # Para containers
```

**Dockerfile highlights:**
- Base: `python:3.11-slim`
- PyTorch CPU (economiza espaço)
- Single-worker Uvicorn (ChromaDB não é thread-safe)
- Health check automático

---

## 📈 Performance & Escalabilidade

### Latência Típica
- **Retrieval + embedding**: ~2-3s
- **LLM rápido (ranking)**: ~1-2s
- **LLM potente (geração)**: ~8-15s
- **Total**: 11-20s por consulta

### Throughput
- **5 concurrent**: 1 consulta/4s (rate limit 10/min padrão)
- **Semáforo interno**: máx. 5 chamadas simultâneas ao Groq
- **Single worker**: recomendado para até 50-100 usuários/dia

### Escalabilidade Horizontal
Para produção em larga escala:
1. **ChromaDB → Qdrant/Pinecone** (cloud-native vector DB)
2. **FastAPI → múltiplos workers** (usar Nginx + multiple Uvicorn)
3. **LLM → batching** (queue Celery + Redis)

---

## 🚨 Troubleshooting

### "ChromaDB connection error"
```bash
# ChromaDB persistência local
# Garanta pasta chroma_db tem permissões
chmod -R 755 chroma_db/
```

### "GROQ_API_KEY not found"
```bash
# Verifique .env existe e está carregado
export GROQ_API_KEY="seu-key"
python -c "import os; print(os.getenv('GROQ_API_KEY'))"
```

### "Rate limit exceeded"
```
Semáforo Groq está full. Reduza conexões simultâneas ou aumente delay entre requisições.
Default: _groq_semaphore = asyncio.Semaphore(5)
```

### "Knowledge base empty"
```bash
# Indexe conteúdo primeiro
python reindexar_pdfs.py --pasta ./PDFs
```

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/sua-feature`
3. Faça commit: `git commit -am "Add feature"`
4. Push: `git push origin feature/sua-feature`
5. Abra um Pull Request

**Checklist antes de PR:**
- ✅ Rodar `test_api.py`
- ✅ Nenhum dado médico sensível em logs
- ✅ Documentar novas env vars em `.env.example`
- ✅ Update `requirements_web.txt` se adicionar deps

---

## 📝 Roadmap

- [ ] Integração com base de dados clínica EHR real
- [ ] Fine-tuning de modelo específico para português médico
- [ ] Dashboard admin (estatísticas, auditoria)
- [ ] Suporte multiusuário com controle de acesso (RBAC)
- [ ] Explicabilidade (mostrar chunks fonte da resposta)
- [ ] Cache distribuído (Redis) para consultas frequentes

---

## ⚖️ Licença

[Conteúdos públicos foram usados, e trabalho feito em parceria com Fábio Ortega, do canal Doutor Ajuda]

---

## 📞 Suporte

**Issues & bugs**: Abra uma issue no GitHub.
**Contato**: joaoaugusto009.7@gmail.com

---

## 🎓 Referências

- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/)
- [Groq API Reference](https://console.groq.com/docs)
- [SentenceTransformers](https://www.sbert.net/)
- [RAG Concepts](https://arxiv.org/abs/2307.09288)
- [LGPD Compliance](https://www.gov.br/cidadania/pt-br/acesso-a-informacao/lgpd)

---


