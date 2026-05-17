"""
sup — Sistema de Apoio Clínico Dr. Ajuda
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistema RAG de apoio à decisão clínica para médicos, baseado no conhecimento
extraído do canal e site do Dr. Ajuda.

Permite que médicos consultem a base de conhecimento para:
  • Raciocínio diagnóstico e diagnóstico diferencial
  • Condutas clínicas e escolha de tratamentos
  • Interpretação de exames e achados
  • Formulação e revisão de laudos médicos
  • Consulta de protocolos e conceitos clínicos

Fontes suportadas:
  • Sites / blogs   → crawl automático de listagem, raspa artigos
  • YouTube         → transcrição automática de vídeos (1 ou vários)
  • Arquivos locais → processa pares .txt + .json já salvos em disco

⚠️  AVISO: Este sistema é de uso exclusivo de profissionais de saúde habilitados.
    As respostas não substituem o julgamento clínico do médico.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALAÇÃO
  pip install requests beautifulsoup4 chromadb groq sentence-transformers youtube-transcript-api python-dotenv
  export GROQ_API_KEY='sua_chave'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — CONSULTAR (uso principal para médicos)

  Consulta clínica interativa (menu):
    sup

  Consulta clínica contínua (várias perguntas na mesma sessão):
    sup --pergunta "critérios de síndrome metabólica?"
    sup --pergunta                    # abre sessão interativa
    # Na sessão: linha vazia ou 'sair' encerra · 'nova' limpa o histórico

  Consulta direta (uma pergunta e encerra):
    sup --pergunta "..." --once

  Ver status e estatísticas:
    sup --status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — INDEXAR (fase de configuração — banco já completo em produção)

  Crawl de site (descobre e indexa todos os artigos):
    sup --crawl "https://site.com/blog"
    sup --crawl "https://site.com/blog" --filtro "/posts/"
    sup --crawl "https://site.com/blog" --sem-paginacao

  Artigos (uma ou várias URLs):
    sup --artigos "https://a.com/p1 https://a.com/p2"
    sup --artigos lista.txt

  Vídeo(s) do YouTube:
    sup --video "https://youtube.com/watch?v=ID"
    sup --video "url1" "url2" "url3"
    sup --video lista.txt

  Arquivos locais (.txt + .json gerados pelo sistema):
    sup --local

  Preparar .txt órfãos (sem .json): detecta texto cru vs. já formatado, limpa, metadados
    Groq como nas URLs, grava par .txt + .json padrão; depois indexe com sup --local
    sup --gerar-json-locais
    sup --gerar-json-locais --sobrescrever   # regenerar JSON/.txt já existentes

  Relatório da pasta (inventário: orphans, pendentes, indexados):
    sup --relatorio

  Localizar arquivo local por nome / tema:
    sup --buscar-local

  Forçar reindexação de uma URL já processada:
    sup --reindexar "https://..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIÁVEIS DE AMBIENTE (opcionais)
  RAG_PASTA_SAIDA          pasta de saída dos arquivos
  RAG_PASTA_CHROMA         pasta do banco vetorial
  RAG_DEBUG_LOG            caminho para log de debug (desativado se vazio)
  GROQ_MODELO_RAPIDO       modelo para metadados/classificação  (padrão: llama-3.1-8b-instant)
  GROQ_MODELO_POTENTE      modelo para respostas finais         (padrão: llama-3.3-70b-versatile)
  GROQ_MODELO_LIMPEZA      modelo para limpeza de transcrição   (padrão: llama-3.1-8b-instant)
  CHROMA_CHUNK_PALAVRAS    tamanho do chunk em palavras         (padrão: 500)
  CHROMA_CHUNK_OVERLAP     sobreposição entre chunks            (padrão: 50)
  LIMPEZA_TRANSCRICAO      local | ia | nenhuma                 (padrão: local)
  GROQ_LIMPEZA_MAX_CHARS   máx. chars por bloco de limpeza IA  (padrão: 7500)
  GROQ_LIMPEZA_PAUSA_S     pausa entre blocos de limpeza        (padrão: 1.25)
  GROQ_METADADOS_MAX_CHARS máx. chars enviados para metadados   (padrão: 4000)
  TIMEOUT_REQUISICAO       timeout HTTP em segundos             (padrão: 15)
  PAUSA_ENTRE_PAGINAS      pausa entre requisições web          (padrão: 1.0)
  GROQ_MAX_RETRIES         tentativas em caso de rate-limit     (padrão: 10)
  RAG_ARTIGOS_FETCH_THREADS  downloads paralelos no lote de artigos (padrão: 1; ex.: 4)
  RAG_HTML_PARSER          lxml | html.parser | html5lib       (padrão: lxml se instalado)
  RAG_MIN_CHARS_ARTIGO     mín. de caracteres no texto raspado (padrão: 0 = aceita curtos)
  RAG_YT_PAUSA_ENTRE_VIDEOS  pausa base entre cada vídeo no lote (padrão: 2.5s)
  RAG_YT_PAUSA_JITTER        atraso aleatório extra 0..N s (padrão: 1.5)
  RAG_YT_APOS_TRANSCRICAO    pausa entre transcrição e oEmbed (padrão: 0.5s)
  RAG_YT_PRE_FETCH           pausa antes do 1º pedido de transcrição por vídeo (padrão: 0.3s)
  RAG_YT_FETCH_RETRIES       tentativas se a transcrição falhar (padrão: 5)
  RAG_YT_FETCH_BACKOFF       base do backoff exponencial entre tentativas (padrão: 2.0s)
  RAG_YT_STOP_ON_IPBLOCK     para o lote ao detectar IpBlocked (padrão: 1)
  RAG_YT_MAX_CONSEC_IPBLOCK  quantos IpBlocked seguidos para parar (padrão: 2)
  RAG_YT_IPBLOCK_COOLDOWN_S  sugestão de espera antes de tentar de novo (padrão: 3600)
  RAG_YT_PROXY               proxy HTTP(S) para youtube-transcript-api (ex.: http://host:porta)
  RAG_YT_DLP_ENABLED         habilita yt-dlp como fallback após bloqueio (padrão: 0)
  RAG_YT_DLP_PRIMARY         usa yt-dlp como método PRIMÁRIO, api como fallback (padrão: 0)
  RAG_YT_DLP_COOKIES_FILE    caminho opcional para cookies.txt do yt-dlp
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import concurrent.futures
import difflib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import OrderedDict
from datetime import date
from typing import Dict, List, Optional, Set, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS GROQ (centralizados)
# ══════════════════════════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVISO LEGAL — SISTEMA DE APOIO À DECISÃO CLÍNICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este sistema é destinado EXCLUSIVAMENTE a profissionais de saúde habilitados.
Todas as respostas são geradas com base no conhecimento extraído do canal e
site do Dr. Ajuda e têm finalidade de APOIO à decisão — não substituem o
julgamento clínico, o exame físico do paciente nem as diretrizes oficiais das
sociedades médicas. O profissional de saúde é inteiramente responsável pela
decisão diagnóstica e terapêutica final.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Aviso médico anexado ao final de toda resposta gerada ao médico
AVISO_MEDICO = (
    "\n\n---\n"
    "⚠️  **Aviso de apoio clínico**: Esta resposta é produzida por um sistema de suporte à decisão "
    "baseado no conteúdo do Dr. Ajuda e destina-se exclusivamente a profissionais de saúde habilitados. "
    "Não substitui o julgamento clínico, o exame físico do paciente nem as diretrizes das sociedades "
    "médicas. O médico assistente é o único responsável pela conduta diagnóstica e terapêutica final."
)

PROMPT_METADADOS_SISTEMA = (
    "Você é um indexador clínico especializado em preparar conteúdo médico para recuperação semântica em RAG.\n"
    "Sua saída alimenta diretamente o ChromaDB — metadados precisos aumentam a recuperabilidade dos chunks.\n\n"
    "Raciocine em ordem antes de preencher cada campo:\n"
    "1. Leia o conteúdo identificando a condição/tema central e as entidades clínicas (fármacos, CIDs, procedimentos).\n"
    "2. Extraia apenas o que está explícito — nunca infira diagnósticos, doses ou afirmações não presentes.\n"
    "3. Preserve terminologia médica exata; prefira nomes genéricos de fármacos.\n"
    "4. Se um campo não puder ser preenchido com segurança: \"\" ou [].\n"
    "5. Retorne APENAS JSON válido, sem markdown, sem texto fora do objeto.\n"
)

PROMPT_METADADOS_USUARIO_HEADER = (
    "Retorne JSON com estas chaves (sem extras):\n"
    "{\n"
    '  "resumo": string,            // 1–3 frases clínicas fiéis; sem extrapolação\n'
    '  "palavras_chave": string[],  // 3–8 termos médicos do texto\n'
    '  "tema_principal": string,    // 1 frase: "Diagnóstico e tratamento de X"\n'
    '  "topicos_abordados": string[], // 3–8 tópicos clínicos reais\n'
    '  "condicoes_clinicas": string[], // patologias/síndromes citadas ([] se nenhuma)\n'
    '  "medicamentos": string[],    // nomes genéricos dos fármacos ([] se nenhum)\n'
    '  "procedimentos": string[],   // exames, cirurgias, escalas ([] se nenhum)\n'
    '  "especialidade": string,     // área médica principal\n'
    '  "nivel_evidencia": "opinião de especialista"|"revisão narrativa"|"estudo clínico"|"diretriz"|"outro",\n'
    '  "nivel_tecnico": "iniciante"|"intermediário"|"avançado",\n'
    '  "linguagem": "pt"|"en"|"outro"\n'
    "}\n"
)

PROMPT_LIMPEZA_SISTEMA = (
    "Você é um editor de transcrições médicas preparando texto para indexação semântica em RAG.\n"
    "Meta: maximizar a recuperabilidade clínica dos chunks — termos médicos precisos e frases completas aumentam hits relevantes.\n\n"
    "PRESERVAR (prioridade máxima):\n"
    "- Toda terminologia médica, nomes de fármacos, doses, valores laboratoriais, CIDs, escalas e índices.\n"
    "- Números sempre com seu contexto ('5 mg/kg/dia', não apenas '5').\n"
    "- Nomes próprios, siglas médicas e acrônimos.\n\n"
    "CORRIGIR:\n"
    "- Pontuação e capitalização óbvias.\n"
    "- Repetições de vício de fala claramente redundantes ('é é é', 'então então').\n"
    "- Marcações de ruído: [música], [aplausos], [risadas], [inaudível] e similares.\n"
    "- Inserir parágrafo a cada mudança clara de tema clínico.\n\n"
    "PROIBIDO:\n"
    "- Resumir, omitir ou condensar qualquer trecho.\n"
    "- Adicionar informações, interpretações ou diagnósticos não presentes.\n\n"
    "Retorne APENAS o texto limpo, sem comentários, cabeçalhos ou markdown.\n"
    "Se indicado 'Trecho N de M', edite apenas esse trecho."
)

PROMPT_CLASSIFICADOR_SISTEMA = (
    "Você é o roteador de busca de um sistema RAG clínico. Sua decisão determina quais chunks serão recuperados.\n\n"
    "Raciocine nesta ordem:\n"
    "1. ESPECIFICA: algum título corresponde diretamente à pergunta? → tipo='especifica', fonte_alvo=título exato.\n"
    "2. COMPARATIVA: compara ≥2 condições, tratamentos ou fármacos? → tipo='comparativa'.\n"
    "3. FORA_DE_ESCOPO: SOMENTE se for claramente não-médico (receita, clima, TI, etc.). "
    "Dúvida vaga, linguagem leiga ou sintomas genéricos → NÃO é fora_de_escopo; use 'geral'.\n"
    "4. Caso contrário → tipo='geral'.\n\n"
    "n_chunks: especifica=10 | geral=12 | comparativa=16 | fora_de_escopo=0.\n"
    "Linguagem leiga ou pergunta ampla → prefira 'geral', fonte_alvo=null, n_chunks=10.\n"
    "Na dúvida entre especifica e geral → geral.\n"
    "Retorne APENAS JSON válido.\n"
)

PROMPT_EXPANDIR_BUSCA_SISTEMA = (
    "Gere consultas de busca vetorial em base clínica. NÃO responda à pergunta.\n"
    "Retorne JSON: {\"queries\": [string]} com 1 ou 2 buscas PRÓXIMAS ao tema original.\n"
    "Reformule em termos clínicos prováveis. Sem inventar condições não sugeridas na pergunta.\n"
)

PROMPT_ROUTER_CONVERSA_SISTEMA = (
    "Você é o roteador de intenções de um CLI de RAG clínico (Dr. Ajuda). Retorne APENAS JSON válido, sem markdown.\n\n"
    "AÇÕES disponíveis → parâmetros obrigatórios:\n"
    "consultar          → pergunta_clinica: string\n"
    "artigos_lote       → entradas: string[]  (URLs http/https não-YouTube ou caminho .txt)\n"
    "videos_lote        → entradas: string[]  (URLs/IDs YouTube ou caminho .txt)\n"
    "crawl_site         → url_listagem: string, filtro_path: string|null, sem_paginacao: boolean\n"
    "processar_local    → (sem parâmetros)\n"
    "buscar_local       → consulta: string|null\n"
    "reindexar          → url: string\n"
    "status             → (sem parâmetros)\n"
    "ajuda              → (sem parâmetros)\n"
    "sair               → (sem parâmetros)\n"
    "perguntar_clarificacao → pergunta: string\n\n"
    "PRIORIDADE de decisão (avalie nesta ordem):\n"
    "1. Contém dúvida/caso clínico → consultar\n"
    "2. Contém URL YouTube ou ID de vídeo → videos_lote\n"
    "3. Contém URL http(s) não-YouTube → artigos_lote (exceto se pedir 'crawl'/'varrer'/'descobrir links' → crawl_site)\n"
    "4. Pede 'reindexar' + URL → reindexar\n"
    "5. Pede 'processar' arquivos locais → processar_local\n"
    "6. Pede 'buscar'/'achar' arquivo local → buscar_local\n"
    "7. Pede status/estatísticas → status\n"
    "8. Parâmetro essencial ausente → perguntar_clarificacao (nunca invente URLs)\n"
)

PROMPT_RESPOSTA_DIRETA_SISTEMA = (
    "Consultor clínico (Dr. Ajuda). Responda usando SOMENTE os trechos numerados.\n"
    "PROIBIDO: inferir, 'podemos inferir', conhecimento geral ou completar com suposição.\n"
    "Se o trecho traz critérios numéricos ou passos diagnósticos, LISTE-OS em bullets.\n\n"
    "FORMATO\n"
    "1. 1º parágrafo: resposta direta à pergunta.\n"
    "2. Bullets com cada critério/dose/valor exato do trecho + *(Fonte N)*.\n"
    "3. Só declare 'não consta na base' para pontos realmente ausentes dos trechos.\n"
    "4. Não misture tema de outro artigo.\n\n"
    "Obrigatório: 📚 Fontes Consultadas — [N] Título — URL\n"
)

PROMPT_RESPOSTA_SISTEMA = PROMPT_RESPOSTA_DIRETA_SISTEMA

PROMPT_RESPOSTA_SISTEMA_ENXUTO = (
    "Consultor clínico. ZERO alucinação: só trechos fornecidos.\n"
    "Sem cobertura → 'Não localizado' + como reformular. Cite (Fonte N).\n"
    "🔍 Análise | 🧠 Síntese | 🔎 Lacunas | 📚 Fontes\n"
)

PROMPT_RESPOSTA_CONVERSA = (
    "Sintetize APENAS os trechos indexados. Sem conhecimento externo.\n"
    "Responda diretamente à pergunta no 1º parágrafo; detalhes em bullets se houver critérios/doses.\n"
    "Cada dado clínico com (Fonte N). Fonte 1 = mais relevante. "
    "Não misture tema de outro artigo. Se não constar: diga claramente.\n"
    "Prosa breve, sem emojis. Referências: títulos ao final.\n"
)

_INSTRUCAO_SINTESE_TRECHOS = (
    "Tarefa: SINTETIZAR somente os trechos abaixo. Não use conhecimento externo.\n"
    "Checklist: (1) Cada dado clínico tem (Fonte N)? (2) Nada foi inventado? (3) Lacunas declaradas?\n"
)

PROMPT_EXTRAIR_FATOS_SISTEMA = (
    "Extraia fatos dos trechos que respondem à PERGUNTA do médico. NÃO redija a resposta final.\n"
    "Retorne APENAS JSON válido:\n"
    '{"fatos":[{"texto":string,"fonte":number,"trecho_literal":string,"relevancia":"alta"|"media"}],'
    '"lacunas":[string],"cobre_pergunta":"sim"|"parcial"|"nao"}\n\n'
    "PRIORIDADE\n"
    "1. Fonte cujo TÍTULO corresponde ao tema da pergunta → extraia TODOS os critérios/condutas/dados pedidos.\n"
    "2. Outras fontes → só fatos que respondem diretamente à pergunta (ignore listas genéricas de outra doença).\n"
    "3. Critérios numéricos, doses, escalas, passos de conduta: copie valores exatos do trecho.\n"
    "4. Proibido inferir ou usar conhecimento externo.\n"
    "5. Se os trechos não cobrem a pergunta: fatos=[], cobre_pergunta=nao, lacunas=[o que faltou].\n"
)

PROMPT_SINTETIZAR_FATOS_SISTEMA = (
    "Redija resposta clínica útil para o médico, APENAS com os fatos do JSON.\n"
    "Comece respondendo diretamente à pergunta (1º parágrafo).\n"
    "Use bullets para critérios, doses, exames ou passos de conduta.\n"
    "Cada dado clínico com (Fonte N). Fatos relevancia=media só se necessários.\n"
    "Se cobre_pergunta=nao: declare o que não consta na base e o que seria preciso indexar.\n"
)

from urllib.parse import urljoin, urlparse, urlencode
from urllib.request import urlopen

# ── Dependências de terceiros ──────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


class RespostaCache:
    def __init__(self, maxsize=200):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def _hash(self, p):
        # Versão do prompt — invalida cache antigo após mudanças de política RAG.
        norm = re.sub(r'[^\w\s]', '', p.lower())
        return hashlib.md5(f"rag-v8-qualidade|{norm}".encode()).hexdigest()

    def get(self, p):
        k = self._hash(p)
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        return None

    def set(self, p, r):
        k = self._hash(p)
        self.cache[k] = r
        self.cache.move_to_end(k)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


cache_respostas = RespostaCache()

# ── .env (CLI): path ao lado deste arquivo, depois CWD ───────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv

    load_dotenv(_env_path)
    load_dotenv()  # opcional: .env no diretório de trabalho atual
except ImportError:
    pass  # pip install python-dotenv para carregar .env no CLI; web_api já depende disso

# Debug NDJSON — ativo apenas se RAG_DEBUG_LOG estiver definido como caminho válido
DEBUG_LOG_PATH = os.getenv("RAG_DEBUG_LOG", "").strip()
DEBUG_SESSION_ID = "sup"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not DEBUG_LOG_PATH:
        return
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES  ← edite aqui ou via variáveis de ambiente
# ══════════════════════════════════════════════════════════════════════════════

# ── Suporte a múltiplas chaves Groq com rotação automática ────────────────────
# Opção 1 — variável única com vírgulas: GROQ_API_KEYS=chave1,chave2,chave3
# Opção 2 — variáveis numeradas:         GROQ_API_KEY=chave1  GROQ_API_KEY_2=chave2 ...
# As duas formas podem ser combinadas; duplicatas são removidas.
def _carregar_groq_keys() -> List[str]:
    """Lê todas as chaves Groq configuradas e retorna lista sem duplicatas."""
    chaves: List[str] = []
    # Opção 1: GROQ_API_KEYS separadas por vírgula
    multi = os.getenv("GROQ_API_KEYS", "").strip()
    if multi:
        chaves.extend(k.strip() for k in multi.split(",") if k.strip())
    # Opção 2: GROQ_API_KEY + GROQ_API_KEY_2, GROQ_API_KEY_3, …
    for sufixo in ["", "_2", "_3", "_4", "_5"]:
        k = os.getenv(f"GROQ_API_KEY{sufixo}", "").strip()
        if k and k not in chaves:
            chaves.append(k)
    # Remove placeholder e duplicatas mantendo ordem
    validas = [k for k in chaves if k and k != "sua_chave_groq_aqui"]
    seen: List[str] = []
    for k in validas:
        if k not in seen:
            seen.append(k)
    return seen

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "sua_chave_groq_aqui")  # mantido p/ compat.
_GROQ_KEYS: List[str] = _carregar_groq_keys()

_PASTA_PADRAO       = os.path.expanduser("~/Documentos/rag_base")
PASTA_SAIDA         = os.getenv("RAG_PASTA_SAIDA",  _PASTA_PADRAO)
PASTA_CHROMA        = os.getenv("RAG_PASTA_CHROMA", os.path.join(PASTA_SAIDA, "chroma_db"))
ARQUIVO_PROCESSADAS = os.path.join(PASTA_SAIDA, ".urls_processadas.json")

# Modelos Groq — 3 papéis distintos para controle de custo/velocidade
MODELO_RAPIDO  = os.getenv("GROQ_MODELO_RAPIDO",  "llama-3.1-8b-instant")   # metadados, classificação
MODELO_POTENTE = os.getenv("GROQ_MODELO_POTENTE", "llama-3.3-70b-versatile") # resposta final
MODELO_LIMPEZA = os.getenv("GROQ_MODELO_LIMPEZA", "llama-3.1-8b-instant")   # limpeza em lotes

# Chunking
CHUNK_PALAVRAS      = int(os.getenv("CHROMA_CHUNK_PALAVRAS", "500"))
CHUNK_OVERLAP       = int(os.getenv("CHROMA_CHUNK_OVERLAP",  "50"))

# Limpeza de transcrição: "local" (0 tokens) | "ia" (Groq) | "nenhuma" (bruto)
LIMPEZA_TRANSCRICAO = os.getenv("LIMPEZA_TRANSCRICAO", "local").strip().lower()
LIMPEZA_MAX_CHARS   = int(os.getenv("GROQ_LIMPEZA_MAX_CHARS", "7500"))   # max chars por bloco IA
LIMPEZA_PAUSA_S     = float(os.getenv("GROQ_LIMPEZA_PAUSA_S", "1.25"))   # pausa entre blocos

# Metadados
METADADOS_MAX_CHARS = int(os.getenv("GROQ_METADADOS_MAX_CHARS", "4000"))

# Economia de tokens (consulta)
# - GROQ_CLASSIF_TITULOS_MAX limita títulos no classificador (sempre, via busca vetorial).
# - GROQ_ECONOMIA reduz ainda mais contexto e chunks na resposta final.
GROQ_ECONOMIA = os.getenv("GROQ_ECONOMIA", "0").strip().lower() in ("1", "true", "yes", "on")
GROQ_CLASSIF_TITULOS_MAX = max(0, int(os.getenv("GROQ_CLASSIF_TITULOS_MAX", "30")))
GROQ_CONTEXTO_CHUNK_MAX_CHARS = max(0, int(os.getenv("GROQ_CONTEXTO_CHUNK_MAX_CHARS", "2200")))
GROQ_CONTEXTO_TOTAL_MAX_CHARS = max(0, int(os.getenv("GROQ_CONTEXTO_TOTAL_MAX_CHARS", "14000")))
GROQ_MAX_N_CHUNKS = max(0, int(os.getenv("GROQ_MAX_N_CHUNKS", "5")))  # trechos na resposta (pós-rerank)
# Pool vetorial antes do reranking local (maior → melhor seleção dos top-k finais)
RAG_CHUNKS_BUSCA_POR_TIPO = {
    "especifica": max(6, int(os.getenv("RAG_CHUNKS_BUSCA_ESPECIFICA", "14"))),
    "geral": max(6, int(os.getenv("RAG_CHUNKS_BUSCA_GERAL", "18"))),
    "comparativa": max(8, int(os.getenv("RAG_CHUNKS_BUSCA_COMPARATIVA", "22"))),
    "fora_de_escopo": 0,
}
# Classificação: heurística local (rápida) ou Groq (mais lenta)
RAG_CLASSIFICAR_GROQ = os.getenv("RAG_CLASSIFICAR_GROQ", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
# Geração: auto=direto se retrieval forte, senão 2 etapas | direto | 2etapas
RAG_MODO_RESPOSTA = os.getenv("RAG_MODO_RESPOSTA", "auto").strip().lower()
RAG_2_ETAPAS_LEGACY = os.getenv("RAG_2_ETAPAS", "0").strip().lower() in ("1", "true", "yes", "on")
# Anexa chunks vizinhos (mesmo documento) aos 2 melhores resultados
RAG_VIZINHOS_CHUNK = os.getenv("RAG_VIZINHOS_CHUNK", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
RAG_VIZINHOS_RAIO = max(0, min(2, int(os.getenv("RAG_VIZINHOS_RAIO", "1"))))
# Exige overlap no título (não só metadados) para manter fontes secundárias
RAG_FILTRO_TITULO_ESTRITO = os.getenv("RAG_FILTRO_TITULO_ESTRITO", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# Expansão local (várias queries, sem Groq) quando o 1º resultado vetorial é fraco
RAG_BUSCA_EXPANDIR_LOCAL = os.getenv("RAG_BUSCA_EXPANDIR_LOCAL", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# Expansão extra via Groq (consultas reformuladas) — opcional, custo de API
RAG_EXPANDIR_BUSCA = os.getenv("RAG_EXPANDIR_BUSCA", "0").strip().lower() in ("1", "true", "yes", "on")
RAG_EXPANDIR_MAX_QUERIES = max(1, min(4, int(os.getenv("RAG_EXPANDIR_MAX_QUERIES", "3"))))
# Fallback lexical nos metadados clínicos indexados (sem varrer a coleção inteira)
RAG_METADATA_FALLBACK = os.getenv("RAG_METADATA_FALLBACK", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
RAG_METADATA_SCAN_MAX = max(100, int(os.getenv("RAG_METADATA_SCAN_MAX", "600")))
# Máximo de trechos do mesmo documento na resposta final (diversidade de fontes)
RAG_MAX_CHUNKS_POR_FONTE = max(1, int(os.getenv("RAG_MAX_CHUNKS_POR_FONTE", "2")))
# Descarta trechos muito distantes do melhor match vetorial (cosine distance Chroma)
RAG_DISTANCIA_RATIO = max(1.05, float(os.getenv("RAG_DISTANCIA_RATIO", "1.18")))
# Limiar de distância: abaixo = match forte; acima = busca fraca (dispara expansão)
RAG_DISTANCIA_BOA = max(0.5, float(os.getenv("RAG_DISTANCIA_BOA", "1.12")))
# Descarta trechos cujo título/tema não combina com a pergunta (ex.: selênio em pergunta de síndrome metabólica)
RAG_FILTRO_TEMA = os.getenv("RAG_FILTRO_TEMA", "1").strip().lower() in ("1", "true", "yes", "on")
RAG_TEMA_MIN_OVERLAP = max(1, int(os.getenv("RAG_TEMA_MIN_OVERLAP", "1")))
# Limite de caracteres por trecho enviado ao LLM
RAG_CHUNK_LLM_MAX_CHARS = max(600, int(os.getenv("RAG_CHUNK_LLM_MAX_CHARS", "2200")))

# Requisições web
TIMEOUT           = int(os.getenv("TIMEOUT_REQUISICAO", "15"))
PAUSA_ENTRE_REQS  = float(os.getenv("PAUSA_ENTRE_PAGINAS", "1.0"))
GROQ_MAX_RETRIES  = int(os.getenv("GROQ_MAX_RETRIES", "10"))

# YouTube: reduzir bloqueio por excesso de requisições
# Tempos reduzidos — ajuste para cima em lotes grandes ou IPs residenciais lentos.
RAG_YT_PAUSA_ENTRE_VIDEOS = float(os.getenv("RAG_YT_PAUSA_ENTRE_VIDEOS", "2.5"))   # era 4.0
RAG_YT_PAUSA_JITTER       = float(os.getenv("RAG_YT_PAUSA_JITTER", "1.5"))          # era 2.0
RAG_YT_APOS_TRANSCRICAO   = float(os.getenv("RAG_YT_APOS_TRANSCRICAO", "0.5"))      # era 0.8
RAG_YT_PRE_FETCH          = float(os.getenv("RAG_YT_PRE_FETCH", "0.3"))             # era 0.5
RAG_YT_FETCH_RETRIES      = max(1, int(os.getenv("RAG_YT_FETCH_RETRIES", "5")))
RAG_YT_FETCH_BACKOFF      = float(os.getenv("RAG_YT_FETCH_BACKOFF", "2.0"))         # era 2.5

# Se bater IpBlocked/RequestBlocked, é bloqueio anti-bot (não é "falta de legenda").
RAG_YT_STOP_ON_IPBLOCK        = os.getenv("RAG_YT_STOP_ON_IPBLOCK", "1").strip().lower() not in ("0", "false", "no")
RAG_YT_IPBLOCK_COOLDOWN_S     = float(os.getenv("RAG_YT_IPBLOCK_COOLDOWN_S", "3600"))  # 1h padrão
RAG_YT_MAX_CONSEC_IPBLOCK     = max(1, int(os.getenv("RAG_YT_MAX_CONSEC_IPBLOCK", "2")))  # era 1

# Proxy HTTP(S) opcional para requisições à YouTube Transcript API
# Formato: http://usuario:senha@host:porta  ou  http://host:porta
RAG_YT_PROXY = os.getenv("RAG_YT_PROXY", "").strip()

# yt-dlp: fallback (0) ou método primário (1) de obtenção de legendas
# PRIMARY=1 → tenta yt-dlp antes da youtube-transcript-api; mais resiliente, mais lento.
RAG_YT_DLP_ENABLED  = os.getenv("RAG_YT_DLP_ENABLED",  "0").strip().lower() in ("1", "true", "yes")
RAG_YT_DLP_PRIMARY  = os.getenv("RAG_YT_DLP_PRIMARY",  "0").strip().lower() in ("1", "true", "yes")
RAG_YT_DLP_COOKIES_FILE = os.getenv("RAG_YT_DLP_COOKIES_FILE", "").strip()  # caminho para cookies.txt (opcional)

# Download paralelo só na fase HTTP+parse do lote de artigos (1 = sequencial, como antes)
RAG_ARTIGOS_FETCH_THREADS = max(1, min(32, int(os.getenv("RAG_ARTIGOS_FETCH_THREADS", "1"))))

# Mínimo de caracteres no corpo do artigo após raspagem (0 = só rejeita vazio; antigo era ~150)
RAG_MIN_CHARS_ARTIGO = max(0, int(os.getenv("RAG_MIN_CHARS_ARTIGO", "0")))


def _pick_bs_parser() -> str:
    """lxml costuma ser bem mais rápido que html.parser; fallback automático."""
    override = os.getenv("RAG_HTML_PARSER", "").strip().lower()
    if override in ("lxml", "html.parser", "html5lib"):
        return override
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


BS_PARSER = _pick_bs_parser()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

# Seletores de conteúdo principal (ordem de prioridade)
SELETORES_CONTEUDO = [
    "article", "main",
    "[class*='post-content']", "[class*='article-content']",
    "[class*='entry-content']", "[class*='article-body']",
    "[class*='post-body']",    "[class*='content']",
    "div#content", "div#main",
]

# Elementos removidos antes de extrair o texto
SELETORES_LIXO = [
    "nav", "header", "footer", "aside", "form",
    "script", "style", "noscript", "iframe",
    "[class*='sidebar']",    "[class*='menu']",       "[class*='social']",
    "[class*='share']",      "[class*='comment']",    "[class*='related']",
    "[class*='newsletter']", "[class*='popup']",      "[class*='cookie']",
    "[class*='banner']",     "[class*='ad-']",        "[class*='ads']",
]


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO E INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

GROQ_ENABLED = bool(_GROQ_KEYS)
if not GROQ_ENABLED:
    print(
        "⚠️  Nenhuma GROQ_API_KEY configurada — recursos de IA ficam indisponíveis.\n"
        "   Para habilitar: defina GROQ_API_KEY (e opcionalmente GROQ_API_KEY_2) no .env\n",
        file=sys.stderr,
    )
else:
    _n = len(_GROQ_KEYS)
    print(
        f"✅  Groq: {_n} chave{'s' if _n > 1 else ''} carregada{'s' if _n > 1 else ''} "
        f"({'rotação automática ativa' if _n > 1 else 'chave única'}).",
        file=sys.stderr,
    )

# groq_client mantido para compatibilidade de imports externos
groq_client = Groq(api_key=_GROQ_KEYS[0]) if GROQ_ENABLED else None


import threading as _threading


class _GroqKeyPool:
    """
    Pool thread-safe de clientes Groq com rotação automática em rate-limit.

    Comportamento:
    - Enquanto houver chaves não tentadas no round atual: rotaciona imediatamente (sem wait).
    - Quando todas as chaves do round falharam com 429/503: espera e recomeça.
    - Repete por GROQ_MAX_RETRIES rounds completos antes de desistir.
    """

    def __init__(self, keys: List[str]) -> None:
        self._clients: List[Groq] = [Groq(api_key=k) for k in keys]
        self._idx: int = 0
        self._lock = _threading.Lock()

    def _rotate(self) -> None:
        with self._lock:
            self._idx = (self._idx + 1) % len(self._clients)

    def create(self, **kwargs):
        n = len(self._clients)
        ultimo: Optional[BaseException] = None
        round_count = 0

        while round_count < GROQ_MAX_RETRIES:
            keys_tried = 0
            while keys_tried < n:
                try:
                    return self._clients[self._idx].chat.completions.create(**kwargs)
                except Exception as e:
                    ultimo = e
                    if not _groq_retryavel(e):
                        raise
                    keys_tried += 1
                    if keys_tried < n:
                        ant = self._idx + 1
                        self._rotate()
                        print(
                            f"   🔄  Rate-limit chave {ant}/{n} → chave {self._idx + 1}/{n}",
                            file=sys.stderr,
                        )
            # Todas as chaves falharam neste round
            round_count += 1
            if round_count >= GROQ_MAX_RETRIES:
                break
            assert ultimo is not None
            w = _retry_wait(ultimo, round_count)
            print(
                f"   ⏸️  Todas as {n} chave(s) em rate-limit — "
                f"aguardando {w:.1f}s (round {round_count}/{GROQ_MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(w)
            self._rotate()

        assert ultimo is not None
        raise ultimo


_groq_pool: Optional[_GroqKeyPool] = _GroqKeyPool(_GROQ_KEYS) if GROQ_ENABLED else None

http: Optional[requests.Session] = None
embedding_fn = None
chroma = None
colecao = None


def _ensure_db() -> None:
    """Inicializa recursos pesados apenas quando necessário (DB/embeddings/HTTP)."""
    global http, embedding_fn, chroma, colecao
    if colecao is not None and chroma is not None and embedding_fn is not None and http is not None:
        return

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    os.makedirs(PASTA_CHROMA, exist_ok=True)

    if http is None:
        http = requests.Session()
        http.headers.update(HEADERS)

    if embedding_fn is None:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    if chroma is None:
        _chroma_host = os.getenv("CHROMA_HOST")
        _chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        if _chroma_host:
            chroma = chromadb.HttpClient(host=_chroma_host, port=_chroma_port)
        else:
            chroma = chromadb.PersistentClient(path=PASTA_CHROMA)

    if colecao is None:
        colecao = chroma.get_or_create_collection(
            name="base_conhecimento",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICAÇÃO DE URLs
# ══════════════════════════════════════════════════════════════════════════════

def carregar_processadas() -> Set[str]:
    _ensure_db()
    if os.path.exists(ARQUIVO_PROCESSADAS):
        with open(ARQUIVO_PROCESSADAS, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def marcar_processada(url: str, processadas: Set[str]) -> None:
    processadas.add(url)
    with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
        json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ — chamada com retry automático e backoff exponencial
# ══════════════════════════════════════════════════════════════════════════════

def _retry_wait(erro: BaseException, tentativa: int) -> float:
    """Extrai o tempo de espera da mensagem da API ou usa backoff exponencial."""
    m = re.search(r"try again in ([\d.]+)\s*s", str(erro), re.I)
    if m:
        return float(m.group(1)) + 0.75
    ra = getattr(getattr(erro, "response", None), "headers", {}).get("retry-after")
    try:
        return float(ra) + 0.75
    except (TypeError, ValueError):
        pass
    return min(120.0, 2.5 * (1.65 ** tentativa))


def _groq_retryavel(erro: BaseException) -> bool:
    status = getattr(erro, "status_code", None)
    if status in (429, 503):
        return True
    if status == 413:
        body = str(erro).lower()
        return "rate_limit" in body or "tokens per minute" in body or "tpm" in body
    return False


def chamar_groq(
    sistema: str,
    usuario: str,
    modelo: str = MODELO_RAPIDO,
    json_mode: bool = False,
    temperatura: float = 0.1,
) -> str:
    """
    Chama a API Groq com rotação automática de chaves em rate-limit.

    Com múltiplas chaves (GROQ_API_KEY + GROQ_API_KEY_2, ...):
      • 429 numa chave → troca para a próxima imediatamente (sem esperar).
      • Todas as chaves em 429 → aguarda backoff exponencial e recomeça.
    """
    if not GROQ_ENABLED or _groq_pool is None:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. Defina a variável de ambiente GROQ_API_KEY para usar recursos de IA."
        )
    kwargs: dict = {
        "model": modelo,
        "temperature": temperatura,
        "messages": [
            {"role": "system", "content": sistema},
            {"role": "user",   "content": usuario},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return _groq_pool.create(**kwargs).choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS GERAIS
# ══════════════════════════════════════════════════════════════════════════════

def slugify(texto: str, limite: int = 60) -> str:
    s = unicodedata.normalize("NFKD", texto)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", s.lower()))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:limite] or "item"


def chunkar(texto: str) -> List[str]:
    """Divide em chunks de palavras com sobreposição configurável."""
    t = (texto or "").strip()
    if not t:
        return []
    palavras = t.split()
    step = max(1, CHUNK_PALAVRAS - CHUNK_OVERLAP)
    out = [
        " ".join(palavras[i : i + CHUNK_PALAVRAS])
        for i in range(0, len(palavras), step)
        if palavras[i : i + CHUNK_PALAVRAS]
    ]
    if not out and t:
        return [t]
    return out


def gerar_metadados(titulo: str, conteudo: str, tipo: str) -> dict:
    """
    Gera metadados semânticos enriquecidos via Groq.
    Funciona para artigos web e transcrições de vídeo.
    """
    raw = chamar_groq(
        sistema=PROMPT_METADADOS_SISTEMA,
        usuario=(
            f"{PROMPT_METADADOS_USUARIO_HEADER}\n\n"
            f"Tipo: {tipo}\n"
            f"Título: {titulo}\n\n"
            "Conteúdo (trecho):\n"
            f"{conteudo[:METADADOS_MAX_CHARS]}"
        ),
        json_mode=True,
    )
    fallback = {
        "resumo": conteudo[:200],
        "palavras_chave": [],
        "tema_principal": "",
        "topicos_abordados": [],
        "condicoes_clinicas": [],
        "medicamentos": [],
        "procedimentos": [],
        "especialidade": "",
        "nivel_evidencia": "outro",
        "nivel_tecnico": "intermediário",
        "linguagem": "pt",
    }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    return parsed


def indexar_conteudo(
    conteudo: str,
    titulo: str,
    url: str,
    tipo: str,
    meta_extra: dict,
    meta_ia: Optional[dict] = None,
) -> int:
    """
    Divide em chunks e faz upsert no ChromaDB.
    Armazena chunk_index, total_chunks e todos os campos de meta_ia.
    Retorna o número de chunks indexados.
    """
    _ensure_db()
    chunks = chunkar(conteudo)
    if not chunks:
        return 0

    slug  = slugify(url)[:80]
    total = len(chunks)
    ids   = [f"{slug}_c{i}" for i in range(total)]

    # Campos IA achatados para o ChromaDB (apenas strings)
    ia_fields: dict = {}
    if meta_ia:
        ia_fields["resumo"]              = str(meta_ia.get("resumo", ""))[:500]
        ia_fields["palavras_chave"]      = ", ".join(meta_ia.get("palavras_chave", []))
        ia_fields["tema_principal"]      = str(meta_ia.get("tema_principal", ""))
        ia_fields["topicos_abordados"]   = ", ".join(meta_ia.get("topicos_abordados", []))
        ia_fields["nivel_tecnico"]       = str(meta_ia.get("nivel_tecnico", ""))
        ia_fields["linguagem"]           = str(meta_ia.get("linguagem", "pt"))
        # Campos clínicos (Dr. Ajuda)
        ia_fields["condicoes_clinicas"]  = ", ".join(meta_ia.get("condicoes_clinicas", []))
        ia_fields["medicamentos"]        = ", ".join(meta_ia.get("medicamentos", []))
        ia_fields["procedimentos"]       = ", ".join(meta_ia.get("procedimentos", []))
        ia_fields["especialidade"]       = str(meta_ia.get("especialidade", ""))
        ia_fields["nivel_evidencia"]     = str(meta_ia.get("nivel_evidencia", ""))

    metas = [
        {
            "titulo":       titulo,
            "url":          url,
            "tipo":         tipo,
            "data":         date.today().isoformat(),
            "chunk_index":  i,
            "total_chunks": total,
            **{k: str(v) for k, v in meta_extra.items()},
            **ia_fields,
        }
        for i in range(total)
    ]

    colecao.upsert(ids=ids, documents=chunks, metadatas=metas)
    return total


def _texto_txt_cabecalho_dr_ajuda(
    tipo: str,
    titulo: str,
    url: str,
    data_coleta: str,
    resumo: str,
    conteudo: str,
) -> str:
    """Mesmo layout de arquivo salvo por ``salvar_arquivos`` (URLs remotas)."""
    return (
        f"TIPO: {tipo}\n"
        f"TÍTULO: {titulo}\n"
        f"URL: {url}\n"
        f"DATA: {data_coleta}\n"
        f"RESUMO: {resumo}\n"
        f"[CONTEÚDO]\n"
        f"{conteudo}"
    )


def _dict_json_sidecar_dr_ajuda(
    tipo: str,
    titulo: str,
    url: str,
    data_coleta: str,
    meta_ia: dict,
    conteudo: str,
    extra: dict,
    *,
    indexado_chroma: bool,
) -> dict:
    """Documento JSON ao lado do .txt — mesmo esquema de ``salvar_arquivos``."""
    return {
        "tipo":              tipo,
        "titulo":            titulo,
        "url":               url,
        "data_coleta":       data_coleta,
        "resumo":            meta_ia.get("resumo", ""),
        "palavras_chave":    meta_ia.get("palavras_chave", []),
        "tema_principal":    meta_ia.get("tema_principal", ""),
        "topicos_abordados": meta_ia.get("topicos_abordados", []),
        "nivel_tecnico":     meta_ia.get("nivel_tecnico", ""),
        "linguagem":         meta_ia.get("linguagem", "pt"),
        "condicoes_clinicas": meta_ia.get("condicoes_clinicas", []),
        "medicamentos":       meta_ia.get("medicamentos", []),
        "procedimentos":      meta_ia.get("procedimentos", []),
        "especialidade":      meta_ia.get("especialidade", ""),
        "nivel_evidencia":    meta_ia.get("nivel_evidencia", ""),
        "tamanho_chars":      len(conteudo),
        "indexado_chroma":    indexado_chroma,
        **extra,
    }


def salvar_arquivos(
    titulo: str,
    url: str,
    conteudo: str,
    meta_ia: dict,
    tipo: str,
    extra: dict,
) -> None:
    """Salva .txt e .json em PASTA_SAIDA. Evita colisão de slug entre URLs diferentes."""
    slug = slugify(titulo)
    base = os.path.join(PASTA_SAIDA, slug)

    sufixo = 1
    while os.path.exists(f"{base}.json"):
        if _ler_url_json(f"{base}.json") == url:
            break  # mesma URL, sobrescreve normalmente
        base = os.path.join(PASTA_SAIDA, f"{slug}_{sufixo}")
        sufixo += 1

    dc = date.today().isoformat()
    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(
            _texto_txt_cabecalho_dr_ajuda(
                tipo,
                titulo,
                url,
                dc,
                meta_ia.get("resumo", ""),
                conteudo,
            )
        )

    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(
            _dict_json_sidecar_dr_ajuda(
                tipo,
                titulo,
                url,
                dc,
                meta_ia,
                conteudo,
                extra,
                indexado_chroma=True,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


def _ler_url_json(caminho: str) -> str:
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f).get("url", "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DA PASTA CENTRAL
# ══════════════════════════════════════════════════════════════════════════════

def salvar_catalogo() -> None:
    """
    Gera (ou atualiza) o arquivo _CATALOGO.md na PASTA_SAIDA com um resumo
    legível de todo o conteúdo indexado: artigos por domínio, vídeos por canal,
    estatísticas gerais e data da última atualização.
    Chamado automaticamente após cada indexação e com --status.
    """
    _ensure_db()
    if not os.path.exists(PASTA_SAIDA):
        return

    arquivos_json = [
        f for f in os.listdir(PASTA_SAIDA)
        if f.endswith(".json") and not f.startswith("_")
    ]

    artigos: list = []
    videos:  list = []

    for nome in sorted(arquivos_json):
        caminho = os.path.join(PASTA_SAIDA, nome)
        try:
            with open(caminho, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if not isinstance(meta, dict):
            continue

        tipo   = meta.get("tipo", "")
        titulo = meta.get("titulo", nome[:-5])
        url    = meta.get("url", "")
        resumo = meta.get("resumo", "")
        data   = meta.get("data_coleta", "")
        chars  = meta.get("tamanho_chars", 0)
        tags   = ", ".join(meta.get("palavras_chave", [])) or "—"
        nivel  = meta.get("nivel_tecnico", "")

        entrada = {
            "titulo": titulo,
            "url":    url,
            "resumo": resumo,
            "data":   data,
            "chars":  chars,
            "tags":   tags,
            "nivel":  nivel,
        }

        if tipo == "video_youtube":
            entrada["canal"]   = meta.get("canal", "")
            entrada["duracao"] = meta.get("duracao", "")
            videos.append(entrada)
        else:
            entrada["dominio"] = meta.get("dominio", urlparse(url).netloc if url else "")
            artigos.append(entrada)

    # ── Agrupa artigos por domínio ────────────────────────────────────────
    dominios: dict = {}
    for a in artigos:
        d = a["dominio"] or "local"
        dominios.setdefault(d, []).append(a)

    # ── Agrupa vídeos por canal ───────────────────────────────────────────
    canais: dict = {}
    for v in videos:
        c = v["canal"] or "Sem canal"
        canais.setdefault(c, []).append(v)

    # ── Escreve o Markdown ────────────────────────────────────────────────
    linhas: list = []
    def w(s=""): linhas.append(s)

    w("# 📚 Catálogo da Base de Conhecimento")
    w()
    w(f"> Gerado automaticamente por **sup** em {date.today().isoformat()}")
    w(f"> Pasta: `{PASTA_SAIDA}`")
    w()
    w("---")
    w()
    w("## 📊 Resumo")
    w()
    w(f"| Item | Quantidade |")
    w(f"|------|-----------|")
    w(f"| Artigos indexados | {len(artigos)} |")
    w(f"| Vídeos indexados  | {len(videos)} |")
    w(f"| Domínios de artigos | {len(dominios)} |")
    w(f"| Canais de vídeo   | {len(canais)} |")
    w(f"| Chunks no banco   | {colecao.count()} |")
    w()

    if dominios:
        w("---")
        w()
        w("## 📄 Artigos")
        w()
        for dominio, itens in sorted(dominios.items()):
            w(f"### 🌐 {dominio} ({len(itens)} artigo(s))")
            w()
            for a in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{a['titulo']}]({a['url']})" if a["url"] else a["titulo"]
                w(f"#### {link}")
                if a["resumo"]:
                    w(f"> {a['resumo']}")
                meta_line = []
                if a["data"]:   meta_line.append(f"📅 {a['data']}")
                if a["chars"]:  meta_line.append(f"📝 {a['chars']:,} chars")
                if a["nivel"]:  meta_line.append(f"🎓 {a['nivel']}")
                if a["tags"] != "—": meta_line.append(f"🏷️ {a['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    if canais:
        w("---")
        w()
        w("## 🎬 Vídeos YouTube")
        w()
        for canal, itens in sorted(canais.items()):
            w(f"### 📺 {canal} ({len(itens)} vídeo(s))")
            w()
            for v in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{v['titulo']}]({v['url']})" if v["url"] else v["titulo"]
                w(f"#### {link}")
                if v["resumo"]:
                    w(f"> {v['resumo']}")
                meta_line = []
                if v["data"]:    meta_line.append(f"📅 {v['data']}")
                if v["duracao"]: meta_line.append(f"⏱️ {v['duracao']}")
                if v["chars"]:   meta_line.append(f"📝 {v['chars']:,} chars")
                if v["nivel"]:   meta_line.append(f"🎓 {v['nivel']}")
                if v["tags"] != "—": meta_line.append(f"🏷️ {v['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    caminho_catalogo = os.path.join(PASTA_SAIDA, "_CATALOGO.md")
    with open(caminho_catalogo, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"  📋 Catálogo atualizado: {caminho_catalogo}")

_RE_TAG     = re.compile(r"\[[^\]]+\]")
_RE_ESPACO  = re.compile(r"[ \t]+")
_RE_NEWLINE = re.compile(r"\n{3,}")
_RE_REPETE  = re.compile(r"(\b\w{1,16}\b)(\s+\1\b)+", re.IGNORECASE)


def _limpar_local(texto: str) -> str:
    """Limpeza por regex — sem tokens Groq, muito rápida."""
    if not texto:
        return ""
    t = _RE_TAG.sub(" ", texto)
    t = _RE_ESPACO.sub(" ", t)
    linhas = [ln.strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in linhas if ln)
    t = _RE_NEWLINE.sub("\n\n", t)
    t = _RE_REPETE.sub(r"\1", t)
    return t.strip()


def _dividir_para_limpeza(texto: str) -> List[str]:
    """
    Parte em blocos de até LIMPEZA_MAX_CHARS chars, quebrando em newlines
    para preservar contexto. Evita 413 / esgotamento de TPM na API Groq.
    """
    if len(texto) <= LIMPEZA_MAX_CHARS:
        return [texto]
    blocos: List[str] = []
    i, n = 0, len(texto)
    while i < n:
        end = min(i + LIMPEZA_MAX_CHARS, n)
        if end < n:
            trecho = texto[i:end]
            nl = trecho.rfind("\n")
            if nl > LIMPEZA_MAX_CHARS // 3:
                end = i + nl + 1
        blocos.append(texto[i:end])
        i = end
    return blocos



def limpar_transcricao(texto_bruto: str) -> str:
    """
    Limpa a transcrição de vídeo conforme LIMPEZA_TRANSCRICAO:
      local  (padrão) → regex, 0 tokens, muito rápido
      ia              → Groq em blocos, melhor qualidade
      nenhuma         → texto bruto
    """
    modo = LIMPEZA_TRANSCRICAO
    if modo in ("nenhuma", "none", "off", "raw"):
        print("  ⏭️  Limpeza desativada — texto bruto.")
        return (texto_bruto or "").strip()
    if modo in ("local", "rapida", "fast", "regex"):
        print("  🧹 Limpeza local (regex, 0 tokens)...")
        return _limpar_local(texto_bruto)

    # modo ia
    print("  🧹 Limpando com Groq (modo ia)...")
    partes = _dividir_para_limpeza(texto_bruto)
    total  = len(partes)
    if total > 1:
        print(f"     ({total} blocos de até {LIMPEZA_MAX_CHARS} chars)")
    saidas: List[str] = []
    for idx, bloco in enumerate(partes, start=1):
        cab = f"Trecho {idx} de {total}.\n\n" if total > 1 else ""
        saidas.append(chamar_groq(
            PROMPT_LIMPEZA_SISTEMA,
            f"{cab}Limpe:\n\n{bloco}",
            modelo=MODELO_LIMPEZA,
        ))
        if idx < total:
            time.sleep(LIMPEZA_PAUSA_S)
    return "\n\n".join(s.strip() for s in saidas if s.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: ARTIGOS WEB
# ══════════════════════════════════════════════════════════════════════════════

def _get_soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = http.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, BS_PARSER)
    except requests.RequestException as e:
        print(f"   ⚠️  GET falhou ({url}): {e}")
        return None


def _raspar_de_soup(url: str, soup: BeautifulSoup) -> Optional[dict]:
    """Extrai título e texto a partir de um BeautifulSoup já carregado."""
    for sel in SELETORES_LIXO:
        for el in soup.select(sel):
            el.decompose()

    h1    = soup.find("h1")
    title = soup.find("title")
    titulo = (
        h1.get_text(strip=True)    if h1    else
        title.get_text(strip=True) if title else
        urlparse(url).path.split("/")[-1] or "Sem título"
    )

    el = None
    for s in SELETORES_CONTEUDO:
        el = soup.select_one(s)
        if el is not None:
            break
    if el is None:
        el = soup.find("body") or soup

    conteudo = re.sub(r"\s{2,}", " ", el.get_text(separator=" ", strip=True)).strip()

    if not conteudo:
        print("   ⚠️  Conteúdo vazio após extração — pulando")
        return None

    if RAG_MIN_CHARS_ARTIGO > 0 and len(conteudo) < RAG_MIN_CHARS_ARTIGO:
        print(
            f"   ⚠️  Conteúdo abaixo do mínimo ({len(conteudo)} < {RAG_MIN_CHARS_ARTIGO} chars) — pulando "
            f"(ajuste RAG_MIN_CHARS_ARTIGO ou use 0 para aceitar textos curtos)"
        )
        return None

    return {"titulo": titulo, "conteudo": conteudo, "url": url}


def raspar_artigo(url: str) -> Optional[dict]:
    """Raspa artigo web com seletores inteligentes e remoção de lixo."""
    soup = _get_soup(url)
    if soup is None:
        return None
    return _raspar_de_soup(url, soup)


def _baixar_e_raspar_artigo(url: str) -> Tuple[str, Optional[dict]]:
    """
    GET + parse em sessão própria (seguro para ThreadPoolExecutor).
    Usado no lote de artigos quando RAG_ARTIGOS_FETCH_THREADS > 1.
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)
    try:
        r = sess.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, BS_PARSER)
        return url, _raspar_de_soup(url, soup)
    except requests.RequestException as e:
        print(f"   ⚠️  GET falhou ({url}): {e}")
        return url, None


def processar_artigo(
    url: str,
    processadas: Set[str],
    *,
    atualizar_catalogo: bool = True,
    artigo: Optional[dict] = None,
) -> int:
    """
    Raspa, gera metadados, indexa e salva um artigo.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro.

    ``artigo``: se já vier raspado (ex.: download paralelo), pula novo GET.
    ``atualizar_catalogo``: em lotes grandes, use False e chame salvar_catalogo() ao final.
    """
    _ensure_db()
    if url in processadas:
        return 0

    if artigo is None:
        artigo = raspar_artigo(url)
    if not artigo:
        return -1

    print(f"   ✓ \"{artigo['titulo'][:65]}\" — {len(artigo['conteudo'])} chars")

    meta_ia = gerar_metadados(artigo["titulo"], artigo["conteudo"], "artigo")
    dominio = urlparse(url).netloc

    n = indexar_conteudo(
        conteudo=artigo["conteudo"],
        titulo=artigo["titulo"],
        url=url,
        tipo="artigo",
        meta_extra={"dominio": dominio},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=artigo["titulo"],
        url=url,
        conteudo=artigo["conteudo"],
        meta_ia=meta_ia,
        tipo="artigo",
        extra={"dominio": dominio},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    if atualizar_catalogo:
        salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

_RE_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


def extrair_video_id(url: str) -> Optional[str]:
    """Aceita URL completa, youtu.be, shorts ou ID puro de 11 chars."""
    s = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _RE_YT_ID.search(s)
    return m.group(1) if m else None


def _metadados_yt(video_id: str) -> dict:
    """Obtém título e canal via oEmbed — sem precisar de API key."""
    query = urlencode({
        "url":    f"https://www.youtube.com/watch?v={video_id}",
        "format": "json",
    })
    try:
        with urlopen(f"https://www.youtube.com/oembed?{query}", timeout=10) as r:
            data = json.loads(r.read().decode())
        return {
            "titulo": data.get("title",       video_id),
            "canal":  data.get("author_name", ""),
        }
    except Exception:
        return {"titulo": video_id, "canal": ""}


def _pausa_entre_videos_no_lote() -> None:
    """Espaça pedidos ao YouTube entre um vídeo e o próximo (base + jitter aleatório)."""
    base = RAG_YT_PAUSA_ENTRE_VIDEOS
    jit = RAG_YT_PAUSA_JITTER
    if base <= 0 and jit <= 0:
        return
    extra = random.uniform(0.0, jit) if jit > 0 else 0.0
    time.sleep(max(0.0, base + extra))


def _fetch_transcricao_youtube(video_id: str):
    """
    Busca legenda com retentativas, backoff e suporte a proxy opcional.

    Configuração via .env:
      RAG_YT_PROXY=http://usuario:senha@host:porta   → roteia via proxy HTTP(S)
      RAG_YT_FETCH_RETRIES                           → tentativas antes de desistir
      RAG_YT_FETCH_BACKOFF                           → base do backoff exponencial (s)
      RAG_YT_PRE_FETCH                               → pausa antes da 1ª tentativa (s)
    """
    ultimo: Optional[BaseException] = None

    # Monta kwargs de proxy uma só vez (None = sem proxy)
    proxy_kwargs: dict = {}
    if RAG_YT_PROXY:
        proxy_kwargs["proxies"] = {"http": RAG_YT_PROXY, "https": RAG_YT_PROXY}

    for tentativa in range(RAG_YT_FETCH_RETRIES):
        try:
            if tentativa == 0 and RAG_YT_PRE_FETCH > 0:
                time.sleep(RAG_YT_PRE_FETCH)
            api = YouTubeTranscriptApi()
            return api.fetch(video_id, languages=["pt", "pt-BR", "en"], **proxy_kwargs)
        except Exception as e:
            ultimo = e
            nome = type(e).__name__
            if nome in ("IpBlocked", "RequestBlocked"):
                # Bloqueio anti-bot: retry imediato piora — sai direto.
                break
            if tentativa >= RAG_YT_FETCH_RETRIES - 1:
                break
            w = min(
                120.0,
                RAG_YT_FETCH_BACKOFF * (2 ** tentativa) + random.uniform(0.25, 1.25),
            )
            print(
                f"   ⏸️  Falha ao obter transcrição — aguardando {w:.1f}s "
                f"(tentativa {tentativa + 1}/{RAG_YT_FETCH_RETRIES})"
            )
            time.sleep(w)
    assert ultimo is not None
    raise ultimo


def _eh_bloqueio_yt(e: BaseException) -> bool:
    n = type(e).__name__
    if n in ("IpBlocked", "RequestBlocked"):
        return True
    msg = str(e).lower()
    return "ipblocked" in msg or "requestblocked" in msg or "youtube is blocking requests" in msg


def _obter_transcricao_ytdlp(url: str) -> Optional[str]:
    """
    Fallback via yt-dlp: tenta baixar legendas (manual/auto) e retornar texto.
    Requer yt-dlp instalado no sistema. Pode usar cookies.txt opcional.
    """
    if not RAG_YT_DLP_ENABLED:
        return None
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("   ⚠️  yt-dlp não encontrado; fallback desativado.")
        return None

    tmp_dir = os.path.join(PASTA_SAIDA, "_tmp_ytdlp_subs")
    os.makedirs(tmp_dir, exist_ok=True)
    out_tpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "pt,pt-BR,en",
        "--sub-format", "vtt",
        "-o", out_tpl,
        url,
    ]
    if RAG_YT_DLP_COOKIES_FILE:
        cmd = cmd[:1] + ["--cookies", RAG_YT_DLP_COOKIES_FILE] + cmd[1:]

    try:
        subprocess.run(cmd, cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("   ⚠️  yt-dlp falhou ao obter legendas.")
        return None

    vid = extrair_video_id(url) or ""
    candidatos = []
    if vid:
        for nome in os.listdir(tmp_dir):
            if nome.startswith(vid) and nome.endswith(".vtt"):
                candidatos.append(os.path.join(tmp_dir, nome))
    if not candidatos:
        return None

    # Preferir pt/pt-BR quando disponível
    candidatos.sort(key=lambda p: (".pt" not in p and ".pt-BR" not in p, len(p)))
    try:
        with open(candidatos[0], encoding="utf-8", errors="ignore") as f:
            vtt = f.read()
    except Exception:
        return None

    linhas = []
    for ln in vtt.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.startswith(("WEBVTT", "NOTE")):
            continue
        if "-->" in t:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        linhas.append(t)
    return " ".join(linhas).strip() or None


def processar_video(
    url: str,
    processadas: Set[str],
    *,
    atualizar_catalogo: bool = True,
) -> int:
    """
    Transcreve, limpa, gera metadados, indexa e salva um vídeo do YouTube.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro | -2=bloqueio IP.

    Ordem de tentativas de transcrição:
      Se RAG_YT_DLP_PRIMARY=1  → yt-dlp primeiro, youtube-transcript-api como fallback.
      Se RAG_YT_DLP_ENABLED=1  → youtube-transcript-api primeiro, yt-dlp após bloqueio.
      Padrão                   → somente youtube-transcript-api.
    """
    _ensure_db()
    if url in processadas:
        return 0

    video_id = extrair_video_id(url)
    if not video_id:
        print(f"   ⚠️  Não foi possível extrair ID do vídeo: {url}")
        return -1

    texto_bruto: Optional[str] = None
    transcript = None

    # ── Caminho 1: yt-dlp como método PRIMÁRIO ────────────────────────────────
    if RAG_YT_DLP_PRIMARY and RAG_YT_DLP_ENABLED:
        print(f"   📹 Obtendo legenda via yt-dlp (primário): {url}")
        texto_bruto = _obter_transcricao_ytdlp(url)
        if texto_bruto:
            transcript = [{"text": texto_bruto, "start": 0.0, "duration": 0.0}]
            print("   ✓ Legenda obtida via yt-dlp.")
        else:
            print("   ⚠️  yt-dlp não retornou legenda — tentando youtube-transcript-api...")

    # ── Caminho 2: youtube-transcript-api (padrão / fallback do caminho 1) ────
    if texto_bruto is None:
        print(f"   📹 Baixando transcrição: {url}")
        try:
            transcript = _fetch_transcricao_youtube(video_id)
            texto_bruto = " ".join(
                t["text"] if isinstance(t, dict) else t.text for t in transcript
            )
        except Exception as e:
            if _eh_bloqueio_yt(e):
                print("   ⛔ YouTube bloqueou requisições deste IP (IpBlocked/RequestBlocked).")
                if RAG_YT_DLP_ENABLED:
                    print("   ↪ Tentando fallback por yt-dlp (legendas)...")
                    texto_bruto = _obter_transcricao_ytdlp(url)
                    if texto_bruto:
                        transcript = [{"text": texto_bruto, "start": 0.0, "duration": 0.0}]
                    else:
                        print(f"   ⏸️  Recomendado aguardar ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tentar novamente.")
                        return -2
                else:
                    print(f"   ⏸️  Recomendado aguardar ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tentar novamente.")
                    print("   Dica: defina RAG_YT_DLP_ENABLED=1 (requer yt-dlp instalado)")
                    if RAG_YT_PROXY:
                        print(f"   Proxy ativo: {RAG_YT_PROXY[:40]}{'...' if len(RAG_YT_PROXY) > 40 else ''}")
                    else:
                        print("   Dica extra: defina RAG_YT_PROXY=http://host:porta para rotear via proxy.")
                    return -2
            print(f"   ⚠️  Transcrição indisponível: {e}")
            return -1

    if texto_bruto is None or transcript is None:
        print("   ⚠️  Não foi possível obter a transcrição por nenhum método.")
        return -1

    # ── Duração estimada a partir do último item ──────────────────────────────
    ultimo  = transcript[-1]
    start   = float(ultimo["start"]    if isinstance(ultimo, dict) else getattr(ultimo, "start",    0))
    dur     = float(ultimo["duration"] if isinstance(ultimo, dict) else getattr(ultimo, "duration", 0))
    dur_s   = int(math.ceil(start + dur))
    duracao = f"{dur_s // 3600:02d}:{(dur_s % 3600) // 60:02d}:{dur_s % 60:02d}"

    if RAG_YT_APOS_TRANSCRICAO > 0:
        time.sleep(RAG_YT_APOS_TRANSCRICAO)

    meta_yt = _metadados_yt(video_id)
    titulo  = meta_yt["titulo"]
    canal   = meta_yt["canal"]
    print(f"   ✓ \"{titulo[:65]}\" — {duracao}")

    conteudo = limpar_transcricao(texto_bruto)
    meta_ia  = gerar_metadados(titulo, conteudo, "video_youtube")

    n = indexar_conteudo(
        conteudo=conteudo,
        titulo=titulo,
        url=url,
        tipo="video_youtube",
        meta_extra={"canal": canal, "duracao": duracao, "video_id": video_id},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=titulo,
        url=url,
        conteudo=conteudo,
        meta_ia=meta_ia,
        tipo="video_youtube",
        extra={"canal": canal, "duracao": duracao, "video_id": video_id},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    if atualizar_catalogo:
        salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: PROCESSAMENTO DE ARQUIVOS LOCAIS (.txt + .json)
# ══════════════════════════════════════════════════════════════════════════════

_MARCADOR_CONTEUDO_TXT = "[CONTEÚDO]"


def _parse_cabecalho_txt_dr_ajuda(head: str) -> Dict[str, str]:
    """Lê linhas TIPO:/TÍTULO:/URL:/DATA:/RESUMO: geradas por salvar_arquivos."""
    out: Dict[str, str] = {}
    for raw in (head or "").splitlines():
        linha = raw.strip()
        if not linha:
            continue
        if linha.startswith("TIPO:"):
            out["tipo"] = linha[5:].strip()
        elif linha.startswith("TÍTULO:") or linha.startswith("TITULO:"):
            out["titulo"] = linha.split(":", 1)[1].strip()
        elif linha.startswith("URL:"):
            out["url"] = linha[4:].strip()
        elif linha.startswith("DATA:"):
            out["data_coleta"] = linha[5:].strip()
        elif linha.startswith("RESUMO:"):
            out["resumo"] = linha[7:].strip()
    return out


def txt_tem_formato_padrao_sistema(texto_bruto: str) -> bool:
    """True se parecer um .txt já no layout Dr. Ajuda (TIPO + … + [CONTEÚDO])."""
    t = (texto_bruto or "").lstrip("\ufeff")
    if _MARCADOR_CONTEUDO_TXT not in t:
        return False
    for raw in t.splitlines():
        s = raw.strip()
        if not s:
            continue
        return s.startswith("TIPO:")
    return False


def extrair_corpo_txt_local_bruto(texto_bruto: str, nome_txt: str) -> Tuple[str, Dict[str, str]]:
    """
    Se o .txt tiver o bloco padrão do sistema (cabeçalhos + [CONTEÚDO]),
    devolve só o corpo e os campos do cabeçalho. Caso contrário devolve o arquivo
    inteiro (strip) e dict vazio.
    """
    _ = nome_txt
    txt = (texto_bruto or "").lstrip("\ufeff")
    if _MARCADOR_CONTEUDO_TXT in txt:
        head, _, tail = txt.partition(_MARCADOR_CONTEUDO_TXT)
        campos = _parse_cabecalho_txt_dr_ajuda(head)
        corpo = tail.lstrip("\n\r")
        return corpo, campos
    return txt.strip(), {}


def titulo_sugerido_do_nome_txt(nome_txt: str) -> str:
    base = nome_txt[:-4] if nome_txt.lower().endswith(".txt") else nome_txt
    s = base.replace("_", " ").replace("-", " ").strip()
    return s or nome_txt


def inspecionar_txt_local(caminho_txt: str) -> Optional[dict]:
    """Resume estado do arquivo: companheiro JSON, formato e rótulos úteis ao usuário."""
    caminho_txt = os.path.abspath(caminho_txt)
    if not caminho_txt.lower().endswith(".txt") or not os.path.isfile(caminho_txt):
        return None
    pasta, nome_txt = os.path.dirname(caminho_txt), os.path.basename(caminho_txt)
    base = nome_txt[:-4]
    caminho_json = os.path.join(pasta, f"{base}.json")
    try:
        with open(caminho_txt, encoding="utf-8") as f:
            bruto = f.read()
    except OSError:
        return None
    formatado = txt_tem_formato_padrao_sistema(bruto)
    tem_json = os.path.isfile(caminho_json)
    return {
        "nome_txt": nome_txt,
        "caminho_txt": caminho_txt,
        "caminho_json": caminho_json,
        "tem_json": tem_json,
        "formatado_sistema": formatado,
        "texto_cru_sem_par": (not formatado) and (not tem_json),
        "formatado_sem_par": formatado and (not tem_json),
        "chars": len(bruto),
    }


def preparar_txt_sem_par_como_fonte_indexavel(
    caminho_txt: str,
    *,
    sobrescrever_json: bool = False,
) -> bool:
    """
    Para um .txt **sem** ``basename.json`` ao lado (ou com ``--sobrescrever``):

    1. Lê o corpo (ignora cabeçalho Dr. Ajuda se já existir).
    2. Limpa com ``limpar_transcricao`` (mesmo caminho que vídeos).
    3. Gera metadados com ``gerar_metadados`` (mesmo que artigos/vídeos remotos).
    4. Grava .txt no formato padrão + .json enriquecido com ``indexado_chroma: false``.

    Depois use ``sup --local`` para indexar no Chroma.
    """
    caminho_txt = os.path.abspath(caminho_txt)
    pasta, nome_txt = os.path.dirname(caminho_txt), os.path.basename(caminho_txt)
    if not nome_txt.lower().endswith(".txt") or not os.path.isfile(caminho_txt):
        return False

    nome_base = nome_txt[:-4]
    caminho_json = os.path.join(pasta, f"{nome_base}.json")
    if os.path.exists(caminho_json) and not sobrescrever_json:
        return False

    try:
        with open(caminho_txt, encoding="utf-8") as f:
            bruto = f.read()
    except OSError:
        return False

    corpo_bruto, cab = extrair_corpo_txt_local_bruto(bruto, nome_txt)
    if not corpo_bruto.strip():
        return False

    texto_limpo = limpar_transcricao(corpo_bruto)

    tipo = (cab.get("tipo") or "").strip() or "texto_local"
    titulo = (cab.get("titulo") or "").strip() or titulo_sugerido_do_nome_txt(nome_txt)
    url = (cab.get("url") or "").strip() or f"local://{nome_txt}"
    data_coleta = (cab.get("data_coleta") or "").strip() or date.today().isoformat()

    meta_ia = gerar_metadados(titulo, texto_limpo, tipo)

    vid = extrair_video_id(url)
    extra: Dict[str, str] = {}
    if tipo == "video_youtube" and vid:
        extra["video_id"] = vid

    doc = _dict_json_sidecar_dr_ajuda(
        tipo,
        titulo,
        url,
        data_coleta,
        meta_ia,
        texto_limpo,
        extra,
        indexado_chroma=False,
    )

    txt_out = _texto_txt_cabecalho_dr_ajuda(
        tipo,
        titulo,
        url,
        data_coleta,
        meta_ia.get("resumo", ""),
        texto_limpo,
    )

    try:
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(txt_out if txt_out.endswith("\n") else txt_out + "\n")
    except OSError:
        return False

    return True


def pipeline_gerar_json_para_txts_sem_json(
    *,
    pasta: Optional[str] = None,
    sobrescrever_json: bool = False,
) -> None:
    """
    Lista .txt sem companheiro .json, distingue texto cru vs. já formatado,
    e prepara cada um como nas fontes remotas (limpeza + metadados + par .txt/.json).
    """
    alvo = os.path.abspath(pasta or PASTA_SAIDA)
    if not os.path.isdir(alvo):
        print(f"Pasta não encontrada: {alvo}")
        return

    nomes = sorted(a for a in os.listdir(alvo) if a.endswith(".txt"))
    if not nomes:
        print(f"Nenhum .txt em: {alvo}")
        return

    candidatos: List[str] = []
    cru_sem_par = 0
    fmt_sem_par = 0
    for nome in nomes:
        info = inspecionar_txt_local(os.path.join(alvo, nome))
        if not info or info["tem_json"]:
            continue
        candidatos.append(nome)
        if info["texto_cru_sem_par"]:
            cru_sem_par += 1
        elif info["formatado_sem_par"]:
            fmt_sem_par += 1

    print(f"\n📋 Pasta: {alvo}")
    print(f"   • .txt sem .json (a preparar): {len(candidatos)}")
    print(f"      — texto cru (sem cabeçalho Dr. Ajuda): {cru_sem_par}")
    print(f"      — já formatado, só falta o .json: {fmt_sem_par}")
    if not candidatos:
        print("\n✓ Nada a preparar (todos os .txt já têm .json ou pasta vazia).")
        return

    criados = 0
    for nome in candidatos:
        base = nome[:-4]
        caminho_txt = os.path.join(alvo, nome)
        info = inspecionar_txt_local(caminho_txt)
        rotulo = "texto cru" if info and info["texto_cru_sem_par"] else "só faltava JSON"
        if preparar_txt_sem_par_como_fonte_indexavel(
            caminho_txt,
            sobrescrever_json=sobrescrever_json,
        ):
            criados += 1
            print(f"  ✅ Preparado ({rotulo}): {base}.json + .txt padronizado")
        else:
            print(f"  ⚠️  Falhou ou conteúdo vazio: {nome}")

    print(f"\n📎 Preparados com sucesso: {criados}/{len(candidatos)}")
    if not GROQ_ENABLED:
        print(
            "\n  ⚠️  Sem GROQ_API_KEY os metadados usam só fallback curto "
            "(como nos outros fluxos quando a IA falha)."
        )


def pipeline_processar_pasta() -> None:
    """
    Varre PASTA_SAIDA e processa pares .txt + .json ainda não indexados.
    Compatível com arquivos gerados por versões anteriores do sistema.
    """
    _ensure_db()
    if not os.path.exists(PASTA_SAIDA):
        print(f"Pasta não encontrada: {PASTA_SAIDA}")
        return

    txts = [a for a in os.listdir(PASTA_SAIDA) if a.endswith(".txt")]
    if not txts:
        print("Nenhum .txt encontrado na pasta.")
        return

    print(f"\n📂 {len(txts)} arquivo(s) encontrado(s)\n{'─'*50}")
    processadas_set = carregar_processadas()

    for nome_txt in txts:
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            print(f"⚠️  JSON não encontrado para: {nome_txt} — gerando automaticamente...")
            ok = preparar_txt_sem_par_como_fonte_indexavel(caminho_txt)
            if not ok:
                print(f"   ❌ Não foi possível gerar o JSON para: {nome_txt} — pulando.")
                continue
            print(f"   ✅ JSON gerado: {nome_base}.json")

        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  JSON inválido ({caminho_json}): {e} — pulando")
            continue

        if meta.get("indexado_chroma") is True:
            print(f"⏭️  Já indexado: {meta.get('titulo', nome_base)}")
            continue

        titulo = meta.get("titulo", nome_base)
        tipo   = meta.get("tipo", "video_youtube")
        print(f"\n🎬 Processando: {titulo}")

        with open(caminho_txt, encoding="utf-8") as f:
            texto_bruto = f.read()

        corpo_arquivo, _ = extrair_corpo_txt_local_bruto(texto_bruto, nome_txt)
        texto_limpo = limpar_transcricao(corpo_arquivo)
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo if texto_limpo.endswith("\n") else texto_limpo + "\n")

        meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
        extra   = {k: v for k, v in meta.items()
                   if k in ("canal", "dominio", "duracao", "video_id")}

        n = indexar_conteudo(
            conteudo=texto_limpo,
            titulo=titulo,
            url=meta.get("url", ""),
            tipo=tipo,
            meta_extra=extra,
            meta_ia=meta_ia,
        )

        meta.update(meta_ia)
        meta["indexado_chroma"] = True
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Registra no histórico de URLs processadas (se tiver URL válida)
        url_meta = meta.get("url", "")
        if url_meta:
            marcar_processada(url_meta, processadas_set)

        print(f"  ✅ {titulo} — {n} chunks indexados")

    print(f"\n🏁 Concluído! Total de chunks no banco: {colecao.count()}")
    salvar_catalogo()


def relatorio_pasta(pasta: Optional[str] = None) -> None:
    """
    Exibe um inventário detalhado da pasta da base sem alterar nada:

    • Total de .txt encontrados
    • Já indexados no Chroma
    • Com JSON mas ainda pendentes de indexação (sup --local)
    • Sem JSON — orphans (sup --gerar-json-locais para corrigir)

    Equivale ao corrigir_orphans.py --relatorio, integrado ao super.py.
    """
    alvo = os.path.abspath(pasta or PASTA_SAIDA)
    if not os.path.isdir(alvo):
        print(f"❌ Pasta não encontrada: {alvo}")
        return

    todos_txts = sorted(
        a for a in os.listdir(alvo)
        if a.endswith(".txt") and not a.startswith("_")
    )
    jsons_existentes = {
        os.path.splitext(a)[0]
        for a in os.listdir(alvo)
        if a.endswith(".json") and not a.startswith("_")
    }

    com_json: List[str]  = []
    sem_json: List[str]  = []
    for nome in todos_txts:
        base = nome[:-4]
        (com_json if base in jsons_existentes else sem_json).append(nome)

    ja_indexados: List[str] = []
    pendentes:    List[str] = []
    for nome in com_json:
        base = nome[:-4]
        caminho_json = os.path.join(alvo, f"{base}.json")
        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("indexado_chroma"):
                ja_indexados.append(nome)
            else:
                pendentes.append(nome)
        except (json.JSONDecodeError, OSError):
            pendentes.append(nome)

    print(f"\n{'═'*62}")
    print(f"  📂 RELATÓRIO DA PASTA: {alvo}")
    print(f"{'═'*62}")
    print(f"  Total de .txt encontrados   : {len(todos_txts)}")
    print(f"  ✅ Já indexados no Chroma   : {len(ja_indexados)}")
    print(f"  ⏳ Com JSON, aguardando     : {len(pendentes)}")
    print(f"  ❌ Sem JSON (orphans)       : {len(sem_json)}")
    print(f"{'─'*62}")

    if sem_json:
        print(f"\n  Arquivos SEM .json — corrija com:  sup --gerar-json-locais\n")
        for nome in sem_json:
            caminho = os.path.join(alvo, nome)
            try:
                bruto = open(caminho, encoding="utf-8").read(512)
                rotulo = "padrão sistema" if txt_tem_formato_padrao_sistema(bruto) else "texto cru"
                tamanho = os.path.getsize(caminho)
            except OSError:
                rotulo, tamanho = "?", 0
            print(f"    • {nome:<52} {tamanho:>7} bytes  [{rotulo}]")

    if pendentes:
        print(f"\n  Com JSON mas NÃO indexados — indexe com:  sup --local\n")
        for nome in pendentes:
            print(f"    • {nome}")

    if not sem_json and not pendentes:
        print("\n  ✅ Tudo indexado — base em dia!")

    print(f"\n{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: LOTE DE VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar_url_yt(url: str) -> str:
    """Normaliza ID puro (11 chars) para URL completa."""
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return f"https://www.youtube.com/watch?v={url}"
    return url


def _ler_urls_arquivo(caminho: str) -> List[str]:
    """
    Lê URLs de um arquivo de texto — uma por linha.
    Ignora linhas em branco e comentários iniciados com '#'.
    """
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}")
        return []
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    urls = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        urls.append(_normalizar_url_yt(linha))
    return urls


_RE_URL_HTTP = re.compile(r"https?://[^\s<>\'\"`,;)]+", re.I)


def _trim_sufixo_url(u: str) -> str:
    return u.rstrip(").,;]\"'>\r\n\t ")


def extrair_urls_http_em_texto(texto: str) -> List[str]:
    """Encontra URLs http(s) em um bloco colado (várias linhas, espaços ou vírgulas)."""
    out: List[str] = []
    seen: Set[str] = set()
    for m in _RE_URL_HTTP.finditer(texto or ""):
        u = _trim_sufixo_url(m.group(0))
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _resolver_entradas_para_urls_artigos(entradas: List[str], *, run_id: str = "pre-fix-paste") -> List[str]:
    """
    Converte linhas coladas + texto livre em lista de URLs de artigo.
    Hipóteses de debug: H1=arquivo; H2=regex multi-link; H3=fallback linha única.
    """
    if not entradas:
        # region agent log
        _debug_log(run_id, "H1", "sup.py:_resolver_entradas_para_urls_artigos", "empty entradas", {})
        # endregion
        return []

    if len(entradas) == 1:
        lone = entradas[0].strip()
        if os.path.exists(lone) and not lone.startswith(("http://", "https://")):
            urls = _ler_urls_arquivo(lone)
            # region agent log
            _debug_log(
                run_id,
                "H1",
                "sup.py:_resolver_entradas_para_urls_artigos",
                "resolved from file path",
                {"count": len(urls)},
            )
            # endregion
            return urls

    blob = "\n".join(entradas)
    extraidas = extrair_urls_http_em_texto(blob)
    if extraidas:
        # region agent log
        _debug_log(
            run_id,
            "H2",
            "sup.py:_resolver_entradas_para_urls_artigos",
            "resolved via http regex",
            {"blob_len": len(blob), "url_count": len(extraidas), "line_count": len(entradas)},
        )
        # endregion
        return extraidas

    out: List[str] = []
    seen: Set[str] = set()
    for linha in entradas:
        t = linha.strip()
        if t.startswith(("http://", "https://")) and t not in seen:
            seen.add(t)
            out.append(t)
    # region agent log
    _debug_log(
        run_id,
        "H3",
        "sup.py:_resolver_entradas_para_urls_artigos",
        "fallback raw lines only",
        {"blob_len": len(blob), "url_count": len(out), "line_count": len(entradas)},
    )
    # endregion
    return out


def _parece_youtube(s: str) -> bool:
    """Heurística rápida: retorna True se a string parece ser URL ou ID de YouTube."""
    s = (s or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return True
    low = s.lower()
    return "youtube.com" in low or "youtu.be" in low


def _resolver_entradas_para_urls_video(entradas: List[str], *, run_id: str = "pre-fix-paste") -> List[str]:
    """Extrai links YouTube e IDs a partir de texto colado ou lista de linhas."""
    if not entradas:
        # region agent log
        _debug_log(run_id, "H4", "sup.py:_resolver_entradas_para_urls_video", "empty entradas", {})
        # endregion
        return []

    if len(entradas) == 1:
        lone = entradas[0].strip()
        if os.path.exists(lone) and not _parece_youtube(lone):
            urls = _ler_urls_arquivo(lone)
            # region agent log
            _debug_log(
                run_id,
                "H4",
                "sup.py:_resolver_entradas_para_urls_video",
                "resolved from file",
                {"count": len(urls)},
            )
            # endregion
            return urls

    blob = "\n".join(entradas)
    out: List[str] = []
    seen: Set[str] = set()

    for u in extrair_urls_http_em_texto(blob):
        u2 = _normalizar_url_yt(u)
        if _eh_url_youtube(u2) and u2 not in seen:
            seen.add(u2)
            out.append(u2)

    for raw in re.split(r"[\s,;|]+", blob):
        tok = raw.strip().strip("()[]<>'\"")
        if not tok:
            continue
        if _parece_youtube(tok):
            u2 = _normalizar_url_yt(tok)
            if _eh_url_youtube(u2) and u2 not in seen:
                seen.add(u2)
                out.append(u2)

    # region agent log
    _debug_log(
        run_id,
        "H4",
        "sup.py:_resolver_entradas_para_urls_video",
        "video URL resolution done",
        {"blob_len": len(blob), "url_count": len(out), "line_count": len(entradas)},
    )
    # endregion
    return out


def pipeline_artigos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs de artigos em sequência.
    Pula URLs inválidas e já indexadas, com resumo ao final.
    """
    _ensure_db()
    urls_validas: List[str] = []
    for url in urls:
        u = url.strip()
        if not _eh_url(u):
            print(f"⚠️  Ignorando (URL inválida): {u}")
            continue
        if _eh_url_youtube(u):
            print(f"⚠️  Ignorando (é YouTube, use opção de vídeos): {u}")
            continue
        urls_validas.append(u.rstrip("/"))

    if not urls_validas:
        print("Nenhuma URL de artigo válida para processar.")
        return

    processadas  = carregar_processadas()
    novas        = [u for u in urls_validas if u not in processadas]
    ja_indexadas = len(urls_validas) - len(novas)
    total        = len(novas)

    print(f"\n{'═'*55}")
    print(f"  📄 LOTE DE ARTIGOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")

    if not novas:
        print("✅ Todos os artigos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros = 0

    threads = RAG_ARTIGOS_FETCH_THREADS
    por_url: Dict[str, Optional[dict]] = {}

    if threads > 1 and len(novas) >= 2:
        w = min(threads, len(novas))
        print(f"  ⚡ Download paralelo: {w} thread(s) | parser HTML: {BS_PARSER}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as ex:
            futuros = [ex.submit(_baixar_e_raspar_artigo, u) for u in novas]
            for fut in concurrent.futures.as_completed(futuros):
                u, dados = fut.result()
                por_url[u] = dados

        for i, url in enumerate(novas, start=1):
            print(f"\n[{i}/{total}] 📄 {url}")
            artigo = por_url.get(url)
            if not artigo:
                erros += 1
                continue
            n = processar_artigo(
                url, processadas, atualizar_catalogo=False, artigo=artigo,
            )
            if n < 0:
                erros += 1
            else:
                total_chunks += n
            if i < total:
                time.sleep(PAUSA_ENTRE_REQS)
    else:
        for i, url in enumerate(novas, start=1):
            print(f"\n[{i}/{total}] 📄 {url}")
            n = processar_artigo(url, processadas, atualizar_catalogo=False)
            if n < 0:
                erros += 1
            else:
                total_chunks += n
            if i < total:
                time.sleep(PAUSA_ENTRE_REQS)

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")
    salvar_catalogo()


def pipeline_videos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs/IDs de vídeos do YouTube em sequência.
    Pula URLs já indexadas, exibe progresso e resumo ao final.
    """
    _ensure_db()
    # Normaliza todas as URLs e filtra inválidas
    urls_normalizadas = []
    for url in urls:
        url = _normalizar_url_yt(url)
        if not _eh_url_youtube(url):
            print(f"⚠️  Ignorando (não é YouTube): {url}")
            continue
        urls_normalizadas.append(url)

    if not urls_normalizadas:
        print("Nenhuma URL válida para processar.")
        return

    processadas   = carregar_processadas()
    novas         = [u for u in urls_normalizadas if u not in processadas]
    ja_indexadas  = len(urls_normalizadas) - len(novas)
    total         = len(novas)

    print(f"\n{'═'*55}")
    print(f"  🎬 LOTE DE VÍDEOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")
    if total >= 2:
        print(
            f"  ⏱️  Pausa entre vídeos: {RAG_YT_PAUSA_ENTRE_VIDEOS}s + aleatório 0–{RAG_YT_PAUSA_JITTER}s "
            f"(export RAG_YT_PAUSA_ENTRE_VIDEOS / RAG_YT_PAUSA_JITTER para ajustar)"
        )

    if not novas:
        print("✅ Todos os vídeos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0
    ipblock_seq  = 0

    for i, url in enumerate(novas, start=1):
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas, atualizar_catalogo=False)
        if n == -2:
            ipblock_seq += 1
            erros += 1
            if RAG_YT_STOP_ON_IPBLOCK and ipblock_seq >= RAG_YT_MAX_CONSEC_IPBLOCK:
                print(
                    f"\n⛔ Bloqueio de IP detectado ({ipblock_seq}x). Parando lote para não agravar."
                    f"\n   Aguarde ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tente novamente."
                )
                break
        elif n < 0:
            erros += 1
        else:
            ipblock_seq = 0
            total_chunks += n
        if i < total:
            _pausa_entre_videos_no_lote()

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: BUSCA FUZZY DE ARQUIVO LOCAL (indexação por nome/tema)
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().strip().split())


def _tokenizar(texto: str) -> Set[str]:
    tokens: Set[str] = set()
    for parte in _normalizar(texto).split():
        limpo = "".join(ch for ch in parte if ch.isalnum())
        if len(limpo) >= 3:
            tokens.add(limpo)
    return tokens


def _score_relevancia(consulta: str, candidato: dict) -> float:
    """
    Score combinado:
      70% sobreposição de tokens (palavras com ≥3 chars em comum)
      30% similaridade de string via difflib
    """
    alvo = _normalizar(
        f"{candidato['titulo']} {candidato['nome_txt']} {candidato.get('amostra', '')}"
    )
    cons = _normalizar(consulta)
    if not cons or not alvo:
        return 0.0
    ratio       = difflib.SequenceMatcher(None, cons, alvo).ratio()
    t_cons      = _tokenizar(cons)
    t_alvo      = _tokenizar(alvo)
    token_score = len(t_cons & t_alvo) / max(1, len(t_cons))
    return 0.7 * token_score + 0.3 * ratio


def _candidatos_locais() -> List[dict]:
    """Retorna arquivos locais (.txt + .json) ainda não indexados."""
    candidatos = []
    if not os.path.exists(PASTA_SAIDA):
        return candidatos

    for nome_txt in os.listdir(PASTA_SAIDA):
        if not nome_txt.endswith(".txt"):
            continue
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            continue
        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("indexado_chroma") is True:
            continue

        try:
            with open(caminho_txt, encoding="utf-8") as f:
                amostra = f.read(2500)
        except OSError:
            amostra = ""

        candidatos.append({
            "titulo":       meta.get("titulo", nome_base),
            "nome_txt":     nome_txt,
            "caminho_txt":  caminho_txt,
            "caminho_json": caminho_json,
            "amostra":      amostra,
        })
    return candidatos


def _selecionar_candidato(candidatos: List[dict], consulta: str) -> Optional[dict]:
    """
    Seleciona o candidato mais relevante por score combinado.
    Abaixo do limiar de confiança (0.35), apresenta as opções ao usuário.
    """
    if not candidatos or not consulta.strip():
        return None

    ranqueados = sorted(
        ((_score_relevancia(consulta, c), c) for c in candidatos),
        reverse=True,
    )
    melhor_score, melhor = ranqueados[0]

    if melhor_score >= 0.35:
        return melhor

    print("\nMatch automático incerto. Candidatos mais próximos:")
    top = ranqueados[:5]
    for i, (score, c) in enumerate(top, start=1):
        print(f"  {i}. [{score:.2f}] {c['titulo']} ({c['nome_txt']})")

    escolha = input("Número correto (ou Enter para cancelar): ").strip()
    if escolha.isdigit():
        idx = int(escolha) - 1
        if 0 <= idx < len(top):
            return top[idx][1]
    return None


def pipeline_indexar_video_por_nome() -> None:
    """
    Pede uma descrição do arquivo local, encontra via busca fuzzy
    e indexa interativamente após confirmação do usuário.
    """
    _ensure_db()
    candidatos = _candidatos_locais()
    if not candidatos:
        print(f"Nenhum arquivo não indexado encontrado em: {PASTA_SAIDA}")
        return

    consulta  = input("Descreva o vídeo/artigo (nome, tema, palavras do conteúdo): ").strip()
    candidato = _selecionar_candidato(candidatos, consulta)

    if not candidato:
        print("Não foi possível identificar o arquivo.")
        return

    print(f"\nEncontrado: {candidato['titulo']} ({candidato['nome_txt']})")
    confirma = input("É esse? (s/n): ").strip().lower()
    if confirma not in ("s", "sim", "y", "yes"):
        print("Cancelado.")
        return

    try:
        with open(candidato["caminho_json"], encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")
        return

    titulo = meta.get("titulo", candidato["nome_txt"])
    tipo   = meta.get("tipo", "video_youtube")
    print(f"\n🎬 Processando: {titulo}")

    with open(candidato["caminho_txt"], encoding="utf-8") as f:
        texto_bruto = f.read()

    corpo_arquivo, _ = extrair_corpo_txt_local_bruto(texto_bruto, candidato["nome_txt"])
    texto_limpo = limpar_transcricao(corpo_arquivo)
    with open(candidato["caminho_txt"], "w", encoding="utf-8") as f:
        f.write(texto_limpo if texto_limpo.endswith("\n") else texto_limpo + "\n")

    meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
    extra   = {k: v for k, v in meta.items()
               if k in ("canal", "dominio", "duracao", "video_id")}

    n = indexar_conteudo(
        conteudo=texto_limpo,
        titulo=titulo,
        url=meta.get("url", ""),
        tipo=tipo,
        meta_extra=extra,
        meta_ia=meta_ia,
    )
    meta.update(meta_ia)
    meta["indexado_chroma"] = True
    with open(candidato["caminho_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Registra no histórico de URLs processadas (se tiver URL válida)
    url_meta = meta.get("url", "")
    if url_meta:
        processadas_set = carregar_processadas()
        marcar_processada(url_meta, processadas_set)

    print(f"\n🏁 Concluído! {n} chunks indexados. Total no banco: {colecao.count()}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: DESCOBERTA DE LINKS (CRAWLER WEB)
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url_youtube(url: str) -> bool:
    return (
        "youtube.com/watch" in url or
        "youtu.be/" in url or
        "youtube.com/shorts/" in url
    )


def _eh_paginacao(tag_a, href: str, url_base: str) -> bool:
    """Detecta links de paginação por texto, classe CSS ou padrão de URL."""
    texto   = tag_a.get_text(strip=True).lower()
    classes = " ".join(tag_a.get("class", [])).lower()
    gatilhos_texto = {"próximo", "proximo", "next", "seguinte", "›", "»", ">", "mais"}
    if any(p in texto for p in gatilhos_texto):
        return True
    if any(p in classes for p in ("next", "pagination", "pager", "proximo")):
        return True
    if re.search(r"(/page/\d+|[?&]p(?:age)?=\d+)$", href):
        return True
    base_path = urlparse(url_base).path.rstrip("/")
    if re.match(rf"^{re.escape(base_path)}/\d+$", urlparse(href).path.rstrip("/")):
        return True
    return False


def descobrir_links(
    url_base: str,
    filtro_path: Optional[str] = None,
    seguir_paginacao: bool = True,
) -> dict:
    """
    Varre a página de listagem (com suporte a paginação) e separa links em:
      - artigos : páginas do mesmo domínio que passam no filtro de path
      - videos  : links do YouTube
    Retorna {"artigos": [...], "videos": [...]}
    """
    dominio   = urlparse(url_base).netloc
    visitadas: Set[str] = set()
    artigos:   Set[str] = set()
    videos:    Set[str] = set()
    fila      = [url_base]

    print(f"\n🔍 Descobrindo links em: {url_base}")
    if filtro_path:
        print(f"   Filtro de path: '{filtro_path}'")

    while fila:
        url_atual = fila.pop(0)
        if url_atual in visitadas:
            continue
        visitadas.add(url_atual)

        soup = _get_soup(url_atual)
        if soup is None:
            continue

        for tag_a in soup.find_all("a", href=True):
            href = urljoin(url_atual, tag_a["href"]).split("#")[0].rstrip("/")
            if not href:
                continue

            if _eh_url_youtube(href):
                videos.add(href)
                continue

            parsed = urlparse(href)
            if parsed.netloc != dominio:
                continue
            if href == url_atual or href in visitadas:
                continue

            if filtro_path and filtro_path not in href:
                if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                    fila.append(href)
                continue

            # Página de paginação: enfileira para crawl mas NÃO indexa como artigo
            if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                fila.append(href)
                continue

            artigos.add(href)

        time.sleep(PAUSA_ENTRE_REQS)

    print(f"   ✓ {len(artigos)} artigo(s) | {len(videos)} vídeo(s) YouTube")
    return {"artigos": sorted(artigos), "videos": sorted(videos)}


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: Q&A COM CLASSIFICAÇÃO INTELIGENTE DE PERGUNTAS
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS_PT = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "e", "é", "ou", "que", "qual", "quais",
    "quando", "onde", "como", "se", "não", "sim", "à", "às", "ao", "aos",
    "the", "and", "or", "of", "in", "on", "to", "for", "is", "are",
})


def _normalizar_texto_busca(texto: str) -> str:
    t = (texto or "").strip().lower()
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


def _tokenizar_consulta(texto: str, *, min_len: int = 3) -> List[str]:
    """Tokens únicos da consulta, sem stopwords (ordem preservada)."""
    norm = _normalizar_texto_busca(texto)
    if not norm:
        return []
    partes = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)?", norm)
    vistos: Set[str] = set()
    out: List[str] = []
    for p in partes:
        if len(p) < min_len or p in _STOPWORDS_PT or p in vistos:
            continue
        vistos.add(p)
        out.append(p)
    return out


def _tokens_campo_meta(valor: str) -> Set[str]:
    """Termos de um campo de metadado (listas separadas por vírgula)."""
    if not valor:
        return set()
    out: Set[str] = set()
    for item in (valor or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        out.add(item)
        for tok in _tokenizar_consulta(item, min_len=2):
            out.add(tok)
    return out


def _chunk_de_resultado_chroma(doc: str, meta: dict, distancia: float) -> dict:
    meta = meta or {}
    return {
        "texto": doc,
        "titulo": meta.get("titulo", ""),
        "url": meta.get("url", ""),
        "tipo": meta.get("tipo", ""),
        "chunk": meta.get("chunk_index", 0),
        "distancia": float(distancia),
        "meta": meta,
    }


def _fundir_chunks_candidatos(*listas: List[dict]) -> List[dict]:
    """Funde listas de chunks mantendo o melhor score (menor distância)."""
    pool: Dict[Tuple[str, int, str], dict] = {}
    for lista in listas:
        for c in lista or []:
            k = _chave_chunk_unica(c)
            if k not in pool or float(c.get("distancia", 99)) < float(pool[k].get("distancia", 99)):
                pool[k] = c
    return sorted(pool.values(), key=lambda x: float(x.get("distancia", 99)))


def _diversificar_chunks_por_fonte(
    chunks: List[dict],
    top_k: int,
    max_por_fonte: Optional[int] = None,
) -> List[dict]:
    """Limita trechos repetidos do mesmo documento (melhor cobertura de fontes)."""
    limite_fonte = max_por_fonte if max_por_fonte is not None else RAG_MAX_CHUNKS_POR_FONTE
    out: List[dict] = []
    por_fonte: Dict[str, int] = {}
    for c in chunks:
        chave_fonte = (c.get("url") or c.get("titulo") or "").strip().lower()
        if not chave_fonte:
            chave_fonte = f"_{id(c)}"
        if por_fonte.get(chave_fonte, 0) >= limite_fonte:
            continue
        por_fonte[chave_fonte] = por_fonte.get(chave_fonte, 0) + 1
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def _buscar_titulos() -> List[str]:
    """Retorna todos os títulos únicos presentes no ChromaDB."""
    _ensure_db()
    resultado = colecao.get(include=["metadatas"])
    if not resultado or not resultado.get("metadatas"):
        return []
    return list({m.get("titulo", "") for m in resultado["metadatas"] if m.get("titulo")})


def _titulos_para_classificacao(pergunta: str) -> List[str]:
    """
    Lista curta e relevante de títulos para o classificador Groq.
    Evita enviar centenas de títulos (413 / TPM) quando a base cresce.
    """
    k = GROQ_CLASSIF_TITULOS_MAX if GROQ_CLASSIF_TITULOS_MAX > 0 else 30
    cand = _buscar_titulos_candidatos(pergunta, k)
    if cand:
        return cand
    todos = _buscar_titulos()
    return todos[:k] if todos else []


def _buscar_titulos_candidatos(pergunta: str, k: int) -> List[str]:
    """
    Retorna até k títulos candidatos (únicos) usando uma busca vetorial rápida
    nos chunks já existentes. Evita enviar uma lista enorme de títulos ao Groq.
    """
    _ensure_db()
    if not pergunta.strip() or k <= 0:
        return []

    total_docs = colecao.count()
    if total_docs <= 0:
        return []

    n = min(max(5, k * 2), total_docs)  # pega mais chunks para formar k títulos únicos
    try:
        res = colecao.query(query_texts=[pergunta], n_results=n)
    except Exception:
        return []

    metas = (res.get("metadatas") or [[]])[0] or []
    out: List[str] = []
    seen: Set[str] = set()
    for m in metas:
        t = (m.get("titulo") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def _classificar_pergunta(pergunta: str, titulos: List[str]) -> dict:
    """
    Usa Groq para classificar a pergunta em:
      especifica   → filtra por fonte_alvo, busca 4 chunks
      geral        → busca ampla, 6 chunks
      comparativa  → busca ampla, 8 chunks
      fora_de_escopo → não busca, resposta direta
    """
    print("  🔀 Classificando consulta...")
    max_chars = max(2000, int(os.getenv("GROQ_CLASSIF_MAX_CHARS", "12000")))
    linhas: List[str] = []
    n_chars = 0
    for t in titulos:
        linha = f"- {t}"
        if linhas and n_chars + len(linha) + 1 > max_chars:
            break
        linhas.append(linha)
        n_chars += len(linha) + 1
    if len(linhas) < len(titulos):
        print(f"  ⚠️  Lista de títulos truncada ({len(linhas)}/{len(titulos)}) para caber no limite da API.")
    lista = "\n".join(linhas)
    raw = chamar_groq(
        sistema=PROMPT_CLASSIFICADOR_SISTEMA,
        usuario=(
            f"Títulos disponíveis:\n{lista}\n\n"
            f"Pergunta: {pergunta}\n\n"
            '{"tipo":"especifica"|"geral"|"comparativa"|"fora_de_escopo","fonte_alvo":string|null,"n_chunks":number,"raciocinio":string}'
        ),
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"tipo": "geral", "fonte_alvo": None, "n_chunks": 6, "raciocinio": "fallback"}


def _chave_chunk_unica(c: dict) -> Tuple[str, int, str]:
    return (
        (c.get("url") or "").strip(),
        int(c.get("chunk") or 0),
        (c.get("titulo") or "").strip(),
    )


def _buscar_chunks(pergunta: str, classificacao: dict) -> List[dict]:
    _ensure_db()
    n = max(1, classificacao.get("n_chunks", 5))
    total_docs = colecao.count()
    if total_docs == 0:
        return []
    n = min(n, total_docs)

    try:
        resultado = colecao.query(query_texts=[pergunta], n_results=n)
    except Exception:
        return []

    docs = resultado.get("documents", [[]])[0] or []
    metas = resultado.get("metadatas", [[]])[0] or []
    dists = (resultado.get("distances") or [[]])[0] or []
    out: List[dict] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        dist = float(dists[i]) if i < len(dists) else 1.0
        out.append(_chunk_de_resultado_chroma(doc, meta, dist))
    return out


def _expandir_queries_busca(pergunta: str) -> List[str]:
    """Variantes de consulta vetorial (local, sem Groq)."""
    base = (pergunta or "").strip()
    if not base:
        return []

    tokens = _tokenizar_consulta(base)
    candidatas: List[str] = [base]

    if len(tokens) > 1:
        sem_stops = " ".join(tokens)
        if len(sem_stops) > 3:
            candidatas.append(sem_stops)
        for i in range(len(tokens) - 1):
            bigrama = f"{tokens[i]} {tokens[i + 1]}"
            if len(bigrama) > 5:
                candidatas.append(bigrama)

    if len(tokens) >= 3:
        nucleo = " ".join(tokens[:3])
        if len(nucleo) > 5:
            candidatas.append(nucleo)

    if tokens:
        termo_forte = max(tokens, key=len)
        if len(termo_forte) > 3:
            candidatas.append(termo_forte)

    norm_ascii = _normalizar_texto_busca(base)
    if norm_ascii and norm_ascii != base.lower() and len(norm_ascii) > 3:
        candidatas.append(norm_ascii)

    vistos: Set[str] = set()
    resultado: List[str] = []
    for q in candidatas:
        q_clean = q.strip()
        chave = q_clean.lower()
        if len(q_clean) > 3 and chave not in vistos:
            vistos.add(chave)
            resultado.append(q_clean)

    return resultado[:RAG_EXPANDIR_MAX_QUERIES]


def _expandir_queries_groq(pergunta: str, existentes: List[str]) -> List[str]:
    """Acrescenta até 1–2 reformulações clínicas via Groq (opcional)."""
    if not GROQ_ENABLED or not (pergunta or "").strip():
        return existentes
    raw = chamar_groq(
        sistema=PROMPT_EXPANDIR_BUSCA_SISTEMA,
        usuario=f"Pergunta: {pergunta.strip()}",
        json_mode=True,
        modelo=MODELO_RAPIDO,
        temperatura=0.0,
    )
    extras: List[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for q in data.get("queries") or []:
                if isinstance(q, str) and q.strip():
                    extras.append(q.strip())
    except json.JSONDecodeError:
        pass

    vistos = {q.lower() for q in existentes}
    merged = list(existentes)
    for q in extras:
        if q.lower() not in vistos:
            vistos.add(q.lower())
            merged.append(q)
        if len(merged) >= RAG_EXPANDIR_MAX_QUERIES:
            break
    return merged[:RAG_EXPANDIR_MAX_QUERIES]


def _buscar_chunks_ampliado(
    pergunta: str,
    classificacao: dict,
    n_pool: int,
    queries: Optional[List[str]] = None,
) -> List[dict]:
    """Busca vetorial com múltiplas queries; funde e deduplica pelo melhor score."""
    queries = queries if queries is not None else _expandir_queries_busca(pergunta)
    if not queries:
        return []

    n_queries = len(queries)
    n_por_query = max(4, (n_pool + n_queries - 1) // n_queries)
    pool: Dict[Tuple[str, int, str], dict] = {}

    for qi, q in enumerate(queries):
        cls = dict(classificacao)
        cls["n_chunks"] = n_por_query
        # Só filtra por fonte na primeira query (evita perder recall nas expansões)
        if qi > 0 and cls.get("tipo") == "especifica":
            cls = dict(cls)
            cls["tipo"] = "geral"
            cls["fonte_alvo"] = None
        for c in _buscar_chunks(q, cls):
            k = _chave_chunk_unica(c)
            if k not in pool or c.get("distancia", 99) < pool[k].get("distancia", 99):
                pool[k] = c

    return list(pool.values())

def _buscar_por_metadados(pergunta: str, n_resultados: int = 5) -> List[dict]:
    """Busca complementar nos metadados clínicos (amostra limitada da coleção)."""
    _ensure_db()
    if not colecao or colecao.count() == 0:
        return []

    tokens = set(_tokenizar_consulta(pergunta, min_len=2))
    if not tokens:
        return []

    try:
        max_docs = min(RAG_METADATA_SCAN_MAX, colecao.count())
        todos = colecao.get(limit=max_docs, include=["documents", "metadatas"])
    except Exception:
        return []

    docs = todos.get("documents", []) or []
    metas = todos.get("metadatas", []) or []
    matches: List[dict] = []

    for doc, meta in zip(docs, metas):
        meta = meta or {}
        score = 0.0

        for campo, peso in (
            ("condicoes_clinicas", 4.0),
            ("medicamentos", 3.5),
            ("palavras_chave", 2.5),
            ("topicos_abordados", 2.0),
            ("procedimentos", 2.0),
            ("tema_principal", 1.5),
            ("especialidade", 1.0),
            ("resumo", 0.8),
        ):
            campo_tokens = _tokens_campo_meta(str(meta.get(campo) or ""))
            overlap = tokens & campo_tokens
            if overlap:
                score += len(overlap) * peso

        titulo_tokens = _tokens_campo_meta(str(meta.get("titulo") or ""))
        overlap_titulo = tokens & titulo_tokens
        if overlap_titulo:
            score += len(overlap_titulo) * 3.0

        if score <= 0:
            continue

        matches.append({
            **_chunk_de_resultado_chroma(
                doc,
                meta,
                max(0.08, 1.0 - min(score / 24.0, 0.92)),
            ),
            "score_metadata": score,
        })

    matches.sort(key=lambda x: x.get("score_metadata", 0), reverse=True)
    return matches[:n_resultados]



def _deduplicar_fontes_consultadas(resposta: str) -> Tuple[str, dict]:
    """Deduplica itens da seção 'Fontes Consultadas' por URL (ou linha normalizada)."""
    texto = resposta or ""
    header_match = re.search(r"(?is)(?:^|\r?\n)((?:📚\s*)?\*\*fontes consultadas\*\*:?)\s*\r?\n", texto)
    if not header_match:
        return resposta, {"section_found": False, "reason": "header_not_found"}

    start = header_match.end()
    tail = texto[start:]
    end_match = re.search(r"(?is)\r?\n(?:---|⚠️\s*\*\*Aviso)", tail)
    end = (start + end_match.start()) if end_match else len(texto)
    header = texto[:start]
    body = texto[start:end]
    trailer = texto[end:]
    raw_lines = [ln for ln in body.splitlines() if ln.strip()]

    dedup_items: List[str] = []
    seen: Set[str] = set()
    for ln in raw_lines:
        clean = ln.strip()
        content = re.sub(r"^\s*(?:[-*]\s*|\d+\.\s*|\[\d+\]\s*)", "", clean)
        urls = re.findall(r"https?://[^\s\)\]]+", content, flags=re.I)
        key = (urls[0].strip().lower() if urls else re.sub(r"\s+", " ", content.lower()).strip())
        if not key or key in seen:
            continue
        seen.add(key)
        dedup_items.append(content)

    if not dedup_items:
        return resposta, {
            "section_found": True,
            "before_count": len(raw_lines),
            "after_count": 0,
        }

    new_body = "\n".join(f"- [{i}] {item}" for i, item in enumerate(dedup_items, 1))
    patched = header + new_body + trailer
    return patched, {
        "section_found": True,
        "end_found": bool(end_match),
        "before_count": len(raw_lines),
        "after_count": len(dedup_items),
        "removed_count": len(raw_lines) - len(dedup_items),
    }


def _forcar_secao_fontes_consultadas(resposta: str, fontes_unicas: List[dict]) -> Tuple[str, dict]:
    """Reescreve a seção de fontes com a lista única do pipeline."""
    texto = resposta or ""
    header_match = re.search(r"(?is)(?:^|\r?\n)((?:📚\s*)?\*\*fontes consultadas\*\*:?)\s*\r?\n", texto)
    if not header_match:
        return resposta, {"section_found": False, "reason": "header_not_found"}

    start = header_match.end()
    tail = texto[start:]
    end_match = re.search(r"(?is)\r?\n(?:---|⚠️\s*\*\*Aviso)", tail)
    end = (start + end_match.start()) if end_match else len(texto)
    prefix = texto[:start]
    suffix = texto[end:]

    linhas = []
    for i, f in enumerate(fontes_unicas, 1):
        tipo_label = "Vídeo" if f.get("tipo") == "video_youtube" else "Artigo"
        linhas.append(f"{i}. {tipo_label}: {f.get('titulo', '').strip()} — {f.get('url', '').strip()}")
    novo_corpo = "\n".join(linhas)

    return prefix + novo_corpo + suffix, {
        "section_found": True,
        "canonical_count": len(linhas),
        "end_found": bool(end_match),
    }


def _resposta_auxilio_sem_cobertura(
    situacao: str,
    *,
    n_fontes: int = 0,
    sugestoes: Optional[List[str]] = None,
    conversa: bool = False,
) -> str:
    """Fallback quando a base não cobre a consulta."""
    padrao = [
        "Especifique condição, síndrome ou cenário clínico (ex.: \"conduta em paciente com X e comorbidade Y\")",
        "Inclua fármaco, dose, exame, escala ou critério diagnóstico de interesse",
        "Tente sinônimos, nomenclatura técnica (CID, DCB) ou termos mais específicos",
    ]
    itens = sugestoes if sugestoes is not None else padrao
    if conversa:
        dicas = " ".join(itens[:2])
        base = f" A base tem {n_fontes} fonte(s) indexada(s)." if n_fontes else ""
        return (
            f"Não encontrei na base indexada material que responda com segurança a esse ponto.{base} "
            f"Se quiser refinar: {dicas}"
        )
    lista = "\n".join(f"  • {s}" for s in itens)
    info_base = f"\n\nBase indexada: {n_fontes} fonte(s)." if n_fontes else ""
    return (
        f"{situacao}\n\n"
        "Não localizei na base do Dr. Ajuda material suficiente para esta consulta.\n\n"
        f"{lista}"
        f"{info_base}"
    )


def _n_chunks_para_busca(classificacao: dict) -> int:
    """Candidatos no Chroma antes do reranking (pool maior que a resposta final)."""
    tipo = str(classificacao.get("tipo", "geral"))
    padrao = RAG_CHUNKS_BUSCA_POR_TIPO.get(tipo, 6)
    try:
        n_class = int(classificacao.get("n_chunks", padrao))
    except Exception:
        n_class = padrao
    return max(padrao, n_class)


def _n_chunks_para_resposta(classificacao: dict) -> int:
    """Trechos enviados ao Groq após reranking local."""
    tipo = str(classificacao.get("tipo", "geral"))
    padrao = {"especifica": 4, "geral": 5, "comparativa": 6}.get(tipo, 5)
    if GROQ_ECONOMIA:
        alvo = {"especifica": 2, "geral": 3, "comparativa": 3}.get(tipo, 3)
        if GROQ_MAX_N_CHUNKS > 0:
            return max(1, min(alvo, GROQ_MAX_N_CHUNKS, padrao))
        return alvo
    if GROQ_MAX_N_CHUNKS > 0:
        return max(1, min(GROQ_MAX_N_CHUNKS, padrao))
    return padrao


def _filtrar_chunks_distancia(chunks: List[dict]) -> List[dict]:
    """Remove trechos fracos demais vs. o melhor match vetorial."""
    if len(chunks) <= 1:
        return chunks
    best = min(float(c.get("distancia", 99)) for c in chunks)
    limite = best * RAG_DISTANCIA_RATIO
    filtrados = [c for c in chunks if float(c.get("distancia", 99)) <= limite]
    return filtrados if filtrados else sorted(chunks, key=lambda x: x.get("distancia", 99))[:1]


def _tokens_tema_chunk(chunk: dict) -> Set[str]:
    """Tokens do título + metadados de tema (condição, palavras-chave, etc.)."""
    tema: Set[str] = set(_tokenizar_consulta(chunk.get("titulo") or "", min_len=2))
    meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
    for campo in ("tema_principal", "condicoes_clinicas", "palavras_chave", "topicos_abordados"):
        tema |= _tokens_campo_meta(str(meta.get(campo) or ""))
    return tema


def _overlap_tema_pergunta(pergunta: str, chunk: dict) -> Tuple[int, float]:
    """Quantos tokens da pergunta aparecem no tema do chunk e a fração coberta."""
    q = set(_tokenizar_consulta(pergunta, min_len=3))
    if len(q) < 2:
        q = set(_tokenizar_consulta(pergunta, min_len=2))
    if not q:
        return 0, 0.0
    tema = _tokens_tema_chunk(chunk)
    comum = q & tema
    return len(comum), len(comum) / len(q)


def _overlap_titulo_pergunta(pergunta: str, chunk: dict) -> int:
    """Overlap só no título do documento (filtro mais rigoroso)."""
    q = set(_tokenizar_consulta(pergunta, min_len=3))
    if len(q) < 2:
        q = set(_tokenizar_consulta(pergunta, min_len=2))
    if not q:
        return 0
    titulo = set(_tokenizar_consulta(chunk.get("titulo") or "", min_len=2))
    return len(q & titulo)


def _classificar_pergunta_heuristica(pergunta: str) -> dict:
    """Classificação local instantânea (sem Groq)."""
    p = (pergunta or "").lower()
    comparadores = (
        "diferença", "diferenca", "compar", " versus ", " vs ", " vs.", "entre ",
        "melhor que", "ou ", " x ",
    )
    if any(m in p for m in comparadores) and len(p) > 20:
        tipo = "comparativa"
    elif re.search(
        r"\b(receita de bolo|previsão do tempo|futebol|bitcoin|python\s+code)\b",
        p,
    ):
        tipo = "fora_de_escopo"
    else:
        tipo = "geral"
    n = RAG_CHUNKS_BUSCA_POR_TIPO.get(tipo, 12)
    return {
        "tipo": tipo,
        "fonte_alvo": None,
        "n_chunks": n,
        "raciocinio": "classificação heurística local",
    }


def _classificar_pergunta_auto(pergunta: str, titulos: List[str]) -> dict:
    if RAG_CLASSIFICAR_GROQ and GROQ_ENABLED:
        return _classificar_pergunta(pergunta, titulos)
    return _classificar_pergunta_heuristica(pergunta)


def _buscar_chunks_vizinhos_documento(chunk: dict) -> List[dict]:
    """Chunks adjacentes do mesmo URL (contexto contínuo do artigo/vídeo)."""
    _ensure_db()
    url = (chunk.get("url") or "").strip()
    if not url or RAG_VIZINHOS_RAIO <= 0:
        return [chunk]
    try:
        idx = int(chunk.get("chunk") or 0)
    except (TypeError, ValueError):
        idx = 0
    lo = max(0, idx - RAG_VIZINHOS_RAIO)
    hi = idx + RAG_VIZINHOS_RAIO
    try:
        res = colecao.get(
            where={
                "$and": [
                    {"url": url},
                    {"chunk_index": {"$gte": lo}},
                    {"chunk_index": {"$lte": hi}},
                ]
            },
            include=["documents", "metadatas"],
        )
    except Exception:
        return [chunk]

    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        return [chunk]

    vizinhos: List[dict] = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        try:
            ci = int(meta.get("chunk_index", 0))
        except (TypeError, ValueError):
            ci = 0
        vizinhos.append(_chunk_de_resultado_chroma(doc, meta, float(chunk.get("distancia", 1.0))))
        vizinhos[-1]["chunk"] = ci

    vizinhos.sort(key=lambda x: int(x.get("chunk") or 0))
    return vizinhos if vizinhos else [chunk]


def _mesclar_chunk_vizinhos(chunk: dict) -> dict:
    """Une texto dos chunks vizinhos em um único bloco para o LLM."""
    partes = _buscar_chunks_vizinhos_documento(chunk)
    if len(partes) <= 1:
        return chunk
    textos: List[str] = []
    vistos: Set[str] = set()
    for p in partes:
        t = (p.get("texto") or "").strip()
        if t and t not in vistos:
            vistos.add(t)
            textos.append(t)
    if not textos:
        return chunk
    out = dict(chunk)
    out["texto"] = "\n\n".join(textos)
    out["vizinhos_mesclados"] = len(textos)
    return out


def _enriquecer_top_chunks_vizinhos(chunks: List[dict], top_n: int = 2) -> List[dict]:
    if not RAG_VIZINHOS_CHUNK or not chunks:
        return chunks
    out: List[dict] = []
    urls: Set[str] = set()
    for i, c in enumerate(chunks):
        url = (c.get("url") or "").strip()
        if i < top_n and url and url not in urls:
            urls.add(url)
            out.append(_mesclar_chunk_vizinhos(c))
        else:
            out.append(c)
    return out


def _ordenar_chunks_relevancia(pergunta: str, chunks: List[dict]) -> List[dict]:
    """Melhor fonte primeiro → vira Fonte 1 no prompt."""

    def _chave(c: dict) -> Tuple:
        return (
            -float(c.get("score_local", 0)),
            -_overlap_titulo_pergunta(pergunta, c),
            -_overlap_tema_pergunta(pergunta, c)[0],
            float(c.get("distancia", 99)),
        )

    return sorted(chunks, key=_chave)


def _qualidade_retrieval(pergunta: str, chunks: List[dict]) -> str:
    """boa | media | ruim — orienta modo direto vs 2 etapas."""
    if not chunks:
        return "ruim"
    top = _ordenar_chunks_relevancia(pergunta, chunks)[0]
    d = float(top.get("distancia", 99))
    n_tit = _overlap_titulo_pergunta(pergunta, top)
    n_tema, ratio = _overlap_tema_pergunta(pergunta, top)
    if d < 0.42 and (n_tit >= 2 or ratio >= 0.45):
        return "boa"
    if d < 0.58 and (n_tit >= 1 or n_tema >= 2):
        return "media"
    return "ruim"


def _resolver_modo_resposta(qualidade: str) -> str:
    modo = RAG_MODO_RESPOSTA
    if modo in ("direto", "direct", "1"):
        return "direto"
    if modo in ("2etapas", "2", "duas"):
        return "2etapas"
    if RAG_2_ETAPAS_LEGACY and modo == "auto":
        return "2etapas"
    if qualidade in ("boa", "media"):
        return "direto"
    return "2etapas"


def _filtrar_chunks_tema_offtopic(pergunta: str, chunks: List[dict]) -> List[dict]:
    """
    Remove trechos claramente fora do tema quando há âncora forte no pool.
    Ex.: pergunta sobre síndrome metabólica → descarta artigo só sobre selênio.
    """
    if not RAG_FILTRO_TEMA or len(chunks) <= 1:
        return chunks

    q_tokens = set(_tokenizar_consulta(pergunta, min_len=3))
    if len(q_tokens) < 2:
        q_tokens = set(_tokenizar_consulta(pergunta, min_len=2))
    if len(q_tokens) < 2:
        return chunks

    overlaps: List[Tuple[int, float, dict]] = []
    for c in chunks:
        n, ratio = _overlap_tema_pergunta(pergunta, c)
        overlaps.append((n, ratio, c))

    best_n = max(n for n, _, _ in overlaps)
    if best_n < RAG_TEMA_MIN_OVERLAP:
        return chunks

    limiar_corpo = max(2, min(len(q_tokens), 3))
    kept: List[dict] = []
    for n, _ratio, c in overlaps:
        if n >= 1:
            kept.append(c)
            continue

        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        texto_t = set(_tokenizar_consulta(c.get("texto") or "", min_len=3))
        corpo_meta = (
            _tokens_campo_meta(str(meta.get("condicoes_clinicas") or ""))
            | _tokens_campo_meta(str(meta.get("medicamentos") or ""))
            | _tokens_campo_meta(str(meta.get("topicos_abordados") or ""))
        )
        if len(q_tokens & (texto_t | corpo_meta)) >= limiar_corpo:
            kept.append(c)
            continue
        if float(c.get("score_metadata") or 0) >= 8:
            kept.append(c)

    if len(kept) >= max(1, min(2, len(chunks) // 2)):
        pass
    else:
        return chunks

    if RAG_FILTRO_TITULO_ESTRITO and len(kept) > 1:
        best_titulo = max(_overlap_titulo_pergunta(pergunta, c) for c in kept)
        if best_titulo >= 2:
            estritos = [c for c in kept if _overlap_titulo_pergunta(pergunta, c) >= 1]
            if estritos:
                kept = estritos

    return kept if kept else chunks


def _busca_original_suficiente(chunks: List[dict], n_min: int) -> bool:
    if len(chunks) < max(1, n_min - 1):
        return False
    filtrados = _filtrar_chunks_distancia(chunks)
    if len(filtrados) < max(1, n_min - 1):
        return False
    best = min(float(c.get("distancia", 99)) for c in filtrados)
    return best < RAG_DISTANCIA_BOA


def _reranking_local(pergunta: str, chunks: List[dict], top_k: int = 3) -> List[dict]:
    """Rerank híbrido: léxico + vetorial + metadados clínicos indexados (sem Groq)."""
    if not chunks:
        return []

    tokens = set(_tokenizar_consulta(pergunta, min_len=2))
    if not tokens:
        return sorted(chunks, key=lambda x: float(x.get("distancia", 99)))[:top_k]

    dists = [float(c.get("distancia", 1.0)) for c in chunks]
    d_min = min(dists) if dists else 1.0
    d_max = max(dists) if dists else 1.0
    span = max(0.01, d_max - d_min)

    melhor_overlap_tema = max(_overlap_tema_pergunta(pergunta, c)[0] for c in chunks)

    for c in chunks:
        texto_tokens = set(_tokenizar_consulta(c.get("texto") or "", min_len=2))
        titulo_tokens = _tokens_tema_chunk(c)

        overlap_texto = len(tokens & texto_tokens) * 1.8
        overlap_titulo = len(tokens & titulo_tokens) * 4.0

        d_norm = (float(c.get("distancia", d_max)) - d_min) / span
        sim_vetorial = (1.0 - d_norm) * 5.0

        match_metadados = 0.0
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        for campo, peso in (
            ("condicoes_clinicas", 3.0),
            ("medicamentos", 2.8),
            ("palavras_chave", 2.2),
            ("topicos_abordados", 1.6),
            ("procedimentos", 1.4),
            ("tema_principal", 1.2),
        ):
            match_metadados += len(tokens & _tokens_campo_meta(str(meta.get(campo) or ""))) * peso

        if c.get("score_metadata"):
            match_metadados += min(float(c["score_metadata"]) * 0.15, 2.5)

        penalidade_tema = 0.0
        n_tema, _ = _overlap_tema_pergunta(pergunta, c)
        if melhor_overlap_tema >= RAG_TEMA_MIN_OVERLAP and n_tema == 0:
            penalidade_tema = 12.0
        elif melhor_overlap_tema >= 2 and n_tema < 2:
            penalidade_tema = 5.0

        c["score_local"] = (
            overlap_texto + overlap_titulo + sim_vetorial + match_metadados - penalidade_tema
        )

    ranked = sorted(chunks, key=lambda x: x.get("score_local", 0), reverse=True)

    if ranked and ranked[0].get("score_local", 0) < 0.45:
        return []

    return ranked[:top_k]


def _complementar_artigo_principal(pergunta: str, chunks: List[dict]) -> List[dict]:
    """
    Se o melhor resultado tem título alinhado à pergunta, funde todos os chunks
  desse URL em Fonte 1 (conteúdo completo do artigo/vídeo).
    """
    if not chunks:
        return chunks
    top = chunks[0]
    if _overlap_titulo_pergunta(pergunta, top) < 2:
        return chunks
    url = (top.get("url") or "").strip()
    if not url:
        return chunks
    todos = _buscar_todos_chunks_url(url)
    if len(todos) <= 1:
        return chunks
    limite = max(10000, RAG_CHUNK_LLM_MAX_CHARS * 4, GROQ_CONTEXTO_CHUNK_MAX_CHARS)
    fundido = _fundir_chunks_mesmo_documento(todos, limite)
    if not fundido.get("texto"):
        return chunks
    fundido["distancia"] = top.get("distancia", 0.5)
    fundido["score_local"] = top.get("score_local", 0) + 10
    resto = [c for c in chunks[1:] if (c.get("url") or "") != url]
    return [fundido] + resto


def _recuperar_chunks_para_resposta(pergunta: str, classificacao: dict) -> List[dict]:
    """
    Pipeline de recuperação: vetorial → expansão local (e Groq opcional) →
    metadados → rerank híbrido → diversidade por fonte.
    """
    n_busca = _n_chunks_para_busca(classificacao)
    n_resposta = _n_chunks_para_resposta(classificacao)
    cls_busca = dict(classificacao)
    cls_busca["n_chunks"] = n_busca

    pool = _filtrar_chunks_distancia(_buscar_chunks(pergunta, cls_busca))

    if not _busca_original_suficiente(pool, n_resposta):
        queries = _expandir_queries_busca(pergunta)
        if RAG_EXPANDIR_BUSCA and GROQ_ENABLED:
            queries = _expandir_queries_groq(pergunta, queries)
        if RAG_BUSCA_EXPANDIR_LOCAL and len(queries) > 1:
            expandidos = _buscar_chunks_ampliado(
                pergunta, cls_busca, n_busca, queries=queries
            )
            pool = _filtrar_chunks_distancia(_fundir_chunks_candidatos(pool, expandidos))

    if RAG_METADATA_FALLBACK and (
        len(pool) < max(2, n_resposta) or not _busca_original_suficiente(pool, n_resposta)
    ):
        meta_hits = _buscar_por_metadados(
            pergunta, n_resultados=max(n_resposta + 2, min(8, n_busca))
        )
        pool = _filtrar_chunks_distancia(_fundir_chunks_candidatos(pool, meta_hits))

    pool = _filtrar_chunks_tema_offtopic(pergunta, pool)
    candidatos = _reranking_local(
        pergunta, pool, top_k=max(n_resposta + 3, n_resposta * 2)
    )
    finais = _diversificar_chunks_por_fonte(candidatos, n_resposta)
    finais = _ordenar_chunks_relevancia(pergunta, finais)
    return _complementar_artigo_principal(pergunta, finais)


def _montar_contexto_de_chunks(
    pergunta: str,
    chunks: List[dict],
) -> Tuple[str, List[dict], int]:
    """
    Monta contexto para o LLM: melhor fonte = Fonte 1, com resumo dos metadados.
    Retorna (contexto_str, chunks_ordenados, n_fontes_unicas).
    """
    chunks = _ordenar_chunks_relevancia(pergunta, chunks)
    chunks = _enriquecer_top_chunks_vizinhos(chunks, top_n=2)

    limite_chunk = RAG_CHUNK_LLM_MAX_CHARS
    if GROQ_ECONOMIA and GROQ_CONTEXTO_CHUNK_MAX_CHARS > 0:
        limite_chunk = min(limite_chunk, GROQ_CONTEXTO_CHUNK_MAX_CHARS)

    total_max = GROQ_CONTEXTO_TOTAL_MAX_CHARS
    total_chars = 0
    blocos: List[str] = []
    fontes_vistas: dict = {}
    fontes_unicas: List[dict] = []

    for i, c in enumerate(chunks):
        chave = (c.get("titulo", ""), c.get("url", ""))
        if chave not in fontes_vistas:
            fontes_vistas[chave] = len(fontes_vistas) + 1
            fontes_unicas.append({
                "titulo": c.get("titulo", ""),
                "url": c.get("url", ""),
                "tipo": c.get("tipo", ""),
            })
        n_fonte = fontes_vistas[chave]

        tipo_label = "Vídeo" if c.get("tipo") == "video_youtube" else "Artigo"
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        resumo = (meta.get("resumo") or "").strip()
        tema = (meta.get("tema_principal") or "").strip()
        n_tit = _overlap_titulo_pergunta(pergunta, c)
        relevancia = "alta" if n_tit >= 2 else ("média" if n_tit >= 1 else "contexto")

        cabecalho = (
            f"[Fonte {n_fonte} — {tipo_label} — relevância: {relevancia}]\n"
            f"Título: {c.get('titulo', '')}\n"
            f"URL: {c.get('url', '')}\n"
        )
        if tema:
            cabecalho += f"Tema: {tema}\n"
        if resumo:
            cabecalho += f"Resumo indexado: {resumo[:400]}\n"
        if c.get("vizinhos_mesclados", 0) > 1:
            cabecalho += f"(Trecho contínuo: {c['vizinhos_mesclados']} partes do documento)\n"

        cap = limite_chunk * 2 if c.get("documento_completo") else limite_chunk
        texto = _truncar_texto_chunk(c.get("texto") or "", cap, pergunta=pergunta)
        bloco = f"{cabecalho}\n{texto}"

        if total_max > 0:
            prox = len(bloco) + (8 if blocos else 0)
            if total_chars + prox > total_max:
                break
            total_chars += prox

        blocos.append(bloco)

    return "\n\n---\n".join(blocos), chunks, len(fontes_vistas)


def _resposta_sem_trechos(
    pergunta: str,
    titulos: List[str],
    *,
    n_fontes: int,
    conversa: bool = False,
) -> str:
    """Sem trechos recuperados: mensagem fixa (sem LLM — evita alucinar sobre títulos)."""
    corpo = _resposta_auxilio_sem_cobertura(
        "Não localizei trechos que respondam diretamente à consulta",
        n_fontes=n_fontes,
        sugestoes=[
            "Reformule com condição, contexto clínico e termos técnicos (CID, DCB, escalas)",
            "Se o tema não foi indexado, use `--artigos` ou `--video` para ampliar a base",
            "Seja específico sobre o que precisa (critério, dose, conduta, exame)",
        ],
        conversa=conversa,
    )
    if not titulos or conversa:
        if conversa and titulos:
            nomes = ", ".join(titulos[:4])
            if len(titulos) > 4:
                nomes += f" e mais {len(titulos) - 4}"
            return f"{corpo} Temas próximos na base (sem trecho recuperado): {nomes}."
        return corpo
    lista = "\n".join(f"  • {t}" for t in titulos[:10])
    return f"{corpo}\n\nTítulos com possível proximidade:\n{lista}"


class SessaoConsultaCLI:
    """Histórico de consulta clínica no CLI (follow-ups na mesma sessão)."""

    def __init__(self, max_turnos: int = 6):
        self.max_turnos = max(1, max_turnos)
        self.turnos: List[Tuple[str, str]] = []
        self.aviso_emitido = False

    def registrar(self, pergunta: str, resposta: str) -> None:
        p = (pergunta or "").strip()
        r = _resposta_sem_aviso_medico(resposta or "")
        if not p:
            return
        self.turnos.append((p, r))
        if len(self.turnos) > self.max_turnos:
            self.turnos = self.turnos[-self.max_turnos :]

    def limpar(self) -> None:
        self.turnos.clear()
        self.aviso_emitido = False

    def vazia(self) -> bool:
        return not self.turnos

    def texto_busca(self, pergunta: str) -> str:
        """Enriquece a busca vetorial quando a pergunta é dependente do contexto."""
        if self.vazia():
            return pergunta
        ultima_p, _ = self.turnos[-1]
        return f"{ultima_p}\n\n{pergunta}"

    def bloco_historico(self, max_chars: int = 1200) -> str:
        """Só perguntas anteriores — respostas do assistente NÃO entram (evita repetir alucinação)."""
        if self.vazia():
            return ""
        linhas: List[str] = []
        total = 0
        for p, _ in self.turnos:
            bloco = f"- {p}"
            if total + len(bloco) > max_chars:
                break
            linhas.append(bloco)
            total += len(bloco)
        return (
            "Perguntas anteriores do médico (contexto; não são fonte clínica):\n"
            + "\n".join(linhas)
            + "\n\n"
        )


def _resposta_sem_aviso_medico(resposta: str) -> str:
    return re.sub(
        r"\n\n---\n\s*⚠️\s*\*\*Aviso de apoio clínico\*\*.*",
        "",
        resposta or "",
        flags=re.I | re.S,
    ).strip()


_RE_SECAO_ESTRUTURADA = re.compile(
    r"(?im)^\s*(?:"
    r"[🔍🧠🩺💊⚠️🔎📚🔗]\s*)?"
    r"\*{0,2}\s*"
    r"(?:análise(?:\s+da\s+consulta)?|síntese(?:\s+clínica)?|raciocínio\s+diagnóstico|"
    r"conduta\s+clínica|pontos\s+de\s+atenção|red\s+flags|lacunas(?:\s+na\s+base)?|"
    r"fontes\s+consultadas|apoio\s+por\s+situações?\s+relacionadas?)"
    r"\s*\*{0,2}\s*:?\s*$"
)


def _limpar_marcadores_estrutura(texto: str) -> str:
    """Remove cabeçalhos de template que o modelo ainda possa emitir."""
    linhas = [ln for ln in (texto or "").splitlines() if not _RE_SECAO_ESTRUTURADA.match(ln)]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(linhas)).strip()
    return out


def _rodape_referencias_discretas(fontes_unicas: List[dict]) -> str:
    if not fontes_unicas:
        return ""
    refs = []
    for i, f in enumerate(fontes_unicas, 1):
        titulo = (f.get("titulo") or "").strip()
        if titulo:
            refs.append(f"{i}. {titulo}")
    if not refs:
        return ""
    return "\n\nReferências: " + "; ".join(refs)


def _formatar_resposta_conversa(resposta: str, fontes_unicas: List[dict]) -> str:
    texto = _limpar_marcadores_estrutura(_resposta_sem_aviso_medico(resposta))
    # Remove bloco 📚 gerado pelo modelo, se existir
    texto = re.sub(
        r"(?is)\n(?:📚\s*)?\*{0,2}fontes\s+consultadas\*{0,2}.*$",
        "",
        texto,
    ).strip()
    if fontes_unicas and not re.search(r"(?i)\breferências\s*:", texto):
        texto += _rodape_referencias_discretas(fontes_unicas)
    return texto


def _anexar_aviso_sessao(resposta: str, sessao: Optional[SessaoConsultaCLI]) -> str:
    if sessao is not None:
        if sessao.aviso_emitido:
            return resposta
        sessao.aviso_emitido = True
    if re.search(
        r"aviso\s+de\s+apoio\s+cl[ií]nico|n[aã]o\s+substitui\s+o\s+julgamento\s+cl[ií]nico",
        resposta,
        re.I,
    ):
        return resposta
    return resposta + AVISO_MEDICO


def _validar_classificacao(classificacao: dict, titulos: List[str]) -> dict:
    """
    Normaliza tipo/n_chunks; não filtra Chroma por título (evita fonte errada).
    """
    c = dict(classificacao)
    tipo = str(c.get("tipo", "geral")).strip().lower()
    if tipo not in RAG_CHUNKS_BUSCA_POR_TIPO:
        tipo = "geral"
    c["tipo"] = tipo

    try:
        n_llm = int(c.get("n_chunks", RAG_CHUNKS_BUSCA_POR_TIPO[tipo]))
    except (TypeError, ValueError):
        n_llm = RAG_CHUNKS_BUSCA_POR_TIPO[tipo]
    c["n_chunks"] = max(RAG_CHUNKS_BUSCA_POR_TIPO[tipo], n_llm)

    # Busca vetorial sempre ampla; fonte_alvo só informativo para logs futuros
    c["fonte_alvo"] = None
    return c


_RE_YT_LIXO = re.compile(
    r"<[\d:.,c>/]+>|"
    r"\d{2}:\d{2}:\d{2}\.\d{3}(?:\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3})?"
)


def _limpar_texto_para_llm(texto: str) -> str:
    """Remove artefatos de transcrição YouTube que atrapalham o LLM."""
    t = (texto or "").strip()
    if not t:
        return ""
    t = _RE_YT_LIXO.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Frases duplicadas consecutivas (vício de transcrição)
    partes = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    unicas: List[str] = []
    for p in partes:
        if not unicas or p.lower() != unicas[-1].lower():
            unicas.append(p)
    return " ".join(unicas) if unicas else t


def _focar_texto_na_pergunta(pergunta: str, texto: str, max_chars: int) -> str:
    """
    Em documentos longos, prioriza frases com termos da pergunta e valores numéricos.
    """
    t = _limpar_texto_para_llm(texto)
    if not t or len(t) <= max_chars:
        return t

    tokens = set(_tokenizar_consulta(pergunta, min_len=3))
    gatilhos = (
        "criterio", "critério", "diagnost", "diagnóst", "considerado", "alterado",
        "defin", "classif", "cm", "mmhg", "mg/dl", "triglicer", "hdl", "glic",
        "conduta", "tratament", "terapia", "medic", "dose", "idos", "idosa",
        "anti-hipert", "pressao", "pressão", "alvo", "meta",
    )
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", t) if f.strip()]
    if not frases:
        return t[:max_chars]

    pontuadas: List[Tuple[float, int, str]] = []
    for i, fr in enumerate(frases):
        ft = set(_tokenizar_consulta(fr, min_len=3))
        score = len(tokens & ft) * 2.0
        if re.search(r"\d", fr):
            score += 2.5
        fl = fr.lower()
        if any(g in fl for g in gatilhos):
            score += 1.5
        if score > 0:
            pontuadas.append((score, i, fr))

    if not pontuadas:
        return t[:max_chars].rstrip() + "…"

    pontuadas.sort(key=lambda x: (-x[0], x[1]))
    indices: Set[int] = set()
    for _, i, _ in pontuadas:
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(frases):
                indices.add(j)

    ordenados = sorted(indices)
    partes = [frases[i] for i in ordenados]
    focado = " ".join(partes)
    if len(focado) > max_chars:
        return focado[:max_chars].rstrip() + "…"
    return focado


def _truncar_texto_chunk(texto: str, limite: int, pergunta: str = "") -> str:
    t = _limpar_texto_para_llm(texto)
    if pergunta and len(t) > max(limite, 3500):
        t = _focar_texto_na_pergunta(pergunta, t, limite if limite > 0 else len(t))
    elif limite > 0 and len(t) > limite:
        t = t[:limite].rstrip() + "…"
    return t


def _buscar_todos_chunks_url(url: str) -> List[dict]:
    """Todos os chunks de um documento (artigo/vídeo) ordenados."""
    _ensure_db()
    if not url:
        return []
    try:
        res = colecao.get(
            where={"url": url},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    out: List[dict] = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        try:
            ci = int(meta.get("chunk_index", 0))
        except (TypeError, ValueError):
            ci = 0
        out.append(_chunk_de_resultado_chroma(doc, meta, 0.5))
        out[-1]["chunk"] = ci
    out.sort(key=lambda x: int(x.get("chunk") or 0))
    return out


def _fundir_chunks_mesmo_documento(chunks: List[dict], limite_chars: int) -> dict:
    """Um único bloco com o documento inteiro (até limite_chars)."""
    if not chunks:
        return {}
    base = dict(chunks[0])
    textos: List[str] = []
    total = 0
    for c in chunks:
        t = _limpar_texto_para_llm(c.get("texto") or "")
        if not t:
            continue
        if limite_chars > 0 and total + len(t) > limite_chars:
            restante = limite_chars - total
            if restante > 200:
                textos.append(t[:restante].rstrip() + "…")
            break
        textos.append(t)
        total += len(t)
    base["texto"] = "\n\n".join(textos)
    base["documento_completo"] = len(chunks)
    return base


def _extrair_fatos_dos_trechos(pergunta: str, contexto: str, n_fontes: int) -> dict:
    """Etapa 1: fatos literais + fonte — sem síntese."""
    if not GROQ_ENABLED:
        return {"fatos": [], "lacunas": ["IA indisponível"], "cobre_pergunta": "nao"}
    raw = chamar_groq(
        sistema=PROMPT_EXTRAIR_FATOS_SISTEMA,
        usuario=(
            f"Pergunta do médico: {pergunta}\n\n"
            f"Trechos numerados (Fonte 1 a {n_fontes} — a mais relevante costuma ser Fonte 1):\n\n"
            f"{contexto}\n"
        ),
        modelo=MODELO_POTENTE,
        json_mode=True,
        temperatura=0.0,
    )
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            fatos = data.get("fatos") or []
            if not isinstance(fatos, list):
                fatos = []
            limpos = []
            for f in fatos:
                if not isinstance(f, dict):
                    continue
                try:
                    n = int(f.get("fonte", 0))
                except (TypeError, ValueError):
                    continue
                if n < 1 or n > n_fontes:
                    continue
                txt = (f.get("texto") or "").strip()
                if not txt:
                    continue
                rel = str(f.get("relevancia", "alta")).lower()
                limpos.append({
                    "texto": txt,
                    "fonte": n,
                    "trecho_literal": (f.get("trecho_literal") or txt)[:500],
                    "relevancia": rel if rel in ("alta", "media") else "alta",
                })
            alta = [f for f in limpos if f.get("relevancia") == "alta"]
            if alta:
                limpos = alta + [f for f in limpos if f.get("relevancia") != "alta"][:4]
            return {
                "fatos": limpos,
                "lacunas": data.get("lacunas") if isinstance(data.get("lacunas"), list) else [],
                "cobre_pergunta": str(data.get("cobre_pergunta", "nao")).lower(),
            }
    except Exception:
        pass
    return {"fatos": [], "lacunas": ["falha na extração"], "cobre_pergunta": "nao"}


def _sintetizar_fatos_json(
    pergunta: str,
    pacote: dict,
    *,
    conversa: bool,
    contexto_historico: str = "",
) -> str:
    """Etapa 2: redação só a partir do JSON de fatos."""
    prompt = PROMPT_SINTETIZAR_FATOS_SISTEMA if conversa else PROMPT_SINTETIZAR_FATOS_SISTEMA
    return chamar_groq(
        sistema=prompt,
        usuario=(
            f"{contexto_historico}"
            f"Pergunta: {pergunta}\n\n"
            f"Fatos extraídos (única fonte permitida):\n{json.dumps(pacote, ensure_ascii=False)}\n"
        ),
        modelo=MODELO_POTENTE,
        temperatura=0.0,
    )


def _gerar_resposta_direta(
    pergunta: str,
    contexto: str,
    *,
    conversa: bool,
    historico: str = "",
) -> str:
    prompt_sistema = PROMPT_RESPOSTA_CONVERSA if conversa else PROMPT_RESPOSTA_DIRETA_SISTEMA
    instrucao = (
        "Responda à pergunta usando os trechos. Fonte 1 = documento principal.\n"
        "Extraia TODOS os critérios/valores explícitos do trecho (números, cortes, passos).\n"
        "Proibido inferir ou usar conhecimento externo."
    )
    return chamar_groq(
        sistema=prompt_sistema,
        usuario=(
            f"{historico}"
            f"{instrucao}\n\n"
            f"Pergunta: {pergunta}\n\n"
            f"=== TRECHOS INDEXADOS ===\n\n{contexto}\n\n=== FIM ==="
        ),
        modelo=MODELO_POTENTE,
        temperatura=0.0,
    )


def _gerar_resposta_ancorada(
    pergunta: str,
    contexto: str,
    n_fontes: int,
    fontes_unicas: List[dict],
    *,
    conversa: bool,
    historico: str = "",
    silencioso: bool = False,
    chunks: Optional[List[dict]] = None,
) -> str:
    if not GROQ_ENABLED:
        return ""

    qualidade = _qualidade_retrieval(pergunta, chunks or [])
    modo = _resolver_modo_resposta(qualidade)
    if not silencioso:
        print(f"  🧠 Modo resposta: {modo} (qualidade retrieval: {qualidade})")

    if modo == "direto":
        return _gerar_resposta_direta(
            pergunta, contexto, conversa=conversa, historico=historico
        )

    if not silencioso:
        print("  📋 Extraindo fatos dos trechos...")
    pacote = _extrair_fatos_dos_trechos(pergunta, contexto, n_fontes)
    cobre = pacote.get("cobre_pergunta", "nao")
    n_fatos = len(pacote.get("fatos") or [])

    if n_fatos == 0 and cobre != "sim":
        if not silencioso:
            print("  ↩️  Extração vazia — tentando síntese direta dos trechos...")
        return _gerar_resposta_direta(
            pergunta, contexto, conversa=conversa, historico=historico
        )

    if not silencioso:
        print(f"  ✓ {n_fatos} fato(s) · cobertura={cobre}")
        print("  💬 Sintetizando a partir dos fatos...")
    resposta = _sintetizar_fatos_json(
        pergunta, pacote, conversa=conversa, contexto_historico=historico
    )
    if (resposta or "").strip():
        return resposta
    return _gerar_resposta_direta(
        pergunta, contexto, conversa=conversa, historico=historico
    )


def _imprimir_resposta_cli(texto: str) -> None:
    print()
    for par in (texto or "").split("\n\n"):
        p = par.strip()
        if p:
            print(p)
            print()


def pipeline_perguntar(
    pergunta: str,
    sessao: Optional["SessaoConsultaCLI"] = None,
    *,
    modo_conversa: Optional[bool] = None,
) -> str:
    """
    Q&A com classificação inteligente:
    1. Classifica a pergunta (específica / geral / comparativa / fora de escopo)
    2. Busca chunks no ChromaDB (com ou sem filtro de fonte)
    3. Responde usando Groq com o contexto encontrado, citando fontes
    4. Anexa aviso médico obrigatório ao final de toda resposta

    sessao: histórico opcional do CLI para perguntas de continuação.
    modo_conversa: prosa natural sem seções (padrão: True se houver sessão).
    """
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "Informe uma pergunta clínica."

    conversa = modo_conversa if modo_conversa is not None else sessao is not None
    silencioso = conversa

    usar_cache = (sessao is None or sessao.vazia()) and not conversa
    if usar_cache:
        resposta = cache_respostas.get(pergunta)
        if resposta:
            return resposta

    pergunta_busca = sessao.texto_busca(pergunta) if sessao else pergunta

    # Sempre limita títulos no classificador (busca vetorial + cap); evita 413 com bases grandes.
    titulos: List[str] = _titulos_para_classificacao(pergunta_busca)
    if not titulos:
        msg = _resposta_auxilio_sem_cobertura(
            "Base de conhecimento ainda vazia",
            sugestoes=[
                "Indexe conteúdo com `--artigos`, `--video` ou `--local` para eu poder buscar nas fontes",
                "Depois, volte com sua dúvida clínica — estarei pronto para apoiar o raciocínio",
            ],
            conversa=conversa,
        )
        return _anexar_aviso_sessao(msg, sessao) if conversa else msg

    t0 = time.time()
    classificacao = _validar_classificacao(
        _classificar_pergunta_auto(pergunta_busca, titulos), titulos
    )
    if not silencioso:
        print(f"  Tipo: {classificacao['tipo']} | {classificacao.get('raciocinio', '')}")

    if classificacao["tipo"] == "fora_de_escopo":
        msg = _resposta_auxilio_sem_cobertura(
            "Consulta sem correspondência clínica na base",
            n_fontes=len(titulos),
            sugestoes=[
                "Reformule com condição, síndrome ou pergunta clínica explícita",
                "Inclua contexto do paciente ou cenário (idade, comorbidades, fase da doença)",
                "Use terminologia médica (CID, DCB, fármaco, dose, exame ou escala)",
            ],
            conversa=conversa,
        )
        return _anexar_aviso_sessao(msg, sessao) if conversa else msg

    n_busca = _n_chunks_para_busca(classificacao)
    n_resposta = _n_chunks_para_resposta(classificacao)
    if not silencioso:
        print(f"  Chunks: busca={n_busca} → rerank → resposta={n_resposta}")

    if not silencioso:
        print("  🔎 Recuperando trechos (vetorial + rerank híbrido)...")
    chunks = _recuperar_chunks_para_resposta(pergunta_busca, classificacao)
    if not silencioso and chunks:
        best_d = min(float(c.get("distancia", 99)) for c in chunks)
        print(f"  ✓ {len(chunks)} trecho(s) · melhor distância={best_d:.3f}")
    if not chunks:
        resposta_final = _resposta_sem_trechos(
            pergunta, titulos, n_fontes=len(titulos), conversa=conversa
        )
        resposta_final = _anexar_aviso_sessao(resposta_final, sessao)
        if usar_cache:
            cache_respostas.set(pergunta, resposta_final)
        return resposta_final

    contexto, chunks, n_fontes_ctx = _montar_contexto_de_chunks(pergunta_busca, chunks)
    fontes_unicas = []
    vistos: Set[Tuple[str, str]] = set()
    for c in chunks:
        chave = (c.get("titulo", ""), c.get("url", ""))
        if chave not in vistos:
            vistos.add(chave)
            fontes_unicas.append({
                "titulo": c.get("titulo", ""),
                "url": c.get("url", ""),
                "tipo": c.get("tipo", ""),
            })

    historico = sessao.bloco_historico() if sessao else ""
    resposta = _gerar_resposta_ancorada(
        pergunta,
        contexto,
        n_fontes_ctx,
        fontes_unicas,
        conversa=conversa,
        historico=historico,
        silencioso=silencioso,
        chunks=chunks,
    )
    if not (resposta or "").strip():
        resposta_final = _resposta_sem_trechos(
            pergunta, titulos, n_fontes=len(titulos), conversa=conversa
        )
        resposta_final = _anexar_aviso_sessao(resposta_final, sessao)
        if usar_cache:
            cache_respostas.set(pergunta, resposta_final)
        return resposta_final
    if not silencioso:
        elapsed = time.time() - t0
        print(f"  ✅ Resposta gerada em {elapsed:.1f}s")

    if conversa:
        resposta_final = _formatar_resposta_conversa(resposta or "", fontes_unicas)
        resposta_final = _anexar_aviso_sessao(resposta_final, sessao)
    else:
        resposta, _ = _deduplicar_fontes_consultadas(resposta or "")
        resposta, _ = _forcar_secao_fontes_consultadas(resposta or "", fontes_unicas)
        resposta_final = _anexar_aviso_sessao(resposta, None)

    if usar_cache:
        cache_respostas.set(pergunta, resposta_final)
    return resposta_final


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL: CRAWLER
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_crawler(
    url_listagem: str,
    filtro_path: Optional[str] = None,
    sem_paginacao: bool = False,
) -> None:
    """Descobre todos os links da listagem e processa artigos + vídeos."""
    processadas = carregar_processadas()

    encontrados = descobrir_links(
        url_listagem,
        filtro_path=filtro_path,
        seguir_paginacao=not sem_paginacao,
    )
    artigos = [u for u in encontrados["artigos"] if u not in processadas]
    videos  = [u for u in encontrados["videos"]  if u not in processadas]
    pulados = (
        (len(encontrados["artigos"]) - len(artigos)) +
        (len(encontrados["videos"])  - len(videos))
    )

    print(f"\n📋 Novos: {len(artigos)} artigo(s) + {len(videos)} vídeo(s)"
          f"  |  {pulados} já processado(s)")

    if not artigos and not videos:
        print("✅ Tudo já indexado. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0
    total        = len(artigos) + len(videos)
    i            = 0

    for url in artigos:
        i += 1
        print(f"\n[{i}/{total}] 📄 {url}")
        n = processar_artigo(url, processadas, atualizar_catalogo=False)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    for url in videos:
        i += 1
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas, atualizar_catalogo=False)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            _pausa_entre_videos_no_lote()

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"  💾 Arquivos em: {PASTA_SAIDA}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS DA BASE
# ══════════════════════════════════════════════════════════════════════════════

def status_indice() -> None:
    _ensure_db()
    processadas   = carregar_processadas()
    arquivos_json = (
        [f for f in os.listdir(PASTA_SAIDA) if f.endswith(".json")]
        if os.path.exists(PASTA_SAIDA) else []
    )
    artigos_count = sum(1 for u in processadas if "youtube" not in u)
    videos_count  = sum(1 for u in processadas if "youtube" in u)
    dominios: dict = {}
    for u in processadas:
        if "youtube" not in u:
            d = urlparse(u).netloc
            dominios[d] = dominios.get(d, 0) + 1

    print(f"\n{'═'*55}")
    print("  📊 STATUS DA BASE DE CONHECIMENTO")
    print(f"{'═'*55}")
    print(f"  Pasta            : {PASTA_SAIDA}")
    print(f"  Banco vetorial   : {PASTA_CHROMA}")
    print(f"  Artigos indexados: {artigos_count}")
    print(f"  Vídeos YT        : {videos_count}")
    print(f"  Arquivos .json   : {len(arquivos_json)}")
    print(f"  Chunks no banco  : {colecao.count()}")
    print(f"  Modelo resposta  : {MODELO_POTENTE}")
    print(f"  Modelo rápido    : {MODELO_RAPIDO}")
    if dominios:
        print(f"\n  {'Domínio':<42} {'Artigos':>7}")
        print(f"  {'─'*50}")
        for d, n in sorted(dominios.items(), key=lambda x: -x[1]):
            print(f"  {d:<42} {n:>7}")
    catalogo = os.path.join(PASTA_SAIDA, "_CATALOGO.md")
    if os.path.exists(catalogo):
        import datetime
        ts = os.path.getmtime(catalogo)
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        print(f"\n  📋 Catálogo atualizado em: {dt}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MENU INTERATIVO
# ══════════════════════════════════════════════════════════════════════════════

def menu() -> None:
    L = 52  # largura do menu
    def _linha(c="═"): print(f"  {c * L}")
    def _titulo(t): print(f"  {t}")
    def _item(n, icone, label, detalhe=""):
        detalhe_fmt = f"  {detalhe}" if detalhe else ""
        print(f"    [{n}] {icone}  {label}{detalhe_fmt}")
    def _sep(): print(f"  {'─' * L}")

    while True:
        print()
        _linha()
        _titulo("🩺  APOIO CLÍNICO DR. AJUDA  |  Base de Conhecimento Médico")
        _linha()
        print()
        _titulo("  INDEXAR CONTEÚDO")
        _sep()
        _item("1", "📄", "Artigos / posts médicos por URL",  "cole várias URLs: linhas, espaço ou vírgula")
        _item("3", "🎬", "Vídeo(s) do YouTube (Dr. Ajuda)", "um ou vários (URL, ID ou arquivo .txt)")
        _item("4", "📂", "Arquivos locais",                  "processa .txt + .json da pasta de saída")
        _item("5", "🔍", "Buscar arquivo local",             "localiza por nome ou tema")
        _item("6", "🔄", "Forçar reindexação",               "reprocessa URL já indexada")
        print()
        _titulo("  CONSULTA CLÍNICA")
        _sep()
        _item("7", "💬", "Consulta (diagnóstico / conduta / laudo)")
        _item("8", "📊", "Status da base de conhecimento")
        print()
        _linha()
        _item("0", "🚪", "Sair")
        _linha()
        print()
        opcao = input("  Opção: ").strip()
        print()

        if opcao == "0":
            print("  Até logo!")
            break

        # ── 1. Artigos por URL (lote) ─────────────────────────────────────
        elif opcao == "1":
            print("  Cole as URLs dos artigos em qualquer formato:")
            print("    • uma por linha, ou várias na mesma linha (espaço / vírgula), ou bloco com links;")
            print("    • ou caminho de um .txt com uma URL por linha.")
            print("  ── linha vazia para confirmar ──")
            entradas: List[str] = []
            while True:
                linha = input("  > ").strip()
                if not linha:
                    break
                entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            urls_lote = _resolver_entradas_para_urls_artigos(entradas, run_id="menu-artigos")

            try:
                pipeline_artigos_em_lote(urls_lote)
            except Exception as e:
                print(f"  ❌ Erro ao processar lote de artigos: {e}")

        # ── 3. Vídeo(s) do YouTube ────────────────────────────────────────
        elif opcao == "3" or _parece_youtube(opcao):
            # Permite colar a URL/ID direto como opção do menu
            if _parece_youtube(opcao):
                entradas = [opcao]
            else:
                print("  Cole links ou IDs do YouTube (linha única com vários, ou uma por linha,")
                print("  ou caminho de um .txt com uma entrada por linha).")
                print("  ── linha vazia para confirmar ──")
                entradas = []
                while True:
                    linha = input("  > ").strip()
                    if not linha:
                        break
                    entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            # Se for um único caminho de arquivo, lê o arquivo
            urls_lote = _resolver_entradas_para_urls_video(entradas, run_id="menu-video")

            if len(urls_lote) == 1:
                url = _normalizar_url_yt(urls_lote[0])
                processadas = carregar_processadas()
                n = processar_video(url, processadas)
                _exibir_resultado_avulso(n)
                print(f"  📦 Total de chunks no banco: {colecao.count()}")
            else:
                pipeline_videos_em_lote(urls_lote)

        # ── 4. Arquivos locais ────────────────────────────────────────────
        elif opcao == "4":
            pipeline_processar_pasta()

        # ── 5. Buscar arquivo local ───────────────────────────────────────
        elif opcao == "5":
            pipeline_indexar_video_por_nome()

        # ── 6. Forçar reindexação ─────────────────────────────────────────
        elif opcao == "6":
            url = input("  URL para reindexar: ").strip()
            processadas = carregar_processadas()
            processadas.discard(url)
            with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
                json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
            url = _normalizar_url_yt(url)
            if _eh_url_youtube(url):
                n = processar_video(url, processadas)
            elif _eh_url(url):
                n = processar_artigo(url, processadas)
            else:
                print("  ❌ URL inválida.")
                continue
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")

        # ── 7. Consulta clínica (sessão contínua) ─────────────────────────
        elif opcao == "7":
            loop_consulta_cli(SessaoConsultaCLI())

        # ── 8. Status da base ─────────────────────────────────────────────
        elif opcao == "8":
            status_indice()

        else:
            print("  ❌ Opção inválida. Tente novamente.")


def _intro_conversa() -> None:
    print("\n  Dr. Ajuda · consulta e indexação")
    print("  Dúvidas clínicas em linguagem natural · /ajuda · /sair\n")


_sessao_conversa_cli = SessaoConsultaCLI()


def loop_consulta_cli(sessao: Optional[SessaoConsultaCLI] = None) -> None:
    """Loop de consulta clínica contínua (menu, --pergunta ou modo conversa)."""
    s = sessao or SessaoConsultaCLI()

    while True:
        try:
            pergunta = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not pergunta:
            continue

        low = pergunta.lower()
        if low in ("/sair", "/quit", "/q", "sair", "exit", "quit"):
            return
        if low in ("/nova", "/limpar", "/reset"):
            s.limpar()
            continue

        resposta = pipeline_perguntar(pergunta, sessao=s, modo_conversa=True)
        _imprimir_resposta_cli(resposta)
        s.registrar(pergunta, resposta)


def _rotear_mensagem_conversa(msg: str) -> dict:
    msg = (msg or "").strip()
    if not msg:
        return {"acao": "perguntar_clarificacao", "pergunta": "O que você gostaria de fazer?"}
    low = msg.lower().strip()
    if low in ("sair", "exit", "quit", "q"):
        return {"acao": "sair"}
    if low in ("ajuda", "help", "h", "?", "/ajuda", "/help"):
        return {"acao": "ajuda"}

    # Fallback heurístico quando IA não estiver disponível
    if not GROQ_ENABLED:
        if re.search(r"\bstatus\b|\bestat[ií]stic", low):
            return {"acao": "status"}
        if re.search(r"\breindex", low):
            m = re.search(r"(https?://\S+)", msg)
            return {"acao": "reindexar", "url": (m.group(1) if m else "")}
        if _parece_youtube(msg):
            return {"acao": "videos_lote", "entradas": [msg]}
        if re.search(r"https?://", msg) and not _parece_youtube(msg):
            return {"acao": "artigos_lote", "entradas": [msg]}
        if re.search(r"\bcrawl\b|\bvarrer\b|\bcrawlear\b", low):
            m = re.search(r"(https?://\S+)", msg)
            return {"acao": "crawl_site", "url_listagem": (m.group(1) if m else ""), "filtro_path": None, "sem_paginacao": False}
        if re.search(r"\blocal\b|\barquivos\b", low):
            return {"acao": "processar_local"}
        return {"acao": "perguntar_clarificacao", "pergunta": "Configure o GROQ_API_KEY para consulta clínica. Para indexação, cole URLs/IDs."}

    try:
        raw = chamar_groq(
            sistema=PROMPT_ROUTER_CONVERSA_SISTEMA,
            usuario=f"Mensagem do usuário:\n{msg}\n",
            modelo=MODELO_RAPIDO,
            json_mode=True,
        )
        d = json.loads(raw)
        if isinstance(d, dict) and d.get("acao"):
            return d
    except Exception:
        pass

    return {"acao": "consultar", "pergunta_clinica": msg}


def conversa() -> None:
    _intro_conversa()
    while True:
        try:
            msg = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not msg:
            continue

        low = msg.lower()
        if low in ("/sair", "/quit", "/q", "sair", "exit", "quit"):
            return
        if low in ("/nova", "/limpar", "/reset"):
            _sessao_conversa_cli.limpar()
            continue

        r = _rotear_mensagem_conversa(msg)
        acao = r.get("acao")

        if acao == "sair":
            return

        if acao == "ajuda":
            _intro_conversa()
            continue

        if acao == "status":
            status_indice()
            continue

        if acao == "consultar":
            pergunta = (r.get("pergunta_clinica") or "").strip()
            if not pergunta:
                continue
            resposta = pipeline_perguntar(
                pergunta, sessao=_sessao_conversa_cli, modo_conversa=True
            )
            _imprimir_resposta_cli(resposta)
            _sessao_conversa_cli.registrar(pergunta, resposta)
            continue

        if acao == "processar_local":
            pipeline_processar_pasta()
            continue

        if acao == "buscar_local":
            consulta = (r.get("consulta") or "").strip()
            if consulta:
                # reutiliza o pipeline existente (interativo) como fallback;
                # quando há consulta, tentamos passar pela variável global via prompt ao usuário.
                print("  Vou abrir a busca local. Se necessário, cole/ajuste a descrição.")
            pipeline_indexar_video_por_nome()
            continue

        if acao == "artigos_lote":
            entradas = r.get("entradas") or []
            if isinstance(entradas, str):
                entradas = [entradas]
            if not entradas:
                entradas = [msg]
            urls = _resolver_entradas_para_urls_artigos(entradas, run_id="conversa-artigos")
            if not urls:
                print("  Não encontrei URLs de artigos válidas nessa mensagem. Cole links http(s) ou um arquivo .txt.")
                continue
            pipeline_artigos_em_lote(urls)
            continue

        if acao == "videos_lote":
            entradas = r.get("entradas") or []
            if isinstance(entradas, str):
                entradas = [entradas]
            if not entradas:
                entradas = [msg]
            urls = _resolver_entradas_para_urls_video(entradas, run_id="conversa-video")
            if not urls:
                print("  Não encontrei URLs/IDs de YouTube válidos nessa mensagem. Cole um link/ID ou um arquivo .txt.")
                continue
            if len(urls) == 1:
                url = _normalizar_url_yt(urls[0])
                processadas = carregar_processadas()
                print(f"\n  🎬 Indexando vídeo: {url}")
                n = processar_video(url, processadas)
                _exibir_resultado_avulso(n)
                print(f"  📦 Total de chunks no banco: {colecao.count()}")
            else:
                pipeline_videos_em_lote(urls)
            continue

        if acao == "crawl_site":
            url_listagem = (r.get("url_listagem") or "").strip()
            if not url_listagem:
                print("  Qual é a URL base/listagem do site para crawl?")
                continue
            filtro_path = r.get("filtro_path")
            sem_paginacao = bool(r.get("sem_paginacao", False))
            pipeline_crawler(url_listagem, filtro_path=filtro_path, sem_paginacao=sem_paginacao)
            continue

        if acao == "reindexar":
            url = (r.get("url") or "").strip()
            if not url:
                print("  Qual URL você quer reindexar?")
                continue
            url = _normalizar_url_yt(url)
            processadas = carregar_processadas()
            processadas.discard(url)
            with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
                json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
            print(f"\n  🔄 Reindexando: {url}")
            if _eh_url_youtube(url):
                n = processar_video(url, processadas)
            elif _eh_url(url):
                n = processar_artigo(url, processadas)
            else:
                print("  ❌ URL inválida.")
                continue
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")
            continue

        if acao == "perguntar_clarificacao":
            pergunta = (r.get("pergunta") or "Como posso ajudar?").strip()
            print(f"  {pergunta}")
            continue

        print("  Não entendi. Digite 'ajuda' para ver exemplos.")


def _exibir_resultado_avulso(n: int) -> None:
    if n < 0:
        print("   ❌ Falha ao processar.")
    elif n == 0:
        print("   ℹ️  Já estava indexado.")
    else:
        print(f"   ✅ {n} chunks indexados.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _main(argv: list) -> None:
    args = argv[1:]

    # ── Sem argumentos: modo conversa (minimalista) ───────────────────────
    if not args:
        conversa()
        return

    # ── --ajuda / --help ────────────────────────────────────────────────
    if args[0] in ("-h", "--help", "--ajuda"):
        print(__doc__)
        return

    # ── --menu ───────────────────────────────────────────────────────────
    if args[0] in ("--menu", "-m"):
        menu()
        return

    # ── --status ─────────────────────────────────────────────────────────
    if args[0] == "--status":
        status_indice()
        return

    # ── --pergunta ───────────────────────────────────────────────────────
    if args[0] == "--pergunta":
        resto = list(args[1:])
        once = False
        if "--once" in resto or "-1" in resto:
            once = True
            resto = [a for a in resto if a not in ("--once", "-1")]
        pergunta = " ".join(resto).strip()
        sessao = SessaoConsultaCLI()
        if pergunta:
            resposta = pipeline_perguntar(pergunta, sessao=sessao, modo_conversa=True)
            _imprimir_resposta_cli(resposta)
            sessao.registrar(pergunta, resposta)
        if once and pergunta:
            return
        if not pergunta and once:
            sys.exit('  Use: --pergunta "sua consulta clínica aqui"')
        loop_consulta_cli(sessao)
        return

    # ── --artigo / --artigos / URL direta (somente links informados) ─────
    if args[0] in ("--artigo", "--artigos", "--crawl", "--site") or (args and _eh_url(args[0]) and not _eh_url_youtube(args[0])):
        entradas = args[1:] if args[0] in ("--artigo", "--artigos", "--crawl", "--site") else args
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --artigos "https://site.com/post-1" "https://site.com/post-2"\n'
                "    --artigos lista.txt"
            )

        urls = _resolver_entradas_para_urls_artigos(entradas, run_id="cli-artigos")
        if not urls:
            sys.exit("  Nenhuma URL de artigo válida encontrada (use http(s):// ou arquivo .txt).")

        pipeline_artigos_em_lote(urls)
        return

    # ── --video  (único, vários ou arquivo .txt) ──────────────────────────
    if args[0] in ("--video", "--videos"):
        entradas = args[1:]
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --video "https://youtube.com/watch?v=ID"\n'
                '    --video "url1" "url2" "url3"\n'
                "    --video lista.txt"
            )
        # Se for um único argumento que é um arquivo existente → lê o arquivo
        urls = _resolver_entradas_para_urls_video(entradas, run_id="cli-video")
        if not urls:
            sys.exit("  Nenhuma URL/ID de YouTube válida encontrada.")

        if len(urls) == 1:
            url = _normalizar_url_yt(urls[0])
            processadas = carregar_processadas()
            print(f"\n  🎬 Indexando vídeo: {url}")
            n = processar_video(url, processadas)
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")
        else:
            pipeline_videos_em_lote(urls)
        return

    # ── --local ───────────────────────────────────────────────────────────
    if args[0] in ("--local", "--pasta", "-p"):
        pipeline_processar_pasta()
        return

    # ── --gerar-json-locais (companheiros para .txt sem .json) ─────────────
    if args[0] in ("--gerar-json-locais", "--json-locais"):
        sobrescrever = "--sobrescrever" in args[1:] or "-f" in args[1:]
        pipeline_gerar_json_para_txts_sem_json(sobrescrever_json=sobrescrever)
        return

    # ── --buscar-local ────────────────────────────────────────────────────
    if args[0] in ("--buscar-local", "--indexar-video"):
        pipeline_indexar_video_por_nome()
        return

    # ── --relatorio ───────────────────────────────────────────────────────
    if args[0] in ("--relatorio", "--relatório", "--report"):
        relatorio_pasta()
        return

    # ── --reindexar ───────────────────────────────────────────────────────
    if args[0] == "--reindexar":
        if len(args) < 2:
            sys.exit('  Use: --reindexar "https://..."')
        url = _normalizar_url_yt(args[1])
        processadas = carregar_processadas()
        processadas.discard(url)
        with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
            json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
        print(f"\n  🔄 Reindexando: {url}")
        if _eh_url_youtube(url):
            n = processar_video(url, processadas)
        elif _eh_url(url):
            n = processar_artigo(url, processadas)
        else:
            sys.exit(f"  ❌ URL inválida: {url}")
        _exibir_resultado_avulso(n)
        print(f"  📦 Total de chunks no banco: {colecao.count()}")
        return

    # ── Argumento não reconhecido ─────────────────────────────────────────
    sys.exit(
        f"  ❌ Argumento não reconhecido: '{args[0]}'\n"
        f"  Use --ajuda para ver todas as opções."
    )


if __name__ == "__main__":
    _main(sys.argv)