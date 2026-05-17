"""
reindexar_pdfs.py — Reindexação de PDFs no sistema sup (super.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problemas corrigidos:
  1. PDFs entrando como binário bruto (%PDF-1.3 ...) — conteúdo ilegível.
  2. PDFs com fontes embutidas gerando sequências (cid:N) no lugar de letras
     (ex: "(cid:70)(cid:82)(cid:80)" no lugar de "com").

Solução:
  Este script baixa os PDFs, extrai o texto corretamente, decodifica
  sequências (cid:N), remove entradas corrompidas do ChromaDB e reindexe
  tudo no formato padrão do super.py.

INSTALAÇÃO
  pip install pdfplumber pymupdf requests

USO
  # Reindexar todos os PDFs de uma lista de URLs:
  python reindexar_pdfs.py --urls urls.txt

  # Reindexar uma URL única:
  python reindexar_pdfs.py --url "https://bvsms.saude.gov.br/bvs/publicacoes/glossario.pdf"

  # Reindexar todos os .txt corrompidos já salvos na pasta do sistema:
  python reindexar_pdfs.py --pasta

  # Ver quais arquivos estão corrompidos (binário) sem reindexar:
  python reindexar_pdfs.py --diagnostico

  # Forçar reindexação mesmo de PDFs já processados:
  python reindexar_pdfs.py --urls urls.txt --forcar

VARIÁVEIS DE AMBIENTE (mesmas do super.py)
  RAG_PASTA_SAIDA      pasta de saída (padrão: ~/Documentos/rag_base)
  RAG_PASTA_CHROMA     pasta do ChromaDB
  GROQ_API_KEY         chave da API Groq (para metadados e limpeza)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import json
import re
import time
import hashlib
import argparse
from datetime import date
from typing import Optional
from urllib.parse import urlparse

# ── Dependências ──────────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    sys.exit("❌ Instale: pip install requests")

# Extração de texto de PDF — tenta pdfplumber primeiro, pymupdf como fallback
_PDF_BACKEND = None
try:
    import pdfplumber
    _PDF_BACKEND = "pdfplumber"
except ImportError:
    pass

if _PDF_BACKEND is None:
    try:
        import fitz  # pymupdf
        _PDF_BACKEND = "pymupdf"
    except ImportError:
        pass

if _PDF_BACKEND is None:
    sys.exit(
        "❌ Nenhum backend de PDF encontrado.\n"
        "   Instale: pip install pdfplumber\n"
        "   ou:      pip install pymupdf"
    )

print(f"✅ Backend PDF: {_PDF_BACKEND}", file=sys.stderr)

# ── Importar super.py ─────────────────────────────────────────────────────────
# O super.py deve estar na mesma pasta ou no PYTHONPATH
try:
    import importlib.util, pathlib

    # Tenta encontrar super.py na mesma pasta do script ou no diretório atual
    _candidatos = [
        pathlib.Path(__file__).parent / "super.py",
        pathlib.Path.cwd() / "super.py",
    ]
    _super_path = next((p for p in _candidatos if p.exists()), None)

    if _super_path is None:
        raise FileNotFoundError("super.py não encontrado")

    spec = importlib.util.spec_from_file_location("super", _super_path)
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    print(f"✅ super.py carregado: {_super_path}", file=sys.stderr)

except Exception as e:
    sys.exit(
        f"❌ Não foi possível importar super.py: {e}\n"
        f"   Coloque este script na mesma pasta que o super.py."
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

PASTA_SAIDA         = sup.PASTA_SAIDA
ARQUIVO_PROCESSADAS = sup.ARQUIVO_PROCESSADAS
TIMEOUT             = getattr(sup, "TIMEOUT", 30)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
#  EXTRAÇÃO DE TEXTO DOS PDFs
# ══════════════════════════════════════════════════════════════════════════════

def extrair_texto_pdf_bytes(dados: bytes, url: str = "") -> Optional[str]:
    """
    Extrai texto legível de bytes de PDF.
    Tenta pdfplumber primeiro, cai para pymupdf se necessário.
    Retorna None se o PDF estiver vazio ou apenas com imagens (sem texto).
    """
    import io

    texto_paginas = []

    # ── pdfplumber ────────────────────────────────────────────────────────
    if _PDF_BACKEND == "pdfplumber":
        try:
            with pdfplumber.open(io.BytesIO(dados)) as pdf:
                for i, pagina in enumerate(pdf.pages):
                    t = pagina.extract_text()
                    if t and t.strip():
                        texto_paginas.append(t.strip())
        except Exception as e:
            print(f"   ⚠️  pdfplumber falhou ({e}), tentando pymupdf...")
            return _extrair_com_pymupdf(dados)

    # ── pymupdf ───────────────────────────────────────────────────────────
    elif _PDF_BACKEND == "pymupdf":
        return _extrair_com_pymupdf(dados)

    if not texto_paginas:
        return None

    texto = "\n\n".join(texto_paginas)
    texto = _limpar_texto_pdf(texto)
    return texto if texto.strip() else None


def _extrair_com_pymupdf(dados: bytes) -> Optional[str]:
    """Extrai texto via pymupdf (fitz)."""
    import io
    try:
        doc = fitz.open(stream=io.BytesIO(dados), filetype="pdf")
        paginas = []
        for pagina in doc:
            t = pagina.get_text()
            if t and t.strip():
                paginas.append(t.strip())
        doc.close()
        if not paginas:
            return None
        return _limpar_texto_pdf("\n\n".join(paginas))
    except Exception as e:
        print(f"   ⚠️  pymupdf falhou: {e}")
        return None


def _decodificar_cid(texto: str) -> str:
    """
    Converte sequências (cid:N) para o caractere Unicode correspondente.

    Esse problema ocorre em PDFs com fontes embutidas com encoding proprietário,
    onde o pdfplumber não consegue mapear os glifos corretamente.
    Exemplo: "(cid:70)(cid:82)(cid:80)" → "Fro" (parte de "From")

    Se após a decodificação o texto ainda estiver ilegível, o PDF usa
    encoding completamente proprietário e vai precisar de OCR.
    """
    def substituir(m):
        codigo = int(m.group(1))
        try:
            return chr(codigo)
        except (ValueError, OverflowError):
            return ""

    resultado = re.sub(r"\(cid:(\d+)\)", substituir, texto)
    # Limpa espaços extras que sobram após a substituição
    resultado = re.sub(r" {2,}", " ", resultado)
    return resultado


def _tem_muitos_cid(texto: str) -> bool:
    """Verifica se o texto tem quantidade significativa de sequências (cid:N)."""
    ocorrencias = len(re.findall(r"\(cid:\d+\)", texto))
    return ocorrencias > 10


def _limpar_texto_pdf(texto: str) -> str:
    """
    Limpeza completa do texto extraído de PDF:
    - Decodifica sequências (cid:N) de fontes embutidas com encoding proprietário
    - Remove hifenização de final de linha (pa-\\nlavra → palavra)
    - Remove quebras de linha simples dentro de parágrafos
    - Normaliza espaços múltiplos
    - Preserva quebras duplas (separação de parágrafos)
    """
    # Decodifica (cid:N) antes de qualquer outra limpeza
    if _tem_muitos_cid(texto):
        texto = _decodificar_cid(texto)

    # Remove hifenização (palavra-\n → palavra)
    texto = re.sub(r"-\n(\w)", r"\1", texto)

    # Une linhas simples dentro do mesmo parágrafo
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)

    # Normaliza espaços múltiplos
    texto = re.sub(r" {2,}", " ", texto)

    # Normaliza quebras múltiplas (mais de 2 → 2)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def _eh_binario_pdf(conteudo: str) -> bool:
    """
    Verifica se o conteúdo de um arquivo .txt é na verdade binário de PDF
    (ou seja, foi salvo incorretamente com o conteúdo cru do arquivo PDF).
    """
    sinais = [
        conteudo.strip().startswith("%PDF"),
        "endobj" in conteudo[:500],
        "xref" in conteudo[:500],
        "startxref" in conteudo[:500],
        # Presença de caracteres de controle ou alta densidade de bytes não-ASCII
        sum(1 for c in conteudo[:200] if ord(c) > 127) > 30,
    ]
    return any(sinais[:3]) or (sinais[4] and any(sinais[:4]))


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def baixar_pdf(url: str) -> Optional[bytes]:
    """Baixa um PDF da URL e retorna os bytes brutos. Retorna None em falha."""
    print(f"   ⬇️  Baixando: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
            print(f"   ⚠️  Content-Type inesperado: {ct}")

        if len(r.content) < 100:
            print("   ⚠️  Resposta muito pequena — possível erro do servidor")
            return None

        return r.content

    except requests.RequestException as e:
        print(f"   ❌ Falha no download: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  REMOÇÃO DE ENTRADAS CORROMPIDAS DO CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def _slug(url: str) -> str:
    """Mesmo algoritmo de slug do super.py."""
    return sup.slugify(url)[:80]


def remover_chunks_corrompidos_do_chroma(url: str) -> int:
    """
    Remove todos os chunks de uma URL do ChromaDB.
    Necessário antes de reindexar para não manter dados duplicados/corrompidos.
    Retorna a quantidade de chunks removidos.
    """
    try:
        colecao = sup.colecao
        slug = _slug(url)

        # Busca todos os IDs que pertencem a esta URL via prefixo do slug
        resultado = colecao.get(where={"url": url})
        ids = resultado.get("ids", [])

        if not ids:
            # Tenta via prefixo do slug (fallback)
            todos = colecao.get()
            ids = [
                doc_id for doc_id in (todos.get("ids") or [])
                if doc_id.startswith(slug)
            ]

        if ids:
            colecao.delete(ids=ids)
            print(f"   🗑️  {len(ids)} chunks removidos do ChromaDB")
            return len(ids)

        return 0

    except Exception as e:
        print(f"   ⚠️  Erro ao remover do ChromaDB: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  REINDEXAÇÃO DE UM PDF
# ══════════════════════════════════════════════════════════════════════════════

def reindexar_pdf_url(url: str, *, forcar: bool = False) -> bool:
    """
    Pipeline completo para um PDF:
    1. Verifica se já foi processado (skip se não forçar)
    2. Remove chunks corrompidos do ChromaDB
    3. Baixa o PDF
    4. Extrai texto limpo
    5. Gera metadados via Groq
    6. Indexa no ChromaDB
    7. Salva .txt e .json corretos na pasta

    Retorna True em sucesso.
    """
    print(f"\n{'─'*60}")
    print(f"📄 PDF: {url}")

    processadas = sup.carregar_processadas()

    # Verifica se já está processado
    if url in processadas and not forcar:
        print("   ⏭️  Já processado. Use --forcar para reindexar.")
        return True

    # Remove do registro de processadas para permitir reindexação
    processadas.discard(url)

    # Remove chunks corrompidos ou antigos do ChromaDB
    removidos = remover_chunks_corrompidos_do_chroma(url)

    # Baixa o PDF
    dados_pdf = baixar_pdf(url)
    if dados_pdf is None:
        print("   ❌ Falha no download — pulando.")
        return False

    # Extrai texto
    print("   📝 Extraindo texto do PDF...")
    texto = extrair_texto_pdf_bytes(dados_pdf, url)

    if not texto or len(texto.strip()) < 100:
        print(
            f"   ❌ Texto extraído insuficiente ({len(texto or '')} chars).\n"
            f"      Possível PDF escaneado (só imagem) — necessita OCR."
        )
        return False

    # Avisa se ainda restam sequências (cid:N) após decodificação
    cid_restantes = len(re.findall(r"\(cid:\d+\)", texto))
    if cid_restantes > 20:
        print(
            f"   ⚠️  {cid_restantes} sequências (cid:N) ainda presentes após decodificação.\n"
            f"      O PDF usa encoding proprietário não mapeável — considere OCR para melhor resultado."
        )

    print(f"   ✅ Texto extraído: {len(texto):,} chars")

    # Título a partir da URL
    nome_arquivo = urlparse(url).path.split("/")[-1].replace(".pdf", "").replace("_", " ").replace("-", " ").title()
    titulo = nome_arquivo or "Documento PDF"

    # Gera metadados via Groq
    print("   🤖 Gerando metadados...")
    meta_ia = sup.gerar_metadados(titulo, texto, "artigo")

    # Usa o resumo gerado como título se for mais descritivo
    tema = meta_ia.get("tema_principal", "").strip()
    if tema and len(tema) > len(titulo):
        titulo = tema

    dominio = urlparse(url).netloc

    # Indexa no ChromaDB
    print("   📦 Indexando no ChromaDB...")
    n_chunks = sup.indexar_conteudo(
        conteudo=texto,
        titulo=titulo,
        url=url,
        tipo="artigo",
        meta_extra={"dominio": dominio},
        meta_ia=meta_ia,
    )

    if n_chunks <= 0:
        print("   ❌ Falha na indexação.")
        return False

    print(f"   ✅ {n_chunks} chunks indexados")

    # Salva .txt e .json na pasta do sistema
    sup.salvar_arquivos(
        titulo=titulo,
        url=url,
        conteudo=texto,
        meta_ia=meta_ia,
        tipo="artigo",
        extra={"dominio": dominio},
    )

    # Marca como processada
    sup.marcar_processada(url, processadas)
    print(f"   💾 Arquivos salvos na pasta do sistema")

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CORREÇÃO DE ARQUIVOS .TXT CORROMPIDOS JÁ SALVOS NA PASTA
# ══════════════════════════════════════════════════════════════════════════════

def reindexar_txts_corrompidos_na_pasta(*, forcar: bool = False) -> None:
    """
    Varre PASTA_SAIDA procurando arquivos .txt que contêm binário de PDF
    (salvos incorretamente), e reindexe cada um baixando novamente o PDF
    e extraindo o texto corretamente.
    """
    if not os.path.isdir(PASTA_SAIDA):
        print(f"❌ Pasta não encontrada: {PASTA_SAIDA}")
        return

    txts = [f for f in os.listdir(PASTA_SAIDA) if f.endswith(".txt")]
    if not txts:
        print("Nenhum .txt encontrado na pasta.")
        return

    corrompidos = []

    print(f"\n🔍 Verificando {len(txts)} arquivo(s) em: {PASTA_SAIDA}\n")

    for nome_txt in sorted(txts):
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, nome_txt[:-4] + ".json")

        try:
            with open(caminho_txt, encoding="utf-8", errors="replace") as f:
                conteudo = f.read(2000)  # Lê só o início para verificar
        except OSError:
            continue

        # Extrai apenas o conteúdo após [CONTEÚDO]
        if "[CONTEÚDO]" in conteudo:
            idx = conteudo.index("[CONTEÚDO]") + len("[CONTEÚDO]")
            trecho = conteudo[idx:].strip()
        else:
            trecho = conteudo

        url = _extrair_url_do_arquivo(caminho_txt, caminho_json)
        status = f"URL: {url}" if url else "URL não encontrada"

        if _eh_binario_pdf(trecho):
            corrompidos.append((nome_txt, url))
            print(f"   ❌ Binário PDF : {nome_txt} — {status}")
        elif _tem_muitos_cid(trecho):
            corrompidos.append((nome_txt, url))
            print(f"   ⚠️  CID (cid:N): {nome_txt} — {status}")
        else:
            print(f"   ✅ OK          : {nome_txt}")

    if not corrompidos:
        print("\n✅ Nenhum arquivo corrompido encontrado!")
        return

    print(f"\n\n{'═'*60}")
    print(f"  {len(corrompidos)} arquivo(s) a corrigir (binário + CID)")
    print(f"{'═'*60}\n")

    sucesso = 0
    falha   = 0

    for nome_txt, url in corrompidos:
        if not url:
            print(f"\n⚠️  {nome_txt}: URL não encontrada — impossível reindexar automaticamente.")
            print(f"   Adicione a URL manualmente e use: python reindexar_pdfs.py --url <URL>")
            falha += 1
            continue

        ok = reindexar_pdf_url(url, forcar=True)
        if ok:
            # Remove os arquivos antigos corrompidos
            caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
            caminho_json = os.path.join(PASTA_SAIDA, nome_txt[:-4] + ".json")
            for f in [caminho_txt, caminho_json]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"   🗑️  Arquivo antigo removido: {os.path.basename(f)}")
            sucesso += 1
        else:
            falha += 1

        time.sleep(1.5)  # Pausa entre downloads

    print(f"\n{'═'*60}")
    print(f"  ✅ Reindexados com sucesso : {sucesso}")
    print(f"  ❌ Falhas                  : {falha}")
    print(f"{'═'*60}")

    sup.salvar_catalogo()


def _extrair_url_do_arquivo(caminho_txt: str, caminho_json: str) -> Optional[str]:
    """Tenta extrair a URL do cabeçalho do .txt ou do .json."""
    # Tenta do .json primeiro
    if os.path.exists(caminho_json):
        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
            url = meta.get("url", "")
            if url and url.startswith("http"):
                return url
        except Exception:
            pass

    # Tenta do cabeçalho do .txt
    try:
        with open(caminho_txt, encoding="utf-8", errors="replace") as f:
            for linha in f:
                if linha.startswith("URL:"):
                    url = linha.split(":", 1)[1].strip()
                    if url.startswith("http"):
                        return url
                if "[CONTEÚDO]" in linha:
                    break  # Chegou no conteúdo, para de procurar
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO (sem modificar nada)
# ══════════════════════════════════════════════════════════════════════════════

def diagnostico() -> None:
    """
    Exibe quais arquivos na pasta estão com conteúdo binário de PDF.
    Não modifica nenhum arquivo.
    """
    if not os.path.isdir(PASTA_SAIDA):
        print(f"❌ Pasta não encontrada: {PASTA_SAIDA}")
        return

    txts = sorted(f for f in os.listdir(PASTA_SAIDA) if f.endswith(".txt"))
    if not txts:
        print("Nenhum .txt encontrado.")
        return

    print(f"\n{'═'*60}")
    print(f"  DIAGNÓSTICO — {PASTA_SAIDA}")
    print(f"{'═'*60}\n")

    corrompidos = []
    com_cid     = []
    ok_count    = 0

    for nome in txts:
        caminho = os.path.join(PASTA_SAIDA, nome)
        try:
            with open(caminho, encoding="utf-8", errors="replace") as f:
                conteudo = f.read(2000)
        except OSError:
            continue

        if "[CONTEÚDO]" in conteudo:
            trecho = conteudo[conteudo.index("[CONTEÚDO]") + 10:].strip()
        else:
            trecho = conteudo

        if _eh_binario_pdf(trecho):
            url = _extrair_url_do_arquivo(
                caminho,
                os.path.join(PASTA_SAIDA, nome[:-4] + ".json"),
            )
            corrompidos.append((nome, url))
        elif _tem_muitos_cid(trecho):
            url = _extrair_url_do_arquivo(
                caminho,
                os.path.join(PASTA_SAIDA, nome[:-4] + ".json"),
            )
            com_cid.append((nome, url))
        else:
            ok_count += 1

    print(f"  ✅ Arquivos OK              : {ok_count}")
    print(f"  ❌ Binário bruto (PDF cru)  : {len(corrompidos)}")
    print(f"  ⚠️  Com sequências (cid:N)  : {len(com_cid)}")
    print()

    if corrompidos:
        print("  Arquivos com binário de PDF:")
        for nome, url in corrompidos:
            url_str = url or "⚠️  URL não encontrada"
            print(f"    • {nome}")
            print(f"      → {url_str}")
        print()

    if com_cid:
        print("  Arquivos com sequências (cid:N) — fontes embutidas não mapeadas:")
        for nome, url in com_cid:
            url_str = url or "⚠️  URL não encontrada"
            print(f"    • {nome}")
            print(f"      → {url_str}")
        print()

    if corrompidos or com_cid:
        print("  Para corrigir tudo, execute:")
        print("    python reindexar_pdfs.py --pasta")


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DE LISTA DE URLs
# ══════════════════════════════════════════════════════════════════════════════

def reindexar_lista(urls: list, *, forcar: bool = False) -> None:
    """Reindexe uma lista de URLs de PDF."""
    if not urls:
        print("Nenhuma URL fornecida.")
        return

    print(f"\n🚀 Reindexando {len(urls)} PDF(s)...\n")
    sucesso = 0
    falha   = 0

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or not url.startswith("http"):
            print(f"⚠️  URL inválida ignorada: {url!r}")
            continue

        print(f"\n[{i}/{len(urls)}]")
        ok = reindexar_pdf_url(url, forcar=forcar)
        if ok:
            sucesso += 1
        else:
            falha += 1

        if i < len(urls):
            time.sleep(1.5)

    sup.salvar_catalogo()

    print(f"\n{'═'*60}")
    print(f"  ✅ Sucesso : {sucesso}/{len(urls)}")
    print(f"  ❌ Falha   : {falha}/{len(urls)}")
    print(f"  📦 Total chunks no banco: {sup.colecao.count()}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reindexação de PDFs no sistema sup (super.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url",
        metavar="URL",
        help='URL de um único PDF para reindexar (ex.: "https://bvsms.saude.gov.br/.../glossario.pdf")',
    )
    parser.add_argument(
        "--urls",
        metavar="ARQUIVO_OU_LISTA",
        help='Arquivo .txt com uma URL por linha, ou URLs separadas por espaço',
        nargs="+",
    )
    parser.add_argument(
        "--pasta",
        action="store_true",
        help="Varre a pasta do sistema e corrige todos os .txt com conteúdo binário de PDF",
    )
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Mostra quais arquivos estão corrompidos, sem modificar nada",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Reindexar mesmo que a URL já esteja marcada como processada",
    )

    args = parser.parse_args()

    # ── Sem argumentos → exibe ajuda ─────────────────────────────────────
    if not any([args.url, args.urls, args.pasta, args.diagnostico]):
        parser.print_help()
        return

    # ── Diagnóstico ───────────────────────────────────────────────────────
    if args.diagnostico:
        diagnostico()
        return

    # ── Corrigir pasta ────────────────────────────────────────────────────
    if args.pasta:
        reindexar_txts_corrompidos_na_pasta(forcar=args.forcar)
        return

    # ── URL única ─────────────────────────────────────────────────────────
    if args.url:
        ok = reindexar_pdf_url(args.url.strip(), forcar=args.forcar)
        sup.salvar_catalogo()
        sys.exit(0 if ok else 1)

    # ── Lista de URLs ─────────────────────────────────────────────────────
    if args.urls:
        urls = []
        for entrada in args.urls:
            if entrada.startswith("http"):
                urls.append(entrada)
            elif os.path.isfile(entrada):
                with open(entrada, encoding="utf-8") as f:
                    urls.extend(
                        linha.strip()
                        for linha in f
                        if linha.strip() and linha.strip().startswith("http")
                    )
            else:
                print(f"⚠️  Entrada ignorada (não é URL nem arquivo): {entrada!r}")

        reindexar_lista(urls, forcar=args.forcar)


if __name__ == "__main__":
    main()