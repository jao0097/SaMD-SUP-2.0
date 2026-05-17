"""
atualizar_jsons_apos_limpeza.py
================================
Atualiza os JSONs após a limpeza das transcrições.

O que faz:
  ✅ Lê cada .txt limpo
  ✅ Atualiza tamanho_chars com o novo tamanho
  ✅ Marca indexado_chroma como false (precisa reindexar)
  ✅ Opcionalmente adiciona o conteudo limpo ao JSON
  ✅ Processa em paralelo para 979+ arquivos

Uso:
  python atualizar_jsons_apos_limpeza.py /home/joao/Documentos/rag_base/
  python atualizar_jsons_apos_limpeza.py /caminho/ --com-conteudo
  python atualizar_jsons_apos_limpeza.py /caminho/ --workers 8
  python atualizar_jsons_apos_limpeza.py /caminho/ --relatorio-apenas

Flags:
  --com-conteudo       adiciona o conteudo limpο ao JSON
  --workers N          número de workers paralelos (padrão: 4)
  --relatorio-apenas   só mostra relatório sem atualizar
  --limpar-cache       força reprocessamento
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm não instalado. Install com: pip install tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Worker para atualizar JSON
# ---------------------------------------------------------------------------

def atualizar_json_worker(args_tuple: Tuple) -> Dict:
    """Worker function para ProcessPoolExecutor."""
    arquivo_json, pasta, com_conteudo, cache_file = args_tuple
    resultado = {
        "arquivo": arquivo_json.name,
        "caminho": str(arquivo_json),
        "status": "OK",
        "erro": None,
        "tamanho_antes": 0,
        "tamanho_depois": 0,
        "tempo_s": 0,
    }

    try:
        inicio = time.time()

        # Ler JSON
        dados_json = json.loads(arquivo_json.read_text(encoding="utf-8"))
        resultado["tamanho_antes"] = dados_json.get("tamanho_chars", 0)

        # Encontrar .txt correspondente
        nome_base = arquivo_json.stem  # remove .json
        arquivo_txt = pasta / f"{nome_base}.txt"

        if not arquivo_txt.exists():
            resultado["status"] = "AVISO"
            resultado["erro"] = "Arquivo .txt não encontrado"
            return resultado

        # Ler conteúdo do .txt limpo
        conteudo = arquivo_txt.read_text(encoding="utf-8")

        # Extrair apenas o [CONTEÚDO] se existir
        if "[CONTEÚDO]" in conteudo:
            idx = conteudo.index("[CONTEÚDO]") + len("[CONTEÚDO]")
            conteudo_puro = conteudo[idx:].strip()
        else:
            conteudo_puro = conteudo.strip()

        # Atualizar campos no JSON
        dados_json["tamanho_chars"] = len(conteudo_puro)
        dados_json["indexado_chroma"] = False  # Marca para reindexação
        resultado["tamanho_depois"] = len(conteudo_puro)

        # Opcionalmente adicionar conteúdo
        if com_conteudo:
            dados_json["conteudo"] = conteudo_puro

        # Salvar JSON atualizado
        arquivo_json.write_text(
            json.dumps(dados_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        resultado["tempo_s"] = time.time() - inicio

        # Registrar no cache
        if cache_file:
            try:
                cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
                cache[str(arquivo_json)] = {
                    "timestamp": time.time(),
                    "tamanho_depois": resultado["tamanho_depois"],
                }
                cache_file.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass  # Silencia erros de cache

    except Exception as e:
        resultado["status"] = "ERRO"
        resultado["erro"] = str(e)
        resultado["tempo_s"] = time.time() - inicio

    return resultado


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def atualizar_jsons_paralelo(
    pasta: Path,
    num_workers: int = 4,
    com_conteudo: bool = False,
    relatorio_apenas: bool = False,
    limpar_cache: bool = False,
) -> None:
    """Atualiza JSONs em paralelo."""

    # Descobrir arquivos JSON
    arquivos_json = sorted(pasta.glob("*.json"))
    if not arquivos_json:
        print(f"❌ Nenhum .json encontrado em {pasta}")
        return

    print(f"\n📂 Pasta: {pasta}")
    print(f"📋 JSONs encontrados: {len(arquivos_json)}")
    print(f"⚙️  Workers: {num_workers}")
    if com_conteudo:
        print(f"📝 Será adicionado o conteúdo aos JSONs")
    print()

    # Cache
    cache_file = pasta / ".atualizacao_cache.json"
    if limpar_cache and cache_file.exists():
        cache_file.unlink()

    # Se relatorio_apenas: ler cache e exibir
    if relatorio_apenas:
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
            print(f"📊 Relatório de atualizações anteriores:\n")
            print(f"   JSONs atualizados: {len(cache)}")

            # Calcular economia de espaço
            economia_total = 0
            for caminho, info in cache.items():
                economia_total += info.get("tamanho_depois", 0)

            print(f"   Tamanho total após limpeza: {economia_total/1024/1024:.1f} MB")

            for caminho, info in list(cache.items())[:5]:
                print(f"   • {Path(caminho).name}: {info['tamanho_depois']} chars")
            if len(cache) > 5:
                print(f"   ... e mais {len(cache) - 5}")
            return
        else:
            print("❌ Nenhum cache encontrado. Execute sem --relatorio-apenas")
            return

    # Processar em paralelo
    tarefas = [
        (arq, pasta, com_conteudo, cache_file) for arq in arquivos_json
    ]

    sucesso = 0
    avisos = 0
    erros = 0
    tempo_total = time.time()
    economia_total = 0
    erros_detalhes = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(atualizar_json_worker, tarefa): tarefa[0]
            for tarefa in tarefas
        }

        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Atualizando",
            unit="JSON",
            ncols=80,
        ) as pbar:
            for future in pbar:
                resultado = future.result()

                if resultado["status"] == "OK":
                    sucesso += 1
                    economia_total += resultado["tamanho_depois"]
                    pbar.set_postfix_str(
                        f"✓{sucesso} ⚠{avisos} ✗{erros} | Último: {resultado['tempo_s']:.2f}s"
                    )
                elif resultado["status"] == "AVISO":
                    avisos += 1
                    pbar.set_postfix_str(
                        f"✓{sucesso} ⚠{avisos} ✗{erros} | AVISO: {resultado['arquivo']}"
                    )
                else:
                    erros += 1
                    erros_detalhes.append(resultado)
                    pbar.set_postfix_str(
                        f"✓{sucesso} ⚠{avisos} ✗{erros} | ERRO: {resultado['arquivo']}"
                    )

    tempo_total = time.time() - tempo_total

    # Relatório final
    print("\n" + "=" * 80)
    print(f"✅ RELATÓRIO FINAL")
    print("=" * 80)
    print(f"Total de JSONs:        {len(arquivos_json)}")
    print(f"Atualizados com êxito:  {sucesso}")
    print(f"Avisos:                {avisos}")
    print(f"Erros:                 {erros}")
    print(f"Taxa de sucesso:       {100*sucesso/len(arquivos_json):.1f}%")
    print(f"Tempo total:           {tempo_total:.1f}s ({tempo_total/60:.1f}m)")
    print(f"Velocidade média:      {len(arquivos_json)/tempo_total:.1f} JSONs/s")

    if economia_total > 0:
        print(f"\n📊 Tamanho total de conteúdo: {economia_total/1024/1024:.1f} MB")

    if com_conteudo:
        print(f"✅ Conteúdo adicionado aos JSONs")
    else:
        print(f"ℹ️  Conteúdo NÃO foi adicionado (use --com-conteudo se desejar)")

    if avisos > 0:
        print(f"\n⚠️  AVISOS ({avisos}):")
        for resultado in [r for r in [atualizar_json_worker(t) for t in tarefas] if r["status"] == "AVISO"][:5]:
            print(f"   • {resultado['arquivo']}: {resultado['erro']}")

    if erros > 0:
        print(f"\n❌ ERROS ({erros}):")
        for erro in erros_detalhes[:10]:
            print(f"   • {erro['arquivo']}: {erro['erro']}")
        if len(erros_detalhes) > 10:
            print(f"   ... e mais {len(erros_detalhes) - 10}")

    print("\n" + "=" * 80)
    print(f"\n🚀 Próximo passo: Reindexar no ChromaDB com super.py --local")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Atualiza JSONs após limpeza de transcrições.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pasta",
        help="Pasta contendo os arquivos .json e .txt",
    )
    parser.add_argument(
        "--com-conteudo",
        action="store_true",
        help="Adiciona o conteudo limpο ao JSON",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de workers paralelos (padrão: 4)",
    )
    parser.add_argument(
        "--relatorio-apenas",
        action="store_true",
        help="Só mostra relatório sem atualizar",
    )
    parser.add_argument(
        "--limpar-cache",
        action="store_true",
        help="Força reprocessamento (ignora cache)",
    )

    args = parser.parse_args()

    pasta = Path(args.pasta)
    if not pasta.is_dir():
        print(f"❌ '{pasta}' não é uma pasta válida.")
        sys.exit(1)

    # Limitar workers ao número de CPUs
    max_workers = os.cpu_count() or 4
    workers = min(args.workers, max_workers)
    if workers != args.workers:
        print(f"⚠️  Limitando a {workers} workers (max CPUs disponíveis)")

    atualizar_jsons_paralelo(
        pasta,
        num_workers=workers,
        com_conteudo=args.com_conteudo,
        relatorio_apenas=args.relatorio_apenas,
        limpar_cache=args.limpar_cache,
    )


if __name__ == "__main__":
    main()
