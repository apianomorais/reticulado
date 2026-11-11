# -*- coding: utf-8 -*-
"""
Gera as páginas do site a partir dos arquivos fonte/*.html.

Cada arquivo fonte começa com um cabeçalho de três linhas:
    titulo: Título da página
    kicker: Módulo 1
    resumo: frase curta (aparece na página inicial e nos metadados)
seguido de uma linha em branco e do conteúdo em HTML. Fórmulas em LaTeX: \\( ... \\) e \\[ ... \\].
Cada <h2 id="..."> vira um item do índice lateral.

Uso:  python fonte/gerar.py      (a partir da raiz do repositório)
"""
import os
import re

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTE = os.path.join(RAIZ, "fonte")

PAGINAS = [  # (arquivo, rótulo no menu, número)
    ("index", "Apresentação", ""),
    ("reticulados", "Reticulados e duais", "1"),
    ("anisotropia", "Anisotropia e designs", "2"),
    ("dinamica", "Dinâmica e indicadores", "3"),
    ("forma-normal", "Forma normal e caos", "4"),
    ("difusao", "Difusão e Pólya", "5"),
    ("codigos", "Códigos", "6"),
    ("referencias", "Referências", ""),
]

MODELO = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo_curto}</title>
<meta name="description" content="{resumo}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/estilo.css">
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']],
  macros: {{ R: '\\\\mathbb{{R}}', Z: '\\\\mathbb{{Z}}', E: '\\\\mathcal{{E}}' }} }},
  chtml: {{ scale: 0.98 }} }};
try {{ var t = localStorage.getItem('tema'); if (t) document.documentElement.dataset.theme = t; }} catch (e) {{}}
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <a class="brand" href="index.html"><div class="t1">Dinâmica em potenciais de rede</div>
  <div class="t2">geometria · caos · transporte</div></a>
  <ul class="nav">
{menu}
  </ul>
{toc}
  <div class="tema"><button onclick="alternarTema()">claro / escuro</button></div>
</aside>
<main>
<div class="kicker">{kicker}</div>
<h1>{titulo}</h1>
{conteudo}
{passos}
<footer>Grupo de pesquisa: L. P. Silva, M. O. Tahim, A. F. Morais (URCA / UECE). Material em construção —
resultados marcados como preliminares podem mudar.</footer>
</main>
</div>
<script>
function alternarTema() {{
  var r = document.documentElement, at = r.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  r.dataset.theme = at === 'dark' ? 'light' : 'dark';
  try {{ localStorage.setItem('tema', r.dataset.theme); }} catch (e) {{}}
}}
</script>
</body>
</html>
"""


def ler(nome):
    with open(os.path.join(FONTE, nome + ".html"), encoding="utf-8") as fh:
        txt = fh.read()
    cab, corpo = txt.split("\n\n", 1)
    meta = dict(l.split(": ", 1) for l in cab.strip().splitlines())
    return meta, corpo


def main():
    metas = {n: ler(n)[0] for n, _, _ in PAGINAS}
    for i, (nome, rot, num) in enumerate(PAGINAS):
        meta, corpo = ler(nome)
        menu = "\n".join(
            f'    <li><a href="{n}.html"{" class=\"atual\"" if n == nome else ""}>'
            f'{f"<span class=n>{k}</span>" if k else ""}{r}</a></li>'
            for n, r, k in PAGINAS)
        secs = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', corpo)
        toc = ""
        if secs:
            toc = '  <div class="toc"><div class="rot">Nesta página</div>\n' + "\n".join(
                f'    <a href="#{a}">{re.sub("<.*?>", "", t)}</a>' for a, t in secs) + "\n  </div>"
        ant = PAGINAS[i - 1] if i > 0 else None
        prox = PAGINAS[i + 1] if i + 1 < len(PAGINAS) else None
        passos = '<nav class="passo">' + (
            f'<a href="{ant[0]}.html"><span class="rot">← anterior</span>{metas[ant[0]]["titulo"]}</a>'
            if ant else "<span></span>") + (
            f'<a href="{prox[0]}.html" style="text-align:right"><span class="rot">próximo →</span>'
            f'{metas[prox[0]]["titulo"]}</a>' if prox else "<span></span>") + "</nav>"
        html = MODELO.format(
            titulo=meta["titulo"], kicker=meta.get("kicker", ""),
            titulo_curto=(meta["titulo"] if nome != "index" else "Dinâmica em potenciais de rede"),
            resumo=meta.get("resumo", "").replace('"', "&quot;"),
            menu=menu, toc=toc, conteudo=corpo, passos=passos)
        with open(os.path.join(RAIZ, nome + ".html"), "w", encoding="utf-8") as fh:
            fh.write(html)
        print("gerado", nome + ".html")


if __name__ == "__main__":
    main()
