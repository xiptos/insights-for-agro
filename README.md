# Material de demonstração: AI Insights for Agro

Sessão **AgroNet Forward | AI Insights for Agro: Aplicações práticas de Inteligência Artificial**
(Predição, Classificação e Controlo de Qualidade).

Este pacote tem **3 notebooks** (já executados, com resultados e gráficos guardados), **4 conjuntos de dados** e uma
ficha de exercício para os participantes.

> **Todos os dados são sintéticos**, gerados com semente fixa (`gerar_dados.py`). Inspiram-se em situações do
> Douro/Trás-os-Montes, mas não representam nenhuma empresa real e os "modelos agronómicos" usados na simulação são
> simplificações didáticas. 

## Conteúdo

```
insights-for-agro/
├── notebooks/
│   ├── 01_predicao_risco_miledio.ipynb          Predição (risco de míldio)
│   ├── 02_anomalias_camara_frio.ipynb           Controlo de qualidade (anomalias)
│   └── 03_classificacao_qualidade_uvas.ipynb    Classificação (A/B/C na receção da adega)
├── dados/
│   ├── meteo_miledio.csv        915 dias · 5 épocas · risco 0/1
│   ├── camara_frio.csv          6 048 leituras (5 em 5 min) · 4 avarias simuladas
│   ├── qualidade_uvas.csv       900 lotes · classe A/B/C
│   └── producao_vendas.csv      546 linhas · para o exercício com IA generativa (com "armadilhas")
├── exercicio_participantes.md   Ficha para distribuir
├── gerar_dados.py               Regenera os CSV (python gerar_dados.py)
└── pyproject.toml / uv.lock     Projeto uv (dependências)
```

## Como usar

**Google Colab (recomendado para os participantes, sem instalação):** abrir o link direto de cada notebook (ver abaixo).
Se não carregarem o CSV para a sessão, o notebook **gera os mesmos dados automaticamente**, por isso não há fricção.
O único passo opcional é a célula final do notebook 1 (meteorologia real).

| Notebook | Abrir no Colab |
|---|---|
| 1 · Predição (míldio) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiptos/insights-for-agro/blob/main/notebooks/01_predicao_risco_miledio.ipynb) |
| 2 · Anomalias (câmara de frio) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiptos/insights-for-agro/blob/main/notebooks/02_anomalias_camara_frio.ipynb) |
| 3 · Classificação (qualidade das uvas) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiptos/insights-for-agro/blob/main/notebooks/03_classificacao_qualidade_uvas.ipynb) |

**Localmente:** este é um projeto [uv](https://docs.astral.sh/uv/). Corra `uv sync` e depois `uv run jupyter lab`,
a partir da pasta raiz (os notebooks em `notebooks/` procuram `../dados/…`).

## Mapa para o outline da sessão

| Bloco da sessão | Material | Mensagem-chave |
|---|---|---|
| Predição (15') | Notebook 1 | O valor está na **probabilidade** + na decisão ajustada ao custo, não no algoritmo. |
| Classificação (15') | Notebook 3 + **Teachable Machine** (ver abaixo) | Modelos **explicáveis** (árvore) e **que erros custam mais**. |
| Controlo de qualidade (10') | Notebook 2 | Sem contexto, um alarme fixo ou avisa tarde ou incomoda; o **contexto** (o que é normal) é o ganho. |
| IA generativa na gestão (10') | `producao_vendas.csv` + `exercicio_participantes.md` | A IA calcula depressa, mas **é preciso verificar**. |
| Mão na massa (10') | Teachable Machine **ou** exercício com IA generativa | Experimentar sem instalar nada. |

Nos notebooks, as secções finais "Exercícios para experimentar" são para quem quiser continuar depois da sessão.

## Resultados de referência (para saber o que esperar ao vivo)

Valores desta versão dos dados (semente fixa, resultados reproduzíveis).

**Notebook 1 (míldio):** época de teste 2024, 34 dias de risco em 183 (19%). O "nunca há risco" tem 81% de
exatidão e 0% de recall (é o exemplo para mostrar que a exatidão engana). Com o limiar por defeito (0,5), o
*random forest* **não é claramente melhor** que a regra manual (chuva 48 h > 5 mm); a vantagem surge ao ajustar o
limiar ao custo: **3 495 € (modelo, limiar 0,15) vs 4 885 € (regra) vs 10 200 € (nunca tratar)**, com custos
**ilustrativos** (35 € por falso alarme, 300 € por risco falhado). Boa pergunta para a plateia: *"e se uma infeção custar 1 000 €?"*.

**Notebook 2 (câmara de frio):**

| Método | Avarias | Atraso médio | Falsos alarmes |
|---|---|---|---|
| A. Limite 8 °C | 3/4 | 122 min | 4 |
| A. Limite 7 °C | 3/4 | 67 min | 13 |
| B. Perfil esperado + linha plana | 4/4 | 98 min | 4 |
| C. Isolation Forest (+ persistência) | 4/4 | 150 min | 1 |

Só há 4 avarias: **não se trata de uma medida de desempenho**, serve para compreender os compromissos.
O sensor encravado é a avaria que nenhum limite apanha.

**Notebook 3 (uvas):** referência 37%; árvore (profundidade 3) 71%; *random forest* 76%. Só °Brix ≈ 74%, todas as
medições ≈ 81% (validação cruzada). Acrescentar só a acidez **não melhora** (ruído): um bom momento para falar de
custo-benefício de medir. Zero erros "graves" (C classificado como A) no conjunto de teste.

## Demonstração com Teachable Machine (a preparar)

Não está incluída porque depende de **fotografias suas**. Sugestão:
- Escolha **2–3 classes** fáceis de distinguir à câmara (ex.: folha sã / folha com mancha; fruto bom / fruto com defeito;
  ou objetos simples).
- **30–50 imagens por classe**, variando ângulo e luz. Treinar demora segundos.
- **Mostre uma falha de propósito**: teste com um fundo diferente ou uma luz diferente e deixe o modelo errar. É a lição
  mais forte sobre **qualidade e variedade dos dados**.
- Alternativa se não tiver fotos: pedir aos participantes para usarem objetos que tenham à mão.
- Para imagens agronómicas públicas, o conjunto **PlantVillage** é uma opção habitual (verificar a licença e o acesso antes).

## Exercício com IA generativa: chave de correção (só para si)

O ficheiro `producao_vendas.csv` tem **armadilhas deliberadas**: 6 linhas duplicadas, 8 valores de `receita_eur` em
falta, 1 erro de digitação (receita de azeite, loja própria, junho de 2024, **10× o valor**) e 2 linhas com `mel` em
minúscula (nome inconsistente). O objetivo é ver quem **verifica** o que a IA diz.

| Pergunta | Resposta "ingénua" (sem limpar) | Resposta limpa (referência) |
|---|---|---|
| Receita total por ano | 2023: 144 473 € · 2024: 173 564 € · 2025: 178 893 € | 2023: ≈ 145 897 € · 2024: ≈ 168 635 € · 2025: ≈ 178 691 € (duplicados removidos, erro corrigido, receitas em falta estimadas por quantidade × preço mediano) |
| Melhor margem percentual | — | **Vinho tinto (≈ 54%)**, seguido de Mel (≈ 46%) |
| Canal que mais cresceu 2023→2025 | — | **Online (≈ +61%)**; Loja própria ≈ +13%; Distribuidor ≈ +10% |

Os valores ingénuos e os limpos **diferem** (sobretudo 2024, por causa do erro de digitação) e há mais de uma forma legítima
de tratar os valores em falta, por isso o que interessa é a IA **declarar o que fez**, não o número exato. Se ignorar os
valores em falta em vez de os estimar, obtém: 143 978 € / 165 155 € / 176 387 €.

Outros factos para curiosidade: o mel aparece como "Mel" e "mel" (a IA que não normaliza trata-os como produtos
diferentes); os meses de maior receita são novembro e dezembro.

## Limitações e coisas a confirmar antes da sessão

- **Testado:** os 3 notebooks foram executados de ponta a ponta (Python 3.12, scikit-learn 1.8, pandas 3.0) sem erros.
  **Não foram testados no Google Colab** propriamente dito: faça uma execução prévia lá.
- **Célula opcional da API Open-Meteo (notebook 1, última célula): não foi validada.** No ambiente em que preparei o
  material o acesso à rede estava restringido (erro 403). A chamada segue a documentação pública da API, mas confirme o
  nome dos parâmetros diários antes de a mostrar. Está protegida por `try/except`, por isso falha sem rebentar o notebook.
- **Resultados dependem das versões das bibliotecas:** em versões diferentes de scikit-learn, valores como "76%" podem mudar
  ligeiramente. Ensaie com a mesma versão que vai usar ao vivo (ou mostre as saídas já guardadas).
- **Plano B:** os notebooks já trazem todas as saídas e gráficos guardados. Se a sessão online falhar, pode **percorrer
  os notebooks só a ler**. Convém também ter à mão capturas ou uma gravação da demonstração do Teachable Machine.
- **Mensagem de honestidade para o público:** dados simulados; poucos eventos; num caso real, o difícil é ter **dados e
  rótulos fiáveis**, não escolher o algoritmo.
