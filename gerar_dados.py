"""
Gera os conjuntos de dados SINTÉTICOS usados nos notebooks da sessão
"AI Insights for Agro" (AgroNet Forward).

Todos os dados são simulados (semente fixa => resultados reproduzíveis).
Inspiram-se em situações reais do Douro/Trás-os-Montes, mas NÃO representam
nenhuma empresa, vinha ou adega concreta, e os modelos agronómicos usados na
simulação são simplificações didáticas, não modelos validados.

Uso:   python gerar_dados.py          (escreve os CSV na pasta ./dados)
"""
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1) Meteorologia + risco de míldio (predição)
# --------------------------------------------------------------------------
def gerar_meteo(seed=42, anos=(2020, 2021, 2022, 2023, 2024)):
    rng = np.random.default_rng(seed)
    blocos = []
    for ano in anos:
        datas = pd.date_range(f"{ano}-04-01", f"{ano}-09-30", freq="D")
        n = len(datas)
        dia = np.arange(n)
        # Temperatura média: sazonalidade (pico em finais de julho) + ruído autocorrelacionado
        base = 19 + 7 * np.sin((dia - 45) / n * np.pi)
        ruido = np.zeros(n)
        for i in range(1, n):
            ruido[i] = 0.7 * ruido[i - 1] + rng.normal(0, 1.8)
        # Variabilidade interanual (anos mais quentes/frescos)
        t_med = base + ruido + rng.normal(0, 1.0)
        amplitude = rng.uniform(9, 15, n)
        t_max = t_med + amplitude / 2
        t_min = t_med - amplitude / 2

        # Precipitação: mais frequente na primavera, rara no verão
        p_chuva = 0.34 - 0.24 * np.sin(dia / n * np.pi)
        chove = rng.random(n) < p_chuva
        # chuva persiste um pouco (sistemas frontais)
        for i in range(1, n):
            if chove[i - 1] and rng.random() < 0.25:
                chove[i] = True
        precip = np.where(chove, rng.gamma(1.2, 4.5, n), 0.0)

        # Humidade relativa: sobe com chuva, desce com calor
        hum = 72 - 1.1 * (t_med - 19) + np.where(chove, 14, 0) + rng.normal(0, 6, n)
        hum = np.clip(hum, 25, 99)

        # Molhamento foliar (h): depende de chuva e humidade
        molh = np.where(chove, rng.uniform(5, 14, n), np.clip((hum - 78) / 3, 0, 8) + rng.normal(0, 1, n))
        molh = np.clip(molh, 0, 24)

        vento = np.clip(rng.gamma(3, 4, n), 0, 60)

        df = pd.DataFrame({
            "data": datas,
            "temp_media": t_med.round(1),
            "temp_max": t_max.round(1),
            "temp_min": t_min.round(1),
            "precip_mm": precip.round(1),
            "humidade_rel": hum.round(0),
            "molhamento_h": molh.round(1),
            "vento_kmh": vento.round(0),
        })

        # Risco (simulado): condições favoráveis de infeção nos 2-3 dias anteriores
        chuva_48h = pd.Series(precip).rolling(2, min_periods=1).sum().to_numpy()
        hum_3d = pd.Series(hum).rolling(3, min_periods=1).mean().to_numpy()
        molh_3d = pd.Series(molh).rolling(3, min_periods=1).mean().to_numpy()
        score = (
            -2.6
            + 1.5 * (chuva_48h > 5)
            + 0.05 * (hum_3d - 60)
            + 0.15 * molh_3d
            - 0.02 * (t_med - 20) ** 2
        )
        prob = 1 / (1 + np.exp(-score))
        df["risco_miledio"] = (rng.random(n) < prob).astype(int)
        blocos.append(df)
    return pd.concat(blocos, ignore_index=True)


# --------------------------------------------------------------------------
# 2) Câmara de frio (deteção de anomalias)
# --------------------------------------------------------------------------
def gerar_camara(seed=7, dias=21):
    rng = np.random.default_rng(seed)
    passo = 5  # minutos
    n = dias * 24 * 60 // passo
    ts = pd.date_range("2026-09-01 00:00", periods=n, freq=f"{passo}min")
    minutos = np.arange(n) * passo

    setpoint = 4.0
    temp = np.full(n, setpoint)
    # Ciclo normal do compressor (oscilação suave ~ 40 min)
    temp += 0.6 * np.sin(2 * np.pi * minutos / 40)
    # Descongelação: de 6 em 6 horas, subida curta e previsível (NORMAL!)
    ciclo = minutos % (6 * 60)
    defrost = (ciclo < 25)
    temp += np.where(defrost, 2.2 * np.sin(np.pi * ciclo / 25), 0)
    # Aberturas de porta (normais): picos curtos em horário de trabalho
    porta = np.zeros(n, dtype=int)
    for d in range(dias):
        for _ in range(rng.integers(5, 10)):
            h = rng.uniform(7, 18)
            ini = int((d * 24 * 60 + h * 60) / passo)
            dur = int(rng.integers(1, 3))
            if ini + dur < n:
                porta[ini:ini + dur] = 1
    atraso = np.convolve(porta, np.array([0.5, 1.0, 0.8, 0.5, 0.3, 0.15]))[:n]
    temp += 1.3 * atraso
    temp += rng.normal(0, 0.12, n)

    anomalia = np.zeros(n, dtype=int)
    tipo = np.array([""] * n, dtype=object)

    # Anomalia A (dia 6): deriva lenta - compressor a perder eficiência (~14 h)
    ini = int((5 * 24 * 60 + 9 * 60) / passo)
    dur = int(14 * 60 / passo)
    temp[ini:ini + dur] += np.linspace(0, 4.5, dur)
    temp[ini + dur:ini + dur + 24] += np.linspace(4.5, 0, 24)
    anomalia[ini:ini + dur + 12] = 1
    tipo[ini:ini + dur + 12] = "deriva_compressor"

    # Anomalia B (dia 11): sensor encravado (valor constante ~ 5 h)
    ini = int((10 * 24 * 60 + 14 * 60) / passo)
    dur = int(5 * 60 / passo)
    temp[ini:ini + dur] = temp[ini] + rng.normal(0, 0.005, dur)
    anomalia[ini:ini + dur] = 1
    tipo[ini:ini + dur] = "sensor_encravado"

    # Anomalia C (dia 16): porta deixada aberta (~ 90 min)
    ini = int((15 * 24 * 60 + 12 * 60) / passo)
    dur = int(90 / passo)
    temp[ini:ini + dur] += np.linspace(0, 7, dur)
    temp[ini + dur:ini + dur + 36] += np.linspace(7, 0, 36)
    porta[ini:ini + dur] = 1
    anomalia[ini:ini + dur + 12] = 1
    tipo[ini:ini + dur + 12] = "porta_aberta"

    # Anomalia D (dia 19): corte de energia (~ 3 h) - subida rápida
    ini = int((18 * 24 * 60 + 2 * 60) / passo)
    dur = int(3 * 60 / passo)
    temp[ini:ini + dur] += np.linspace(0, 9, dur)
    temp[ini + dur:ini + dur + 48] += np.linspace(9, 0, 48)
    anomalia[ini:ini + dur + 24] = 1
    tipo[ini:ini + dur + 24] = "corte_energia"

    hum = np.clip(85 - 2.0 * (temp - 4) + rng.normal(0, 1.5, n), 40, 99)
    return pd.DataFrame({
        "timestamp": ts,
        "temp_c": temp.round(2),
        "humidade_pct": hum.round(1),
        "porta_aberta": porta,
        "anomalia_real": anomalia,
        "tipo_anomalia": tipo,
    })


# --------------------------------------------------------------------------
# 3) Qualidade de uvas na receção da adega (classificação tabular)
# --------------------------------------------------------------------------
def gerar_uvas(seed=11, n=900):
    rng = np.random.default_rng(seed)
    castas = ["Touriga Nacional", "Touriga Franca", "Tinta Roriz", "Tinto Cão"]
    casta = rng.choice(castas, n, p=[0.3, 0.3, 0.25, 0.15])
    # Qualidade latente (maturação/sanidade) - não observada diretamente
    q = rng.normal(0, 1, n)
    brix = np.clip(21.5 + 1.6 * q + rng.normal(0, 0.7, n), 15, 29)
    acidez = np.clip(6.2 - 0.5 * q + rng.normal(0, 0.6, n), 3.0, 9.5)
    ph = np.clip(3.35 + 0.08 * q + rng.normal(0, 0.07, n), 2.9, 3.9)
    peso_bago = np.clip(1.35 - 0.04 * q + rng.normal(0, 0.12, n), 0.7, 2.0)
    cor = np.clip(60 + 9 * q + rng.normal(0, 6, n), 20, 100)
    danificados = np.clip(6 - 2.0 * q + rng.exponential(2.0, n), 0, 40)
    temp_rec = np.clip(rng.normal(22, 4, n), 10, 38)
    dias_colheita = rng.integers(0, 3, n)

    # Qualidade final (regras + ruído): reflete o que um enólogo valoriza
    pontos = (
        0.9 * (brix - 21.5)
        - 0.7 * np.maximum(acidez - 7.0, 0)
        + 1.9
        - 0.25 * danificados
        + 0.05 * (cor - 60)
        - 0.06 * np.maximum(temp_rec - 26, 0)
        + rng.normal(0, 0.8, n)
    )
    classe = np.where(pontos > 1.2, "A", np.where(pontos > -1.0, "B", "C"))
    return pd.DataFrame({
        "lote_id": [f"L{1000 + i}" for i in range(n)],
        "casta": casta,
        "grau_brix": brix.round(1),
        "acidez_total_gl": acidez.round(1),
        "ph": ph.round(2),
        "peso_bago_g": peso_bago.round(2),
        "indice_cor": cor.round(0),
        "bagos_danificados_pct": danificados.round(1),
        "temp_rececao_c": temp_rec.round(1),
        "dias_desde_colheita": dias_colheita,
        "classe_qualidade": classe,
    })


# --------------------------------------------------------------------------
# 4) Produção e vendas (exercício com assistente de IA generativa)
# --------------------------------------------------------------------------
def gerar_vendas(seed=3):
    rng = np.random.default_rng(seed)
    produtos = {
        "Vinho tinto": (6.5, 2.9, "mes_11"),
        "Azeite": (7.8, 4.6, "mes_12"),
        "Castanha": (3.2, 1.7, "mes_11"),
        "Mel": (6.0, 3.1, "mes_12"),
        "Enchidos": (11.0, 6.4, "mes_12"),
    }
    canais = {"Loja propria": 0.8, "Distribuidor": 1.3, "Online": 0.6}
    regioes = ["Douro", "Tamega e Sousa", "Alto Tamega", "Norte litoral", "Lisboa", "Exportacao"]
    linhas = []
    for ano in (2023, 2024, 2025):
        for mes in range(1, 13):
            for prod, (preco, custo, pico) in produtos.items():
                sazon = 1.0
                if pico == "mes_11" and mes in (10, 11, 12):
                    sazon = 1.8
                if pico == "mes_12" and mes in (11, 12):
                    sazon = 1.7
                for canal, peso in canais.items():
                    tendencia = 1 + 0.06 * (ano - 2023)
                    if canal == "Online":
                        tendencia += 0.25 * (ano - 2023)
                    qtd = max(5, int(rng.normal(120 * peso * sazon * tendencia, 18)))
                    p = preco * rng.normal(1.0, 0.03) * (0.92 if canal == "Distribuidor" else 1.0)
                    c = custo * rng.normal(1.0, 0.04)
                    linhas.append({
                        "ano": ano,
                        "mes": mes,
                        "produto": prod,
                        "canal": canal,
                        "regiao_cliente": rng.choice(regioes, p=[0.25, 0.2, 0.1, 0.15, 0.15, 0.15]),
                        "quantidade": qtd,
                        "receita_eur": round(qtd * p, 2),
                        "custo_eur": round(qtd * c, 2),
                        "devolucoes": int(rng.binomial(qtd, 0.015)),
                    })
    df = pd.DataFrame(linhas)

    # --- Armadilhas deliberadas (para ensinar a verificar o que a IA diz) ---
    # 1) Linhas duplicadas
    df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)
    # 2) Valores em falta
    df.loc[df.sample(8, random_state=2).index, "receita_eur"] = np.nan
    # 3) Erro de digitação (vírgula a mais: 10x o valor)
    idx = df[(df.ano == 2024) & (df.produto == "Azeite") & (df.mes == 6) & (df.canal == "Loja propria")].index[0]
    df.loc[idx, "receita_eur"] = round(df.loc[idx, "receita_eur"] * 10, 2)
    # 4) Nome inconsistente
    idx2 = df[(df.ano == 2025) & (df.produto == "Mel") & (df.mes == 3)].index[:2]
    df.loc[idx2, "produto"] = "mel"
    return df.sort_values(["ano", "mes", "produto", "canal"]).reset_index(drop=True)


def main(destino="dados"):
    os.makedirs(destino, exist_ok=True)
    gerar_meteo().to_csv(f"{destino}/meteo_miledio.csv", index=False)
    gerar_camara().to_csv(f"{destino}/camara_frio.csv", index=False)
    gerar_uvas().to_csv(f"{destino}/qualidade_uvas.csv", index=False)
    gerar_vendas().to_csv(f"{destino}/producao_vendas.csv", index=False)
    print("Ficheiros gerados em", destino)


if __name__ == "__main__":
    main()
