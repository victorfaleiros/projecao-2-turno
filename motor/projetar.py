#!/usr/bin/env python3
"""Projeção do resultado final do 2º turno a partir da apuração parcial do TSE (por município).
Modos:
  projetar.py ao-vivo [--uma-vez]          lê o TSE em loop e grava dados.json
  projetar.py simular ANO HH:MM [HH:MM...]  monta o feed municipal a partir da base por seção (teste do código)
"""
import sys, os, json, time, math, threading, datetime as dt
import numpy as np
try:
    import requests
except ImportError:
    requests = None

AQUI = os.path.dirname(os.path.abspath(__file__))
K = 50000.0          # força do encolhimento (em votos): quanto maior, mais o município/vizinhos obedecem ao nível de cima
P = 2.0              # desconto de município incompleto: peso x fração apurada^P
VIZ = 12             # municípios vizinhos usados
ELEICAO = os.environ.get('ELEICAO', '6258'); CICLO = 'ele2026'
BASE_TSE = f'https://resultados.tse.jus.br/oficial/{CICLO}/{ELEICAO}/dados'
SAIDA = os.environ.get('SAIDA', '/var/lib/docker/volumes/retratos-ia_nginx_data/_data/projecao')
TAXA = float(os.environ.get('TAXA', '35'))   # requisições por segundo (limite do TSE: 100)
FUSO = dt.timezone(dt.timedelta(hours=-3))
# faixa de incerteza da DIFERENÇA projetada (pontos), por % de seções apuradas; calibrada na reprise de 2022 com folga
FAIXA = [(2, 8.0), (5, 5.0), (10, 3.0), (20, 1.5), (40, 1.2), (60, 0.9), (80, 0.6), (95, 0.4), (99, 0.2), (100.01, 0.1)]   # meia-largura (±) da diferença projetada

def faixa(pct):
    for lim, f in FAIXA:
        if pct < lim: return f
    return 0.1

class Modelo:
    def __init__(self, caminho):
        m = json.load(open(caminho, encoding='utf-8'))
        self.A, self.B = m['A'], m['B']; self.grid = np.array(m['grid']); self.final_1t = m['final_1t']
        ms = m['municipios']; self.n = len(ms)
        self.chave = [x['chave'] for x in ms]; self.uf = np.array([x['uf'] for x in ms]); self.cd5 = [x['cd5'] for x in ms]; self.nome = [x['nome'] for x in ms]
        g = lambda k, d=np.nan: np.array([d if x[k] is None else x[k] for x in ms], float)
        self.X1, self.TX1, self.VR1, self.te2, self.nom1 = g('X1'), g('TX1'), g('VR1'), g('te2', 0), g('nom1', 0)
        lat, lon = g('lat'), g('lon')
        ng = len(self.grid); Z = np.zeros((self.n, ng))
        self.bias_cnt, self.bias_rem, self.biasT_cnt, self.biasT_rem = Z.copy(), Z.copy(), Z.copy(), Z.copy()
        for i, ch in enumerate(self.chave):
            c = m['curvas'].get(ch)
            if c: self.bias_cnt[i], self.bias_rem[i], self.biasT_cnt[i], self.biasT_rem[i] = np.nan_to_num(np.array(c, float))
        # preenche gabarito faltante com a média da UF
        self.ufs = sorted(set(self.uf)); self.uf_id = np.array([self.ufs.index(u) for u in self.uf]); NU = len(self.ufs)
        for arr, w in [(self.X1, self.nom1), (self.TX1, self.te2), (self.VR1, self.nom1)]:
            ok = ~np.isnan(arr)
            med = np.bincount(self.uf_id[ok], (arr * w)[ok], NU) / np.maximum(np.bincount(self.uf_id[ok], w[ok], NU), 1)
            arr[~ok] = med[self.uf_id[~ok]]
        self.NU = NU
        for arr in (self.X1, self.TX1, self.VR1): arr[np.isnan(arr)] = np.nanmean(arr)
        # vizinhos por distância entre centroides
        ok = ~np.isnan(lat); idx = np.where(ok)[0]
        x = lon[ok] * 111.32 * np.cos(np.deg2rad(lat[ok])); y = lat[ok] * 110.57
        self.viz = np.full((self.n + 1, VIZ), self.n, int)      # linha n = "ninguém"
        for i0 in range(0, len(idx), 500):
            sl = slice(i0, i0 + 500)
            d2 = (x[sl, None] - x[None, :]) ** 2 + (y[sl, None] - y[None, :]) ** 2
            d2[np.arange(d2.shape[0]), np.arange(i0, min(i0 + 500, len(idx)))] = np.inf
            self.viz[idx[sl]] = idx[np.argpartition(d2, VIZ, axis=1)[:, :VIZ]]

    def interp(self, M, f):
        ng = len(self.grid); pos = np.clip(f * (ng - 1), 0, ng - 1); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, ng - 1); t = pos - i0
        r = np.arange(self.n); return M[r, i0] * (1 - t) + M[r, i1] * t

    def hier(self, d, w):
        sh = lambda s, n, parent: (s + K * parent) / (n + K)
        nat = (d * w).sum() / max(w.sum(), 1e-9)
        e_uf = sh(np.bincount(self.uf_id, d * w, self.NU), np.bincount(self.uf_id, w, self.NU), nat)[self.uf_id]
        s = np.append(d * w, 0); n = np.append(w, 0)
        e_viz = sh(s[self.viz[:self.n]].sum(1), n[self.viz[:self.n]].sum(1), e_uf)
        return sh(d * w, w, e_viz), nat

    def projetar(self, feed):
        """feed: dict com arrays por município (mesma ordem do modelo): te, est, c, nom, vA, vB, st, ts. Devolve o resumo."""
        feed = {k: np.nan_to_num(np.asarray(v, float)) for k, v in feed.items()}
        te = np.where(feed['te'] > 0, feed['te'], np.nan_to_num(self.te2)); est, c, nom, vA, vB = feed['est'], feed['c'], feed['nom'], feed['vA'], feed['vB']
        fm = np.clip(est / np.maximum(te, 1), 0, 1)
        bc, br, btc, btr = self.interp(self.bias_cnt, fm), self.interp(self.bias_rem, fm), self.interp(self.biasT_cnt, fm), self.interp(self.biasT_rem, fm)
        dA = np.where(nom > 0, vA / np.maximum(nom, 1) - (self.X1 + bc), 0)
        dT = np.where(est > 0, c / np.maximum(est, 1) - (self.TX1 + btc), 0)
        dV = np.where(c > 0, nom / np.maximum(c, 1) - self.VR1, 0)
        g = fm ** P
        eA, natA = self.hier(dA, nom * g); eT, _ = self.hier(dT, est * g); eV, _ = self.hier(dV, c * g)
        rem = np.maximum(te - est, 0)
        comp_r = rem * np.clip(self.TX1 + btr + eT, 0, 1); nom_r = comp_r * np.clip(self.VR1 + eV, 0, 1); a_r = nom_r * np.clip(self.X1 + br + eA, 0, 1)
        tot_a, tot_n = vA.sum() + a_r.sum(), nom.sum() + nom_r.sum()
        pA = tot_a / max(tot_n, 1); pB = 1 - pA
        apurado_sec = 100 * feed['st'].sum() / max(feed['ts'].sum(), 1); apurado_ele = 100 * est.sum() / max(te.sum(), 1)
        parcial_A = vA.sum() / max(nom.sum(), 1) if nom.sum() > 0 else None
        comp_proj = (c.sum() + comp_r.sum()) / max(te.sum(), 1)
        ufs = []
        for j, u in enumerate(self.ufs):
            k = self.uf_id == j
            na, nn = vA[k].sum() + a_r[k].sum(), nom[k].sum() + nom_r[k].sum()
            ufs.append({'uf': u, 'apurado': round(100 * est[k].sum() / max(te[k].sum(), 1), 1), 'parcial_A': (round(100 * vA[k].sum() / nom[k].sum(), 1) if nom[k].sum() > 0 else None),
                        'proj_A': round(100 * na / max(nn, 1), 1), 'eleitores': int(te[k].sum()), 'gabarito_A': round(100 * np.nansum(self.X1[k] * self.nom1[k]) / max(self.nom1[k].sum(), 1), 1)})
        fx = faixa(apurado_sec); margem = 100 * (pA - pB)
        return {'apurado_secoes': round(apurado_sec, 2), 'apurado_eleitores': round(apurado_ele, 2),
                'parcial': {'A': None if parcial_A is None else round(100 * parcial_A, 2), 'B': None if parcial_A is None else round(100 * (1 - parcial_A), 2), 'votos_A': int(vA.sum()), 'votos_B': int(vB.sum()), 'validos': int(nom.sum()),
                            'comparecimento': (round(100 * c.sum() / est.sum(), 2) if est.sum() > 0 else None)},
                'projecao': {'A': round(100 * pA, 2), 'B': round(100 * pB, 2), 'margem': round(margem, 2), 'faixa': fx, 'cravado': bool(abs(margem) > fx and apurado_sec >= 5),
                             'comparecimento': round(100 * comp_proj, 2), 'deslocamento_nacional': round(100 * natA, 2)},
                'ufs': ufs}

# ---------------- leitura do TSE ----------------
class Limitador:
    def __init__(self, taxa): self.taxa = taxa; self.lock = threading.Lock(); self.prox = time.monotonic()
    def esperar(self):
        with self.lock:
            agora = time.monotonic(); self.prox = max(self.prox, agora) + 1.0 / self.taxa; atraso = self.prox - agora
        if atraso > 0: time.sleep(atraso)
num = lambda s: float(str(s).replace(',', '.') or 0)

def ler_municipio(sess, lim, uf, cd5, A, B):
    url = f'{BASE_TSE}/{uf.lower()}/{uf.lower()}{cd5}-c0001-e{ELEICAO.zfill(6)}-u.json'
    lim.esperar()
    r = sess.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0 (projecao-2-turno; @vfaleiros)'})
    if r.status_code != 200: raise RuntimeError(f'HTTP {r.status_code}')
    d = r.json(); s, e, v = d.get('s', {}), d.get('e', {}), d.get('v', {})
    votos = {}
    for cg in d.get('carg', []):
        for ag in cg.get('agr', []):
            for pa in ag.get('par', []):
                for c in pa.get('cand', []): votos[int(c['n'])] = int(c.get('vap') or 0)
    return {'ts': int(s.get('ts') or 0), 'st': int(s.get('st') or 0), 'te': int(e.get('te') or 0), 'est': int(e.get('est') or 0), 'c': int(e.get('c') or 0),
            'nom': int(v.get('vnom') or v.get('vvc') or 0), 'vA': votos.get(A, 0), 'vB': votos.get(B, 0), 'hg': d.get('hg', ''), 'dg': d.get('dg', '')}

def ao_vivo(modelo, uma_vez=False):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(SAIDA, exist_ok=True)
    estado = {}; sess = requests.Session(); lim = Limitador(TAXA)
    serie_path = os.path.join(SAIDA, 'serie.json')
    serie = json.load(open(serie_path)) if os.path.exists(serie_path) else []
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16); sess.mount('https://', adapter)
    while True:
        t0 = time.time()
        pend = [i for i in range(modelo.n) if not (i in estado and estado[i]['ts'] > 0 and estado[i]['st'] >= estado[i]['ts'])]
        falhas = 0
        def tarefa(i):
            try: return i, ler_municipio(sess, lim, modelo.uf[i], modelo.cd5[i], modelo.A['n'], modelo.B['n'])
            except Exception as ex: return i, ex
        with ThreadPoolExecutor(12) as ex:
            for i, r in ex.map(tarefa, pend):
                if isinstance(r, Exception): falhas += 1
                else: estado[i] = r
        feed = {k: np.array([estado[i][k] if i in estado else 0 for i in range(modelo.n)], float) for k in ['ts', 'st', 'te', 'est', 'c', 'nom', 'vA', 'vB']}
        res = modelo.projetar(feed)
        agora = dt.datetime.now(FUSO)
        hora_tse = max((estado[i]['hg'] for i in estado if estado[i].get('hg')), default='')
        res.update({'atualizado': agora.strftime('%Y-%m-%dT%H:%M:%S%z'), 'hora': agora.strftime('%H:%M'), 'hora_tse': hora_tse[:5], 'candidatos': {'A': modelo.A, 'B': modelo.B}, 'final_1t': {k: round(100 * v, 2) for k, v in modelo.final_1t.items()},
                    'municipios_lidos': len(estado), 'municipios_pendentes': len(pend), 'falhas': falhas, 'duracao_ciclo_s': round(time.time() - t0, 1)})
        if res['apurado_secoes'] > 0 and (not serie or serie[-1]['apurado'] != res['apurado_secoes']):
            serie.append({'h': res['hora'], 'apurado': res['apurado_secoes'], 'parcial_A': res['parcial']['A'], 'proj_A': res['projecao']['A'], 'faixa': res['projecao']['faixa']})
            json.dump(serie, open(serie_path + '.tmp', 'w')); os.replace(serie_path + '.tmp', serie_path)
        res['serie'] = serie[-600:]
        tmp = os.path.join(SAIDA, 'dados.json.tmp'); json.dump(res, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':')); os.chmod(tmp, 0o644); os.replace(tmp, os.path.join(SAIDA, 'dados.json'))
        print(res['hora'], f"apurado {res['apurado_secoes']}% | parcial A {res['parcial']['A']} | proj A {res['projecao']['A']} (dif {res['projecao']['margem']} ± {res['projecao']['faixa']}) | pend {len(pend)} falhas {falhas} | {res['duracao_ciclo_s']}s", flush=True)
        if uma_vez or (res['apurado_secoes'] >= 100 and not pend): break
        time.sleep(max(5, 30 - (time.time() - t0)))

def simular(modelo, ano, horas):
    import pandas as pd
    b = pd.read_parquet(f'work/{ano}_base.parquet'); b = b[b.aptos_2.notna()].copy()
    for col in ['aptos_2', 'comp_2', 'vNom_2', 'vA_2', 'vB_2']: b[col] = b[col].fillna(0)
    b['chave'] = b.SG_UF + '-' + b.CD_MUNICIPIO.astype(int).astype(str); pos = {c: i for i, c in enumerate(modelo.chave)}; b['i'] = b.chave.map(pos)
    b = b[b.i.notna()]; b['i'] = b.i.astype(int)
    tot_sec = np.bincount(b.i, minlength=modelo.n).astype(float); te = np.bincount(b.i, b.aptos_2, modelo.n)
    verdade = 100 * b.vA_2.sum() / b.vNom_2.sum(); print(f'verdade A = {verdade:.2f}')
    data = b.t_tot_2.min().strftime('%Y-%m-%d')
    for h in horas:
        T = pd.Timestamp(f'{data} {h}'); c = (b.t_tot_2 <= T).to_numpy()
        feed = {'ts': tot_sec, 'st': np.bincount(b.i[c], minlength=modelo.n).astype(float), 'te': te, 'est': np.bincount(b.i[c], b.aptos_2[c], modelo.n), 'c': np.bincount(b.i[c], b.comp_2[c], modelo.n),
                'nom': np.bincount(b.i[c], b.vNom_2[c], modelo.n), 'vA': np.bincount(b.i[c], b.vA_2[c], modelo.n), 'vB': np.bincount(b.i[c], b.vB_2[c], modelo.n)}
        r = modelo.projetar(feed)
        print(f"{h} apurado {r['apurado_secoes']:5.1f}%  parcial {r['parcial']['A']}  proj {r['projecao']['A']}  erro {r['projecao']['A'] - verdade:+.2f}  margem {r['projecao']['margem']:+.2f} ± {r['projecao']['faixa']}  cravado={r['projecao']['cravado']}")

if __name__ == '__main__':
    modo = sys.argv[1] if len(sys.argv) > 1 else 'ao-vivo'
    if modo == 'simular':
        ano = sys.argv[2]; m = Modelo(os.path.join(AQUI, f'modelo_{ano}.json')); simular(m, ano, sys.argv[3:])
    else:
        m = Modelo(os.path.join(AQUI, os.environ.get('MODELO', 'modelo_2026.json'))); ao_vivo(m, uma_vez='--uma-vez' in sys.argv)
