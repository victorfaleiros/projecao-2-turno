"""Reprise do 2º turno de 2022 na ordem real de totalização. Compara modelos de projeção."""
import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
K = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0
KF = float(sys.argv[2]) if len(sys.argv) > 2 else K
P = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
KNB_LOC, KNB_MUN = 40, 12
b = pd.read_parquet('work/2022_base.parquet')
b = b[b.aptos_2.notna() & (b.aptos_2 > 0)].reset_index(drop=True)
N = len(b)
f = lambda c: b[c].fillna(0).to_numpy(float)
aptos2, comp2, nom2, vA2 = f('aptos_2'), f('comp_2'), f('nom_2'), f('vA_2')
aptos1, comp1, nom1, vA1 = f('aptos_1'), f('comp_1'), f('nom_1'), f('vA_1')
t2 = b.t_tot_2.to_numpy()
uf = pd.factorize(b.SG_UF)[0]; mun = pd.factorize(b.mun_id)[0]; loc = pd.factorize(b.local_id)[0]
NU, NM, NL = uf.max()+1, mun.max()+1, loc.max()+1
uf_of_mun = np.zeros(NM, int); uf_of_mun[mun] = uf
mun_of_loc = np.zeros(NL, int); mun_of_loc[loc] = mun
def wmean_fill(num, den, groups):
    """taxa = num/den por seção; onde den==0, usa a taxa do grupo (município -> UF -> nacional)."""
    r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    for g in groups:
        gn = np.bincount(g, num); gd = np.bincount(g, den)
        gr = np.where(gd > 0, gn / np.where(gd > 0, gd, 1), np.nan)
        r = np.where(np.isnan(r), gr[g], r)
    return np.where(np.isnan(r), num.sum() / den.sum(), r)
tx1 = wmean_fill(comp1, aptos1, [mun, uf]); vr1 = wmean_fill(nom1, comp1, [mun, uf]); x1 = wmean_fill(vA1, nom1, [mun, uf])
VERDADE = vA2.sum() / nom2.sum()
# ---- vizinhos espaciais de locais de votação (km) ----
lat = b.lat.to_numpy(float); lon = b.lon.to_numpy(float)
loc_lat = np.full(NL, np.nan); loc_lon = np.full(NL, np.nan)
ok = ~np.isnan(lat)
loc_lat[loc[ok]] = lat[ok]; loc_lon[loc[ok]] = lon[ok]
def xy(la, lo): return np.c_[lo * 111.32 * np.cos(np.deg2rad(la)), la * 110.57]
has = ~np.isnan(loc_lat); idx_has = np.where(has)[0]
tree = cKDTree(xy(loc_lat[has], loc_lon[has]))
_, nn = tree.query(xy(loc_lat[has], loc_lon[has]), k=KNB_LOC + 1)
NB_LOC = np.full((NL + 1, KNB_LOC), NL, int)        # linha NL = "nada" (zero)
NB_LOC[idx_has] = idx_has[nn[:, 1:]]
# vizinhos de municípios (centroides)
mun_lat = np.bincount(mun, np.nan_to_num(lat), minlength=NM) / np.maximum(np.bincount(mun, ok.astype(float), minlength=NM), 1)
mun_lon = np.bincount(mun, np.nan_to_num(lon), minlength=NM) / np.maximum(np.bincount(mun, ok.astype(float), minlength=NM), 1)
mhas = np.bincount(mun, ok.astype(float), minlength=NM) > 0; midx = np.where(mhas)[0]
mt = cKDTree(xy(mun_lat[mhas], mun_lon[mhas])); _, mnn = mt.query(xy(mun_lat[mhas], mun_lon[mhas]), k=KNB_MUN + 1)
NB_MUN = np.full((NM + 1, KNB_MUN), NM, int); NB_MUN[midx] = midx[mnn[:, 1:]]
def shrink(s, n, parent, k=None):
    k = K if k is None else k
    return (s + k * parent) / (n + k)
def hier_secao(d, w, counted, espacial):
    """estimativa do deslocamento d (ponderado por w) para TODAS as seções: nacional -> UF -> município -> [vizinhos] -> local."""
    wc = w * counted; num = d * wc
    nat = num.sum() / max(wc.sum(), 1e-9)
    e_uf = shrink(np.bincount(uf, num, NU), np.bincount(uf, wc, NU), nat)
    e_mun = shrink(np.bincount(mun, num, NM), np.bincount(mun, wc, NM), e_uf[uf_of_mun])
    s_loc = np.append(np.bincount(loc, num, NL), 0); n_loc = np.append(np.bincount(loc, wc, NL), 0)
    parent = e_mun[mun_of_loc]
    if espacial:
        parent = shrink(s_loc[NB_LOC[:NL]].sum(1), n_loc[NB_LOC[:NL]].sum(1), parent)
    e_loc = shrink(s_loc[:NL], n_loc[:NL], parent)
    return e_loc[loc], nat, e_uf[uf], e_mun[mun]
def projeta(eT, eV, eA, counted):
    c = aptos2 * np.clip(tx1 + eT, 0, 1); n = c * np.clip(vr1 + eV, 0, 1); a = n * np.clip(x1 + eA, 0, 1)
    tot_a = vA2[counted].sum() + a[~counted].sum(); tot_n = nom2[counted].sum() + n[~counted].sum()
    return tot_a / tot_n
# ---- modelo "feed municipal": só o que o JSON do TSE dá por município ----
X1_m = np.bincount(mun, vA1, NM) / np.maximum(np.bincount(mun, nom1, NM), 1)
TX1_m = np.bincount(mun, comp1, NM) / np.maximum(np.bincount(mun, aptos1, NM), 1)
VR1_m = np.bincount(mun, nom1, NM) / np.maximum(np.bincount(mun, comp1, NM), 1)
TE_m = np.bincount(mun, aptos2, NM)
# curva de viés de ordem do 1º turno, por município: viés da parte apurada e da parte que falta, em função da fração f do eleitorado apurado
GRID = np.linspace(0, 1, 21)
BIAS_CNT = np.zeros((NM, len(GRID))); BIAS_REM = np.zeros((NM, len(GRID))); BIAS_T_CNT = np.zeros((NM, len(GRID))); BIAS_T_REM = np.zeros((NM, len(GRID)))
t1o = b.t_tot_1.to_numpy()
order = np.lexsort((t1o, mun))
for m0, m1 in zip(*np.unique(mun[order], return_index=True)[::-1] if False else (np.unique(mun[order], return_index=True)[1], np.append(np.unique(mun[order], return_index=True)[1][1:], N))):
    ix = order[m0:m1]; m = mun[ix[0]]
    ca = np.cumsum(aptos1[ix]); cn = np.cumsum(nom1[ix]); cv = np.cumsum(vA1[ix]); cc = np.cumsum(comp1[ix])
    if ca[-1] <= 0 or cn[-1] <= 0: continue
    frac = ca / ca[-1]
    sh_cnt = cv / np.maximum(cn, 1); sh_rem = (cv[-1] - cv) / np.maximum(cn[-1] - cn, 1)
    tx_cnt = cc / np.maximum(ca, 1); tx_rem = (cc[-1] - cc) / np.maximum(ca[-1] - ca, 1)
    BIAS_CNT[m] = np.interp(GRID, frac, sh_cnt) - X1_m[m]; BIAS_REM[m] = np.interp(GRID, frac, sh_rem) - X1_m[m]
    BIAS_T_CNT[m] = np.interp(GRID, frac, tx_cnt) - TX1_m[m]; BIAS_T_REM[m] = np.interp(GRID, frac, tx_rem) - TX1_m[m]
    BIAS_REM[m, -1] = 0; BIAS_T_REM[m, -1] = 0
def interp_rows(M, fcol):
    pos = np.clip(fcol * (len(GRID) - 1), 0, len(GRID) - 1); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, len(GRID) - 1); t = pos - i0
    return M[np.arange(NM), i0] * (1 - t) + M[np.arange(NM), i1] * t
def hier_mun(d_m, w_m, espacial):
    nat = (d_m * w_m).sum() / max(w_m.sum(), 1e-9)
    e_uf = shrink(np.bincount(uf_of_mun, d_m * w_m, NU), np.bincount(uf_of_mun, w_m, NU), nat, KF)
    parent = e_uf[uf_of_mun]
    if espacial:
        s = np.append(d_m * w_m, 0); n = np.append(w_m, 0)
        parent = shrink(s[NB_MUN[:NM]].sum(1), n[NB_MUN[:NM]].sum(1), parent, KF)
    return shrink(d_m * w_m, w_m, parent, KF), nat
def feed_municipal(counted, corrigir, espacial=True, desconto=True):
    cw = counted.astype(float)
    est = np.bincount(mun, aptos2 * cw, NM); c_m = np.bincount(mun, comp2 * cw, NM); n_m = np.bincount(mun, nom2 * cw, NM); a_m = np.bincount(mun, vA2 * cw, NM)
    fm = est / np.maximum(TE_m, 1)
    bc = interp_rows(BIAS_CNT, fm) if corrigir else 0; br_ = interp_rows(BIAS_REM, fm) if corrigir else 0
    btc = interp_rows(BIAS_T_CNT, fm) if corrigir else 0; btr = interp_rows(BIAS_T_REM, fm) if corrigir else 0
    dA = np.where(n_m > 0, a_m / np.maximum(n_m, 1) - (X1_m + bc), 0)
    dT = np.where(est > 0, c_m / np.maximum(est, 1) - (TX1_m + btc), 0)
    dV = np.where(c_m > 0, n_m / np.maximum(c_m, 1) - VR1_m, 0)
    g = fm ** P if desconto else 1.0
    eA, _ = hier_mun(dA, n_m * g, espacial); eT, _ = hier_mun(dT, est * g, espacial); eV, _ = hier_mun(dV, c_m * g, espacial)
    rem = TE_m - est
    comp_r = rem * np.clip(TX1_m + btr + eT, 0, 1); nom_r = comp_r * np.clip(VR1_m + eV, 0, 1); a_r = nom_r * np.clip(X1_m + br_ + eA, 0, 1)
    return (a_m.sum() + a_r.sum()) / (n_m.sum() + nom_r.sum())
# ---- reprise ----
horas = pd.date_range('2022-10-30 17:05', '2022-10-30 19:30', freq='5min').tolist() + pd.to_datetime(['2022-10-30 20:00', '2022-10-30 21:00']).tolist()
linhas = []
for T in horas:
    counted = t2 <= np.datetime64(T)
    if counted.sum() == 0: continue
    dT = np.where(counted, comp2 / np.maximum(aptos2, 1) - tx1, 0); dV = np.where(counted & (comp2 > 0), nom2 / np.maximum(comp2, 1) - vr1, 0); dA = np.where(counted & (nom2 > 0), vA2 / np.maximum(nom2, 1) - x1, 0)
    r = {'hora': T.strftime('%H:%M'), 'pct_secoes': 100 * counted.mean(), 'pct_eleitores': 100 * aptos2[counted].sum() / aptos2.sum(), 'ingenuo': vA2[counted].sum() / max(nom2[counted].sum(), 1)}
    for nome, esp in [('secao_hier', False), ('secao_espacial', True)]:
        eT, natT, ufT, munT = hier_secao(dT, aptos2, counted, esp); eV, natV, ufV, munV = hier_secao(dV, comp2, counted, esp); eA, natA, ufA, munA = hier_secao(dA, nom2, counted, esp)
        r[nome] = projeta(eT, eV, eA, counted)
        if not esp:
            r['nacional'] = projeta(np.full(N, natT), np.full(N, natV), np.full(N, natA), counted)
            r['uf'] = projeta(ufT, ufV, ufA, counted)
            r['municipio'] = projeta(munT, munV, munA, counted)
    r['feed_uf_semdesc'] = feed_municipal(counted, corrigir=False, espacial=False, desconto=False)
    r['feed_uf_desc'] = feed_municipal(counted, corrigir=False, espacial=False, desconto=True)
    r['feed_esp_desc'] = feed_municipal(counted, corrigir=False, espacial=True, desconto=True)
    r['feed_esp_corr_semdesc'] = feed_municipal(counted, corrigir=True, espacial=True, desconto=False)
    r['feed_esp_corr_desc'] = feed_municipal(counted, corrigir=True, espacial=True, desconto=True)
    linhas.append(r)
df = pd.DataFrame(linhas)
modelos = ['ingenuo', 'nacional', 'uf', 'municipio', 'secao_hier', 'secao_espacial', 'feed_uf_semdesc', 'feed_uf_desc', 'feed_esp_desc', 'feed_esp_corr_semdesc', 'feed_esp_corr_desc']
err = df[modelos].sub(VERDADE) * 100
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.float_format', '{:+.2f}'.format)
print(f'K={K:.0f} KF={KF:.0f} P={P:.0f}  VERDADE (A = Lula, % válidos) = {100*VERDADE:.2f}')
out = pd.concat([df[['hora']], df[['pct_secoes', 'pct_eleitores']].round(1), err], axis=1)
print(out.to_string(index=False))
print('\nerro absoluto médio (pontos) por faixa de apuração:')
fx = pd.cut(df.pct_secoes, [0, 20, 40, 60, 80, 95, 100.1], labels=['0-20', '20-40', '40-60', '60-80', '80-95', '95-100'])
print(err.abs().groupby(fx, observed=True).mean().round(2).to_string())
out.to_csv(f'work/reprise_2022_K{K:.0f}_KF{KF:.0f}_P{P:.0f}.csv', index=False)
