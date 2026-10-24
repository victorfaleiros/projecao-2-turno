"""Gera o arquivo do modelo para domingo: por município, o gabarito do 1º turno de 2026 e as curvas de viés de ordem."""
import json, numpy as np, pandas as pd
import sys
ANO = sys.argv[1]
A, B = (22, 13) if ANO == '2026' else (13, 22)
b = pd.read_parquet('work/' + ANO + '_base.parquet')
b = b[b.aptos_1.notna() & (b.aptos_1 > 0)].copy()
for col in ['comp_1', 'vNom_1', 'vA_1', 'vB_1']: b[col] = b[col].fillna(0)
if ANO == '2026':
    cfg = json.load(open('work/mun-e006258-cm.json'))
    lista = [(uf['cd'].upper(), int(m['cd']), m['nm'], m['cd']) for uf in cfg['abr'] for m in uf['mu']]
    cfgdf = pd.DataFrame(lista, columns=['SG_UF', 'CD_MUNICIPIO', 'nome', 'cd5'])
else:
    cfgdf = b.drop_duplicates(['SG_UF', 'CD_MUNICIPIO'])[['SG_UF', 'CD_MUNICIPIO', 'mun_nome']].rename(columns={'mun_nome': 'nome'}); cfgdf['cd5'] = cfgdf.CD_MUNICIPIO.astype(int).astype(str).str.zfill(5)
print('municípios no config do 2º turno:', len(cfgdf), '| UFs:', cfgdf.SG_UF.nunique())
# eleitorado do 2º turno por seção (arquivo de locais de votação, turno 2) quando existir
lv = pd.read_parquet('work/' + ANO + '_lv.parquet'); lv['CD_MUNICIPIO'] = lv.CD_MUNICIPIO.astype(int)
print('turnos no arquivo de locais:', lv.NR_TURNO.value_counts().to_dict())
K = ['SG_UF', 'CD_MUNICIPIO', 'NR_ZONA', 'NR_SECAO']
lv2 = lv[lv.NR_TURNO == 2].drop_duplicates(K).set_index(K)['QT_ELEITOR_SECAO'] if (lv.NR_TURNO == 2).any() else None
b = b.set_index(K)
b['aptos_2'] = lv2.reindex(b.index).values if lv2 is not None else np.nan
cobertura = b.aptos_2.notna().mean(); print('cobertura do eleitorado do 2º turno por seção:', round(cobertura, 3))
b['aptos_2'] = b.aptos_2.fillna(b.aptos_1)
b = b.reset_index()
# agregados por município
g = b.groupby(['SG_UF', 'CD_MUNICIPIO'])
m = pd.DataFrame({'aptos1': g.aptos_1.sum(), 'comp1': g.comp_1.sum(), 'nom1': g.vNom_1.sum(), 'vA1': g.vA_1.sum(), 'vB1': g.vB_1.sum(), 'te2': g.aptos_2.sum(),
                  'lat': g.lat.mean(), 'lon': g.lon.mean(), 'secoes': g.size()}).reset_index()
m['nom1'] = m.vA1 + m.vB1; m['X1'] = m.vA1 / m.nom1; m['TX1'] = m.comp1 / m.aptos1; m['VR1'] = m.nom1 / m.comp1
# curvas de viés de ordem (fração do eleitorado apurado no 1º turno -> viés da parte apurada e da parte que falta)
GRID = np.linspace(0, 1, 21)
b = b.sort_values(['SG_UF', 'CD_MUNICIPIO', 't_tot_1'])
curvas = {}
for (uf, cd), d in b.groupby(['SG_UF', 'CD_MUNICIPIO'], sort=False):
    ca = d.aptos_1.cumsum().to_numpy(); cn = (d.vA_1 + d.vB_1).cumsum().to_numpy(); cv = d.vA_1.cumsum().to_numpy(); cc = d.comp_1.cumsum().to_numpy()
    if ca[-1] <= 0 or cn[-1] <= 0: continue
    X = cv[-1] / cn[-1]; TX = cc[-1] / ca[-1]; frac = ca / ca[-1]
    sh_cnt = np.interp(GRID, frac, cv / np.maximum(cn, 1)) - X
    sh_rem = np.interp(GRID, frac, (cv[-1] - cv) / np.maximum(cn[-1] - cn, 1)) - X; sh_rem[-1] = 0
    tx_cnt = np.interp(GRID, frac, cc / np.maximum(ca, 1)) - TX
    tx_rem = np.interp(GRID, frac, (cc[-1] - cc) / np.maximum(ca[-1] - ca, 1)) - TX; tx_rem[-1] = 0
    curvas[f'{uf}-{cd}'] = [np.round(sh_cnt, 4).tolist(), np.round(sh_rem, 4).tolist(), np.round(tx_cnt, 4).tolist(), np.round(tx_rem, 4).tolist()]
m['chave'] = m.SG_UF + '-' + m.CD_MUNICIPIO.astype(str)
m = m.merge(cfgdf[['SG_UF', 'CD_MUNICIPIO', 'nome', 'cd5']], on=['SG_UF', 'CD_MUNICIPIO'], how='outer', indicator=True)
print(m._merge.value_counts().to_dict())
print('sem gabarito (só no config):', m[m._merge == 'right_only'][['SG_UF', 'nome']].values.tolist()[:10])
m = m[m._merge != 'left_only'].drop(columns='_merge')
saida = {'A': {'n': A, 'nome': ('Flávio Bolsonaro' if ANO == '2026' else 'Lula')}, 'B': {'n': B, 'nome': ('Lula' if ANO == '2026' else 'Bolsonaro')}, 'grid': GRID.tolist(), 'final_1t': {'A': float(b.vA_1.sum() / b.vNom_1.sum()), 'B': float(b.vB_1.sum() / b.vNom_1.sum()), 'A_2c': float(b.vA_1.sum() / (b.vA_1.sum() + b.vB_1.sum()))},
         'municipios': [], 'curvas': curvas}
for r in m.itertuples():
    saida['municipios'].append({'uf': r.SG_UF, 'cd5': r.cd5, 'nome': r.nome, 'chave': r.chave, 'X1': None if pd.isna(r.X1) else round(r.X1, 5), 'TX1': None if pd.isna(r.TX1) else round(r.TX1, 5), 'VR1': None if pd.isna(r.VR1) else round(r.VR1, 5),
                               'nom1': None if pd.isna(r.nom1) else int(r.nom1), 'te2': None if pd.isna(r.te2) else int(r.te2), 'lat': None if pd.isna(r.lat) else round(r.lat, 4), 'lon': None if pd.isna(r.lon) else round(r.lon, 4)})
json.dump(saida, open('work/modelo_' + ANO + '.json', 'w'), ensure_ascii=False, separators=(',', ':'))
import os; print('modelo_2026.json', os.path.getsize('work/modelo_' + ANO + '.json') // 1024, 'KB |', len(saida['municipios']), 'municípios |', len(curvas), 'curvas | eleitorado 2t', int(m.te2.sum()))
