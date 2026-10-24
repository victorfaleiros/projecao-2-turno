"""Monta a tabela por seção (chave UF+município+zona+seção) com 1º turno, 2º turno (se houver), hora de totalização e coordenadas."""
import sys, numpy as np, pandas as pd
ano = sys.argv[1]; A, B = (13, 22) if ano == '2022' else (22, 13)   # A = candidato que vai ser medido (2022: Lula 13; 2026: Flávio 22)
K = ['SG_UF','CD_MUNICIPIO','NR_ZONA','NR_SECAO']
v = pd.read_parquet(f'work/{ano}_votos.parquet'); d = pd.read_parquet(f'work/{ano}_detalhe.parquet'); lv = pd.read_parquet(f'work/{ano}_lv.parquet')
v['CD_MUNICIPIO'] = v.CD_MUNICIPIO.astype(int); d['CD_MUNICIPIO'] = d.CD_MUNICIPIO.astype(int); lv['CD_MUNICIPIO'] = lv.CD_MUNICIPIO.astype(int)
def turno(t):
    vt = v[v.NR_TURNO==t]
    piv = vt.pivot_table(index=K, columns='NR_VOTAVEL', values='QT_VOTOS', aggfunc='sum', fill_value=0)
    out = pd.DataFrame({'vA': piv.get(A, 0), 'vB': piv.get(B, 0)}, index=piv.index)
    cand_cols = [c for c in piv.columns if c < 90]
    out['vNom'] = piv[cand_cols].sum(axis=1)
    dt = d[d.NR_TURNO==t].drop_duplicates(K).set_index(K)
    out = out.join(dt[['QT_APTOS','QT_COMPARECIMENTO','QT_VOTOS_NOMINAIS','QT_VOTOS_BRANCOS','QT_VOTOS_NULOS','NR_LOCAL_VOTACAO','DT_PRIM_TOT_PARCIAL_HOR_TSE','DT_RECEBIMENTO_BU_HOR_TSE','ST_SECAO_INSTALADA']], how='outer')
    out['t_tot'] = pd.to_datetime(out.DT_PRIM_TOT_PARCIAL_HOR_TSE, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    out['t_rec'] = pd.to_datetime(out.DT_RECEBIMENTO_BU_HOR_TSE, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    out = out.rename(columns={'QT_APTOS':'aptos','QT_COMPARECIMENTO':'comp','QT_VOTOS_NOMINAIS':'nom','QT_VOTOS_BRANCOS':'br','QT_VOTOS_NULOS':'nu','NR_LOCAL_VOTACAO':'local'})
    out = out.drop(columns=['DT_PRIM_TOT_PARCIAL_HOR_TSE','DT_RECEBIMENTO_BU_HOR_TSE'])
    return out.add_suffix(f'_{t}')
t1 = turno(1)
base = t1.join(turno(2), how='outer') if (v.NR_TURNO==2).any() else t1
# coordenadas: por seção; se faltar, média do local; se faltar, média do município
lvt = lv[lv.NR_TURNO==lv.NR_TURNO.max()].drop_duplicates(K).set_index(K)
lvt['lat'] = lvt.NR_LATITUDE.where((lvt.NR_LATITUDE>-60)&(lvt.NR_LATITUDE<20)&(lvt.NR_LATITUDE!=-1))
lvt['lon'] = lvt.NR_LONGITUDE.where((lvt.NR_LONGITUDE>-80)&(lvt.NR_LONGITUDE<-25))
base = base.join(lvt[['lat','lon','NR_LOCAL_VOTACAO','QT_ELEITOR_SECAO','NM_MUNICIPIO']].rename(columns={'NR_LOCAL_VOTACAO':'local_lv','QT_ELEITOR_SECAO':'eleitores_lv'}), how='left')
base = base.reset_index()
base['local'] = base.local_2.fillna(base.local_1).fillna(base.local_lv) if 'local_2' in base else base.local_1.fillna(base.local_lv)
base['local_id'] = base.SG_UF + '-' + base.CD_MUNICIPIO.astype(str) + '-' + base.NR_ZONA.astype(int).astype(str) + '-' + base.local.fillna(-1).astype(int).astype(str)
base['mun_id'] = base.SG_UF + '-' + base.CD_MUNICIPIO.astype(str)
for c in ['lat','lon']:
    base[c] = base[c].fillna(base.groupby('local_id')[c].transform('mean')).fillna(base.groupby('mun_id')[c].transform('mean'))
base['mun_nome'] = base.NM_MUNICIPIO
print('seções', len(base), '| com t1', base.aptos_1.notna().sum(), '| com t2', base.aptos_2.notna().sum() if 'aptos_2' in base else 0, '| com coord', base.lat.notna().mean().round(3), flush=True)
if 'aptos_2' in base:
    ok = base.t_tot_2.notna()
    print('t2: totalizadas com hora', ok.sum(), '| A =', base.vA_2.sum()/base.vNom_2.sum(), '| 1º turno A =', base.vA_1.sum()/base.vNom_1.sum())
    print(base.loc[ok,'t_tot_2'].dt.floor('30min').value_counts().sort_index().cumsum()/ok.sum())
base.to_parquet(f'work/{ano}_base.parquet'); print('FIM base', ano)
