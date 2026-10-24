import sys, zipfile, io, pandas as pd
ano = sys.argv[1]; base = 'dl/'
out = f'work/{ano}'
def biggest_csv(zpath):
    z = zipfile.ZipFile(zpath); m = max([i for i in z.infolist() if i.filename.lower().endswith('.csv')], key=lambda i: i.file_size)
    print(zpath, '->', m.filename, m.file_size, flush=True); return z, m
KEY = ['NR_TURNO','SG_UF','CD_MUNICIPIO','NR_ZONA','NR_SECAO']
# 1) votos por candidato por seção (Presidente)
z, m = biggest_csv(f'{base}votacao_secao_{ano}_BR.zip')
parts = []
with z.open(m) as fh:
    for ch in pd.read_csv(io.TextIOWrapper(fh, encoding='latin-1'), sep=';', quotechar='"', usecols=KEY+['CD_CARGO','NR_VOTAVEL','QT_VOTOS','NR_LOCAL_VOTACAO'], dtype=str, chunksize=1_500_000):
        ch = ch[ch.CD_CARGO=='1']; parts.append(ch.drop(columns='CD_CARGO'))
        print('votos chunk', sum(len(p) for p in parts), flush=True)
v = pd.concat(parts); 
for c in ['NR_TURNO','NR_ZONA','NR_SECAO','NR_VOTAVEL','QT_VOTOS','NR_LOCAL_VOTACAO']: v[c] = v[c].astype(int)
v.to_parquet(f'{out}_votos.parquet'); print('votos', v.shape, v.NR_TURNO.value_counts().to_dict(), flush=True)
# 2) detalhe (aptos, comparecimento, brancos, nulos, hora de totalização)
z, m = biggest_csv(f'{base}detalhe_votacao_secao_{ano}.zip')
parts = []
cols = KEY+['CD_CARGO','QT_APTOS','QT_COMPARECIMENTO','QT_ABSTENCOES','QT_VOTOS_NOMINAIS','QT_VOTOS_BRANCOS','QT_VOTOS_NULOS','NR_LOCAL_VOTACAO','DT_RECEBIMENTO_BU_HOR_TSE','DT_PRIM_TOT_PARCIAL_HOR_TSE','ST_SECAO_INSTALADA','ST_SECAO_ANULADA']
with z.open(m) as fh:
    for ch in pd.read_csv(io.TextIOWrapper(fh, encoding='latin-1'), sep=';', quotechar='"', usecols=cols, dtype=str, chunksize=1_500_000):
        ch = ch[ch.CD_CARGO=='1']; parts.append(ch.drop(columns='CD_CARGO'))
        print('detalhe chunk', sum(len(p) for p in parts), flush=True)
d = pd.concat(parts)
for c in ['NR_TURNO','NR_ZONA','NR_SECAO','QT_APTOS','QT_COMPARECIMENTO','QT_ABSTENCOES','QT_VOTOS_NOMINAIS','QT_VOTOS_BRANCOS','QT_VOTOS_NULOS','NR_LOCAL_VOTACAO']: d[c] = pd.to_numeric(d[c], errors='coerce')
d.to_parquet(f'{out}_detalhe.parquet'); print('detalhe', d.shape, d.NR_TURNO.value_counts().to_dict(), flush=True)
# 3) coordenadas dos locais de votação
z, m = biggest_csv(f'{base}lv{ano}.zip')
with z.open(m) as fh:
    lv = pd.read_csv(io.TextIOWrapper(fh, encoding='latin-1'), sep=';', quotechar='"', usecols=['NR_TURNO','SG_UF','CD_MUNICIPIO','NR_ZONA','NR_SECAO','NR_LOCAL_VOTACAO','NR_LATITUDE','NR_LONGITUDE','QT_ELEITOR_SECAO','NM_MUNICIPIO'], dtype=str)
for c in ['NR_TURNO','NR_ZONA','NR_SECAO','NR_LOCAL_VOTACAO','QT_ELEITOR_SECAO']: lv[c] = pd.to_numeric(lv[c], errors='coerce')
for c in ['NR_LATITUDE','NR_LONGITUDE']: lv[c] = pd.to_numeric(lv[c].str.replace(',','.'), errors='coerce')
lv.to_parquet(f'{out}_lv.parquet'); print('lv', lv.shape, 'com coord:', ((lv.NR_LATITUDE>-90)&(lv.NR_LATITUDE!=-1)).mean(), flush=True)
print('FIM', ano)
