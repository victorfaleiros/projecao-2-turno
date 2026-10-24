# Projeção do 2º turno 2026

Página pública que projeta o resultado final do 2º turno enquanto o TSE apura, município a município, usando o 1º turno como gabarito.

- Página: `index.html` (GitHub Pages em https://victorfaleiros.github.io/projecao-2-turno/)
- Dados ao vivo: `https://retratos-ia.duckdns.org/projecao/dados.json`, gravado pelo robô `motor/projetar.py` que roda no VPS (`/opt/projecao`, serviço `projecao.service`, ligado pelo `projecao.timer` às 16:50 de domingo).
- `motor/preparar.py` gera o arquivo do modelo (`modelo_2026.json`) a partir dos dados abertos do TSE (votação por seção, detalhe por seção e locais de votação).
- `motor/reprise.py` reprisa o 2º turno de 2022 na ordem real de totalização para comparar modelos e calibrar a faixa de incerteza.

Método, em uma frase: para cada município, compara-se o que já foi apurado com o resultado dele no 1º turno (deslocamento), projeta-se o que falta com o deslocamento medido no próprio município, nos vizinhos e no estado, corrigindo o viés da ordem em que as seções chegam, e soma-se tudo.
