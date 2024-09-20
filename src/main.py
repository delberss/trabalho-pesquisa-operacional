import pulp as lp
import csv

# Ler os dados
def ler_dados(arquivo):
    dados = {}
    with open(arquivo, 'r') as f:
        for linha in f:
            if "=" in linha:
                chave, valor = linha.split('=')
                chave = chave.strip()
                valor = valor.strip()
                if ',' in valor:  # Se for uma lista de valores
                    valor = [float(x) if '.' in x else int(x) for x in valor.split(',')]
                elif '.' in valor:  # Se for um valor float
                    valor = float(valor)
                else:  # Caso contrário, é inteiro
                    valor = int(valor)
                dados[chave] = valor
    return dados

# Ler os dados do arquivo
dados = ler_dados('dados.txt')

# Atribuição dos dados lidos para variáveis
num_veiculos = dados['num_veiculos']
num_pontos = dados['num_pontos']
num_dias = dados['num_dias']
demanda = dados['demanda']
tempo_maximo = dados['tempo_maximo']
capacidade_veiculo = dados['capacidade_veiculo']
preco_gasolina = dados['preco_gasolina']
consumo_veiculo = dados['consumo_veiculo']
manuntencao_veiculo = dados['manuntencao_veiculo']
velocidade_media = dados['velocidade_media']

# Cálculo do custo por veículo com base no consumo de combustível e manutenção
custo_veiculo = [(preco_gasolina / consumo + manutencao) if consumo != 0 else 0
                 for consumo, manutencao in zip(consumo_veiculo, manuntencao_veiculo)]



# Remover espaços extras e padronizar endereços
def padronizar_endereco(endereco):
    return ' '.join(endereco.split()).lower()

# Carregar os endereços do arquivo
def ler_enderecos(arquivo):
    enderecos = []
    with open(arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            enderecos.append(padronizar_endereco(linha.strip()))
    return enderecos

# Ler os endereços do arquivo
enderecos = ler_enderecos('enderecos.txt')

# Carregar as distâncias e associá-las aos endereços
distancia = [[0] * num_pontos for _ in range(num_pontos)]
with open('distancias.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Pular o cabeçalho
    for row in reader:
        origem = padronizar_endereco(row[0])
        destino = padronizar_endereco(row[1])
        dist = float(row[2])

        i = enderecos.index(origem)
        j = enderecos.index(destino)
        distancia[i][j] = dist

# Criar o modelo
model = lp.LpProblem("Roteamento_de_Entregas_Multiplas_Visitas", lp.LpMinimize)

# Variáveis de decisão
X = lp.LpVariable.dicts("X", (range(num_pontos), range(num_pontos), range(num_veiculos), range(num_dias)), cat='Binary')
Y = lp.LpVariable.dicts("Y", (range(num_pontos), range(num_veiculos), range(num_dias)), cat='Binary')
q = lp.LpVariable.dicts("q", (range(num_pontos), range(num_veiculos), range(num_dias)), lowBound=0, cat='Continuous')
U = lp.LpVariable.dicts("U", (range(num_pontos), range(num_veiculos), range(num_dias)), lowBound=0, upBound=num_pontos*num_dias, cat='Continuous')

# Função objetivo: minimizar o custo total de transporte
model += lp.lpSum(custo_veiculo[k] * distancia[i][j] * X[i][j][k][d]
                 for k in range(num_veiculos)
                 for i in range(num_pontos)
                 for j in range(num_pontos)
                 for d in range(num_dias))

# 1. Restrição de fluxo: Para cada ponto de entrega, o veículo que entra deve sair
for k in range(num_veiculos):
    for d in range(num_dias):
        for i in range(num_pontos):
            model += lp.lpSum(X[i][j][k][d] for j in range(num_pontos) if j != i) == lp.lpSum(X[j][i][k][d] for j in range(num_pontos) if j != i), f"fluxo_{i}_{k}_{d}"

# 2. Restrição de capacidade: Uso de q_{i,k,d}
for k in range(num_veiculos):
    for d in range(num_dias):
        model += lp.lpSum(q[i][k][d] for i in range(1, num_pontos)) <= capacidade_veiculo[k], f"capacidade_{k}_{d}"

# 3. Restrição de demanda: Assegura que cada ponto receba sua demanda total
for i in range(1, num_pontos):
    model += lp.lpSum(q[i][k][d] for k in range(num_veiculos) for d in range(num_dias)) == demanda[i], f"demanda_{i}"

# 4. Restrição de tempo: As entregas devem ser feitas dentro de um prazo de 10 horas por dia
for k in range(num_veiculos):
    for d in range(num_dias):
        model += lp.lpSum(distancia[i][j] / velocidade_media * X[i][j][k][d] for i in range(num_pontos) for j in range(num_pontos)) <= tempo_maximo, f"tempo_{k}_{d}"

# 5. Eliminação de subciclos: Restrições MTZ
for k in range(num_veiculos):
    for d in range(num_dias):
        model += U[0][k][d] == 0, f"U_depot_{k}_{d}"
        for i in range(1, num_pontos):
            for j in range(1, num_pontos):
                if i != j:
                    model += U[i][k][d] - U[j][k][d] + num_pontos * X[i][j][k][d] <= num_pontos - 1, f"MTZ_{i}_{j}_{k}_{d}"

# 6. Retorno ao ponto de origem (fábrica) no final do dia
for k in range(num_veiculos):
    for d in range(num_dias):
        # O veículo deve sair da fábrica
        model += lp.lpSum(X[0][j][k][d] for j in range(1, num_pontos)) <= 1, f"saida_fabrica_{k}_{d}"
        # O veículo deve retornar à fábrica no final do dia
        model += lp.lpSum(X[i][0][k][d] for i in range(1, num_pontos)) <= 1, f"retorno_fabrica_{k}_{d}"

# 7. Continuidade de rota: Se um veículo atende um ponto, ele deve sair do ponto
for i in range(1, num_pontos):
    for k in range(num_veiculos):
        for d in range(num_dias):
            model += lp.lpSum(X[i][j][k][d] for j in range(num_pontos) if j != i) == Y[i][k][d], f"continuidade_saida_{i}_{k}_{d}"
            model += lp.lpSum(X[j][i][k][d] for j in range(num_pontos) if j != i) == Y[i][k][d], f"continuidade_entrada_{i}_{k}_{d}"

# 8. Ligar q_{i,k,d} e Y_{i,k,d}
for i in range(1, num_pontos):
    for k in range(num_veiculos):
        for d in range(num_dias):
            model += q[i][k][d] <= capacidade_veiculo[k] * Y[i][k][d], f"q_Y_{i}_{k}_{d}"

# Resolver o modelo
status = model.solve(lp.GUROBI(msg=1))


# Verificar o status da solução
if status == lp.LpStatusOptimal:
    print("Solução ótima encontrada!")
elif status == lp.LpStatusInfeasible:
    print("O modelo é inviável.")
elif status == lp.LpStatusUnbounded:
    print("O modelo é ilimitado.")
else:
    print("Solução não encontrada.")

# Função para reconstruir a rota
def reconstruir_rota(X, k, d):
    rota = [0]  # Começa na fábrica
    atual = 0
    visitados = set()
    while True:
        encontrou = False
        for j in range(num_pontos):
            if X[atual][j][k][d].varValue is not None and X[atual][j][k][d].varValue > 0.5:
                rota.append(j)
                if j != 0:
                    visitados.add(j)
                atual = j
                encontrou = True
                break
        if not encontrou or atual == 0:
            break
    return rota


# === Gerar arquivo CSV para escrita da saída ===
with open('resultado.csv', 'w', newline='', encoding='utf-8') as arquivo_csv:
    writer = csv.writer(arquivo_csv)

    # Cabeçalho do CSV
    writer.writerow(["Veículo", "Dia", "Rota", "Endereços", "Distância Total (km)", "Valor Gasto (R$)"])

    # Exibir os resultados detalhados e escrever no CSV
    for k in range(num_veiculos):
        for d in range(num_dias):
            rota_dia = reconstruir_rota(X, k, d)
            rota_enderecos = [enderecos[i] for i in rota_dia]
            if rota_dia and len(rota_dia) > 1:
                # Calcular a distância total percorrida e o custo
                distancia_total = 0
                for i in range(num_pontos):
                    for j in range(num_pontos):
                        if X[i][j][k][d].varValue is not None and X[i][j][k][d].varValue > 0.5:
                            distancia_total += distancia[i][j]
                custo_total = distancia_total * custo_veiculo[k]

                # Escrever no CSV
                writer.writerow([k, d, rota_dia, rota_enderecos, distancia_total, custo_total])
# === fim ===


# ===Abrir um arquivo txt para escrita da saída ===
with open('resultado.txt', 'w', encoding='utf-8') as arquivo_saida:
    # Função para reconstruir a rota
    def reconstruir_rota(X, k, d):
        rota = [0]  # Começa na fábrica
        atual = 0
        while True:
            encontrou = False
            for j in range(num_pontos):
                if X[atual][j][k][d].varValue is not None and X[atual][j][k][d].varValue > 0.5:
                    rota.append(j)
                    atual = j
                    encontrou = True
                    break
            if not encontrou or atual == 0:
                break
        return rota


    # Exibir os resultados detalhados
    arquivo_saida.write("Rotas dos veículos:\n")
    for k in range(num_veiculos):
        for d in range(num_dias):
            rota_dia = reconstruir_rota(X, k, d)
            rota_enderecos = [enderecos[i] for i in rota_dia]
            if rota_dia and len(rota_dia) > 1:
                # Calcular a distância total percorrida e o custo
                distancia_total = 0
                for i in range(num_pontos):
                    for j in range(num_pontos):
                        if X[i][j][k][d].varValue is not None and X[i][j][k][d].varValue > 0.5:
                            distancia_total += distancia[i][j]
                custo_total = distancia_total * custo_veiculo[k]
                mensagem = f"\nVeículo {k} - Dia {d}:\n"
                mensagem += f"Rota: {rota_dia}\n"
                mensagem += "Endereços:\n"
                for endereco in rota_enderecos:
                    mensagem += f"{endereco}\n"
                mensagem += f"Distância total percorrida: {distancia_total} km\n"
                mensagem += f"Valor gasto: R$ {custo_total:.2f}\n"

                arquivo_saida.write(mensagem)

    # Exibir a demanda atendida por dia em cada ponto
    arquivo_saida.write("\nDemanda atendida por dia em cada ponto:\n")
    for d in range(num_dias):
        mensagem_dia = f"\nDia {d}:\n"
        arquivo_saida.write(mensagem_dia)
        for i in range(1, num_pontos):
            demanda_dia = 0
            for k in range(num_veiculos):
                if q[i][k][d].varValue is not None and q[i][k][d].varValue > 0:
                    demanda_dia += q[i][k][d].varValue
            if demanda_dia > 0:
                mensagem_ponto = f"Ponto {i} ({enderecos[i]}): Demanda atendida = {demanda_dia}\n"
                arquivo_saida.write(mensagem_ponto)

    # Exibir a demanda total atendida por ponto
    arquivo_saida.write("\nDemanda total atendida por ponto:\n")
    for i in range(1, num_pontos):
        demanda_total_atendida = 0
        for d in range(num_dias):
            for k in range(num_veiculos):
                if q[i][k][d].varValue is not None and q[i][k][d].varValue > 0:
                    demanda_total_atendida += q[i][k][d].varValue
        mensagem_total = f"Ponto {i} ({enderecos[i]}): Demanda total atendida = {demanda_total_atendida}\n"
        arquivo_saida.write(mensagem_total)

    # Exibir o custo total
    custo_total = lp.value(model.objective)
    mensagem_custo = f"\nCusto Total: R$ {custo_total:.2f}"
    arquivo_saida.write(f"\nCusto Total: R$ {custo_total:.2f}\n")


# === fim === #
# Exibir os resultados detalhados no TERMINAL
print("\nRotas dos veículos:")
for k in range(num_veiculos):
    for d in range(num_dias):
        rota_dia = reconstruir_rota(X, k, d)
        rota_enderecos = [enderecos[i] for i in rota_dia]
        if rota_dia and len(rota_dia) > 1:
            # Calcular a distância total percorrida e o custo
            distancia_total = 0
            for i in range(num_pontos):
                for j in range(num_pontos):
                    if X[i][j][k][d].varValue is not None and X[i][j][k][d].varValue > 0.5:
                        distancia_total += distancia[i][j]
            custo_total = distancia_total * custo_veiculo[k]
            print(f"\nVeículo {k} - Dia {d}:")
            print(f"Rota: {rota_dia}")
            print("Endereços:")
            for endereco in rota_enderecos:
                print(endereco)
            print(f"Distância total percorrida: {distancia_total} km")
            print(f"Valor gasto: R$ {custo_total:.2f}")

# Exibir a demanda atendida por dia em cada ponto
print("\nDemanda atendida por dia em cada ponto:")
for d in range(num_dias):
    print(f"\nDia {d}:")
    for i in range(1, num_pontos):
        demanda_dia = 0
        for k in range(num_veiculos):
            if q[i][k][d].varValue is not None and q[i][k][d].varValue > 0:
                demanda_dia += q[i][k][d].varValue
        if demanda_dia > 0:
            print(f"Ponto {i} ({enderecos[i]}): Demanda atendida = {demanda_dia}")

# Exibir a demanda total atendida por ponto
print("\nDemanda total atendida por ponto:")
for i in range(1, num_pontos):
    demanda_total_atendida = 0
    for d in range(num_dias):
        for k in range(num_veiculos):
            if q[i][k][d].varValue is not None and q[i][k][d].varValue > 0:
                demanda_total_atendida += q[i][k][d].varValue
    print(f"Ponto {i} ({enderecos[i]}): Demanda total atendida = {demanda_total_atendida}")

# Exibir o custo total
print(f"\nCusto Total: R$ {lp.value(model.objective):.2f}")
