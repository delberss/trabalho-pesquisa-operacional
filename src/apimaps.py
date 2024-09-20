import googlemaps
import csv

# Chave de API do Google Maps
API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
gmaps = googlemaps.Client(key=API_KEY)

# Lista de endereços
enderecos = [
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

def geocodificar_endereco(endereco):
    geocode_result = gmaps.geocode(endereco)
    if geocode_result:
        localizacao = geocode_result[0]['geometry']['location']
        return localizacao['lat'], localizacao['lng']
    else:
        raise Exception(f"Erro ao geocodificar o endereço: {endereco}")

def calcular_distancia(origem, destino):
    matriz_resultado = gmaps.distance_matrix(origem, destino, units='metric')
    if matriz_resultado['status'] == 'OK':
        distancia = matriz_resultado['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convertendo para km
        return distancia
    else:
        raise Exception(f"Erro ao calcular a distância entre {origem} e {destino}")

# Geocodificar todos os endereços
coordenadas = [geocodificar_endereco(endereco) for endereco in enderecos]

# Calcular distâncias entre todos os pares de endereços e salvar em um arquivo CSV
with open('distancias.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Origem', 'Destino', 'Distância (km)'])
    for i, origem in enumerate(enderecos):
        for j, destino in enumerate(enderecos):
            if i != j:
                distancia = calcular_distancia(coordenadas[i], coordenadas[j])
                # Formatar a distância para ter uma precisão adequada
                distancia_formatada = f"{distancia:.2f}"
                writer.writerow([origem, destino, distancia_formatada])
                print(f"Distância entre {origem} e {destino}: {distancia_formatada} km")
