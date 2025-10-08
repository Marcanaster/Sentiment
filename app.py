from flask import Flask, request, jsonify, render_template
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# --- 1. CONFIGURAÇÃO DA API E CARREGAMENTO DOS COMPONENTES ---

app = Flask(__name__)

# Definir os caminhos para os arquivos salvos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_regressao_logistica.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizador.pkl')
# Carregar o modelo e o vetorizador na memória quando a API iniciar
try:
    modelo = joblib.load(MODEL_PATH)
    vetorizador = joblib.load(VECTORIZER_PATH)
    print("Modelo e Vetorizador carregados com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar o modelo/vetorizador: {e}")
    # A API não deve rodar sem os componentes essenciais
    exit()

# --- 2. FUNÇÕES DE PRÉ-PROCESSAMENTO (IDÊNTICAS AO NOTEBOOK) ---
stopwords_en = set(stopwords.words('english'))
stopwords_pt = set(stopwords.words('portuguese'))

lista_stopwords = stopwords_en.union(stopwords_pt)

def limpar_texto(texto):
    """Função para remover HTML e caracteres não-alfabéticos."""
    soup = BeautifulSoup(texto, 'html.parser')
    texto_limpo = soup.get_text()
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto_limpo)
    return texto_limpo.lower()

def remover_stopwords(texto):
    """Função para remover as stopwords e tokenizar."""
    palavras = word_tokenize(texto)
    palavras_filtradas = [palavra for palavra in palavras if palavra not in lista_stopwords]
    return " ".join(palavras_filtradas)

@app.route('/')
def index():
    """Rota para servir a página principal da aplicação."""
    return render_template('index.html')

# --- 3. ROTA DA API ---

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Endpoint que recebe um texto e retorna a classificação de sentimento."""
    
    # 1. Obter os dados enviados na requisição (JSON)
    data = request.get_json(force=True)
    texto_recebido = data.get('text', '')

    if not texto_recebido:
        return jsonify({'error': 'Nenhum texto fornecido'}), 400

    # 2. Pré-processar o texto de entrada
    texto_limpo = limpar_texto(texto_recebido)
    texto_processado = remover_stopwords(texto_limpo)

    # 3. Vetorizar o texto
    # O vetorizador espera uma lista, então passamos [texto_processado]
    vetor_texto = vetorizador.transform([texto_processado])

    # 4. Fazer a Previsão
    previsao_numerica = modelo.predict(vetor_texto)[0] # [0] pega o resultado da lista
    
    # 5. Converter o resultado para o rótulo de texto
    sentimento = 'Positivo' if previsao_numerica == 1 else 'Negativo'

    # 6. Retornar o resultado
    return jsonify({
        'status': 'sucesso',
        'texto_original': texto_recebido,
        'sentimento_previsto': sentimento
    })

# --- 4. INICIAR APLICAÇÃO ---

if __name__ == '__main__':
    # Para o NLTK funcionar no servidor, ele precisa dos dados
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('portuguese')

    # Roda a API no modo debug (para desenvolvimento)
    app.run(debug=True, host='0.0.0.0', port=5000)