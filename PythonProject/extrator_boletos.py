import os
import pandas as pd
import openai
from PIL import Image
from pdf2image import convert_from_path
import re
import easyocr
import cv2
import numpy as np

# CONFIGURAÇÕES
openai.api_key = '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'

PASTA_BOLETOS = "analisar_boletos"
ARQUIVO_SAIDA = "boletos_extraidos.xlsx"

print("Inicializando EasyOCR. Baixando modelos de idioma (se necessário)...")
try:
    reader = easyocr.Reader(['pt', 'en'])
    print("EasyOCR inicializado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar EasyOCR: {e}")
    print("Verifique sua conexão com a internet ou a instalação do EasyOCR.")
    exit()


def extrair_texto_arquivo(caminho):
    ext = caminho.lower().split('.')[-1]
    texto_total = ""

    if ext in ["png", "jpg", "jpeg"]:
        # EasyOCR prefere arrays NumPy (OpenCV), então carregamos com cv2
        img = cv2.imread(caminho)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem em: {caminho}")

        # Realiza o OCR na imagem
        results = reader.readtext(img)

        # Concatena todo o texto reconhecido
        for (bbox, text, prob) in results:
            texto_total += text + " "  # Adiciona um espaço para separar as palavras/frases
        return texto_total.strip()  # Remove espaços extras no início/fim

    elif ext == "pdf":
        # Para PDFs, convertemos página por página para imagem com pdf2image
        # E então passamos cada imagem para o EasyOCR
        paginas = convert_from_path(
            caminho,
            dpi=300,  # Mantém alta resolução
            # Mantenha o poppler_path se for necessário para você no Windows
            poppler_path=r"C:\Users\USER\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin"
        )

        for i, pagina in enumerate(paginas):
            # Converte a página PIL Image para array NumPy (OpenCV)
            # EasyOCR funciona melhor com cv2.imread ou np.array
            img_np = np.array(pagina)  # Converte PIL Image para NumPy array (RGB)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Converte de RGB para BGR

            print(f"  Extraindo texto da página {i + 1} do PDF...")
            results = reader.readtext(img_np)

            for (bbox, text, prob) in results:
                texto_total += text + " "
            texto_total += "\n"  # Adiciona uma quebra de linha entre páginas
        return texto_total.strip()

    else:
        return ""


def extrair_dados_gpt(texto_ocr):
    prompt = f"""
Você é um assistente de extração de dados de analisar_boletos bancários. A partir do texto OCR abaixo, extraia:

- Data de vencimento
- Fornecedor
- Pagador
- Valor

Texto OCR:
{texto_ocr}

Retorne os dados em formato JSON com as chaves: "vencimento", "fornecedor", "pagador", "valor"
    """

    resposta = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    conteudo = resposta.choices[0].message.content
    try:
        import json
        dados = json.loads(conteudo)
        return dados
    except Exception:
        return {
            "vencimento": "",
            "fornecedor": "",
            "pagador": "",
            "valor": "",
            "erro": f"Erro ao processar resposta do GPT: {conteudo}"
        }


def processar_boletos():
    resultados = []

    arquivos_na_pasta = [f for f in os.listdir(PASTA_BOLETOS) if os.path.isfile(os.path.join(PASTA_BOLETOS, f))]

    if not arquivos_na_pasta:
        print(f"Nenhum arquivo encontrado na pasta '{PASTA_BOLETOS}'. Adicione seus boletos para análise.")
        return

    for arquivo in os.listdir(PASTA_BOLETOS):
        caminho = os.path.join(PASTA_BOLETOS, arquivo)
        print(f"📄 Processando: {arquivo}")

        try:
            texto = extrair_texto_arquivo(caminho)
            dados = extrair_dados_gpt(texto)
            dados["arquivo"] = arquivo
            resultados.append(dados)

        except Exception as e:
            print(f"❌ Erro ao processar {arquivo}: {e}")
            resultados.append({
                "vencimento": "",
                "fornecedor": "",
                "pagador": "",
                "valor": "",
                "arquivo": arquivo,
                "erro": str(e)
            })
    if resultados:
        df = pd.DataFrame(resultados)
        df.to_excel(ARQUIVO_SAIDA, index=False)
        print(f"\n✅ Dados salvos em: {ARQUIVO_SAIDA}")
    else:
        print("\nNenhum boleto processado com sucesso para salvar na planilha.")

if __name__ == "__main__":
    processar_boletos()
