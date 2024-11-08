import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        print(self.df.head())
        print(self.df.info())
        #Removendo dados nulos em cada coluna
        for name in self.df.columns.tolist():
            self.df = self.df.dropna(subset=[name])

        #Verificando se existe dado nulo em cada coluna
        for col in self.df.columns.tolist():
            print('Número de missing na coluna {}: {}'.format(col, self.df[col].isnull().sum()))

        self.df.hist(figsize=(10, 8))
       
        plt.show()

        pass

    def TreinaReg(self,X,y):
        #transformacao em codigo do dado do tipo string
        self.encoder = OneHotEncoder()
        y_encoded = self.encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

        # Regressão Linear (One-vs-Rest)
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression

        # Criar e treinar o modelo de regressão linear
        self.reg = OneVsRestClassifier(LinearRegression())
        X_train, self.X_test_reg, y_train, self.y_test_reg = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        self.reg.fit(X_train, y_train)

        pass

    def TreinaSVM(self,X,y):
        # SVM
        self.svm = SVC()
        X_train, self.X_test_svm, y_train, self.y_test_svm = train_test_split(X, y, test_size=0.2, random_state=42)

        self.svm.fit(X_train, y_train)

        pass

    def TreinaArvoreDevisao(self,X,y):
        # Dividir os dados em treinamento e teste
        X_train, self.X_test_dt, y_train, self.y_test_dt = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.arvoredecisao = DecisionTreeClassifier()
        # Treinar o modelo
        self.arvoredecisao.fit(X_train, y_train)

        pass

    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        self.X = self.df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']]  
        self.y = self.df['Species']

        self.TreinaReg(self.X, self.y)
        self.TreinaSVM(self.X, self.y)
        self.TreinaArvoreDevisao(self.X, self.y)
        

        pass


    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """
        self.ModeloLinear()
        self.ModeloSVM()
        self.ModeloArvore()


        pass

    def ModeloSVM(self):
        # Previsões com SVM
        y_pred_svm = self.svm.predict(self.X_test_svm)
        print("\nModelo SVM")
        print("\nAccuracy (SVM):", accuracy_score(self.y_test_svm, y_pred_svm))
        print("Classification Report (SVM):\n", classification_report(self.y_test_svm, y_pred_svm))
        print("Confusion Matrix (SVM):\n", confusion_matrix(self.y_test_svm, y_pred_svm))
        print("Acurácia média(SVM):", cross_val_score(self.svm, self.X, self.y, cv=5).mean())

        pass

    def ModeloLinear(self):
        # Previsões com regressão linear
        print(self.X_test_reg)
        y_pred_reg = self.reg.predict(self.X_test_reg)
        print(y_pred_reg)
#        y_pred_reg = self.encoder.inverse_transform(y_pred_reg)
        # Avaliação
        print("\nModelo Linear")
        print("Accuracy (Linear Regression):", accuracy_score(self.y_test_reg, y_pred_reg))
        print("Classification Report (Linear Regression):\n", classification_report(self.y_test_reg, y_pred_reg))
        print("Acurácia média(regressao):", cross_val_score(self.reg, self.X, self.y, cv=5).mean())

        pass

    def ModeloArvore(self):
        y_pred_dt = self.arvoredecisao.predict(self.X_test_dt)
    
        # Avaliar o modelo
        print("\nModelo Árvore de Decisão")
        print("Accuracy (Decision Tree):", accuracy_score(self.y_test_dt, y_pred_dt))
        print("\nClassification Report (Decision Tree):\n", classification_report(self.y_test_dt, y_pred_dt)) 
        print("\nConfusion Matrix (Decision Tree):\n", confusion_matrix(self.y_test_dt, y_pred_dt))        
        print("Acurácia média(arvore):", cross_val_score(self.arvoredecisao, self.X, self.y, cv=5).mean())
        pass

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento()  # Executa o treinamento do modelo

# Lembre-se de instanciar as classes após definir suas funcionalidades
# Recomenda-se criar ao menos dois modelos (e.g., Regressão Linear e SVM) para comparar o desempenho.
# A biblioteca já importa LinearRegression e SVC, mas outras escolhas de modelo são permitidas.

modelo = Modelo()
modelo.Train()
modelo.Teste()