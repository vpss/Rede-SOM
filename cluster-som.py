import numpy as np
import pandas as pd
import somoclu

"""
Essa função seleciona apenas algumas colunas dos Dados, utilizando o usecols nativo da biblioteca pandas
WIP, pois precisamos definir como as colunas serão passadas como Parâmetro

"""
def select_columns(dataset_path, columns):
  if(columns=='all'):
    dataset = pd.read_csv(dataset_path, keep_default_na=False)
  else:
    dataset = pd.read_csv(dataset_path, keep_default_na=False, usecols=columns)    
  return dataset

"""
Projeto da disciplina de Inteligência Artificial - POLI - UPE


Biblioteca utilizada: somoclu
Repositório original: https://github.com/peterwittek/somoclu

"""

""" Carrega os dados """
dataset = select_columns('./colunas-normalizadas.csv', 'all')


"""Seleciona os ID's, que serão os rótulos dos neurônios no mapa"""
labels = dataset.iloc[:, 0]




"""Remove os ID's para não influenciarem no agrupamento"""
data = np.float32(dataset.iloc[:, 1:].values)
"""Se o np.float32() não for usado, será emitido o seguinte alerta durante
    a execução: Warning: data was not float32. A 32-bit copy was made
   e os dados serão transformados automaticamente para o tipo float32
   """                     




""" Quantidade de neurônios na rede.
    Os valores das linhas e colunas podem ser alterados (valores muito grandes
    exigirão muito processamento).
"""
n_rows, n_columns = 100, 100




som = somoclu.Somoclu(n_columns, n_rows, maptype="planar", gridtype="rectangular",
                      compactsupport=True, initialization="pca")
   
"""SOMOCLU: Classe para treino e visualização do SOM.
    Atributos:
        codebook     Codebook do SOM
        bmus         As BMUs(best matching points) correspondentes as dados.
    :param n_columns: Número de colunas no mapa.
    :type n_columns: int.
    :param n_rows: Número de linhas no mapa.
    :type n_rows: int.
    :param initialcodebook: Parametro opcional para inicializar o treinamento com 
                            um dado codebook.
    :type initialcodebook: 2D numpy.array of float32.
    :param kerneltype: Parametro opcional para especificar qual kernel será usado
                           * 0: dense CPU kernel (padrão)
                           * 1: dense GPU kernel 
    :type kerneltype: int.
    :param maptype: Parâmetro Opcional para especificar a topologia do mapa
                           * "planar": Mapa planar (padrão)
                           * "toroid": Mapa toróide
    :type maptype: str.
    :param gridtype: Parâmetro opcional para especificar a forma da matriz de nós
                           * "rectangular": neuronios retangulares (padrão)
                           * "hexagonal": neuronios hexagonais
    :type gridtype: str.
    :param compactsupport: Parâmetro opcional para impedir as atualizações do mapa além
                            do raio de treinamento com a vizinhança gaussiana.
                           Default: True.
    :type compactsupport: bool.
    :param neighborhood: Parâmetro opcional para especificar função de vizinhança:
                           * "gaussian":    Vizinhança Gaussiana (Padrão)
                           * "bubble": Função de vizinhança bubble
    :type neighborhood: str.
    :param std_coeff: Parâmetro opcional para definir o coeficiente função de 
                       vizinhança Gaussiana exp(-||x-y||^2/(2*(coeff*radius)^2))
                      Default: 0.5
    :type std_coeff: float.
    :param initialization: Parâmetro opcional para especificar a inicialização:
                           * "random": pesos inicializados randomicamente
                           * "pca": codebook é inicializado a partir do primeiro
                              subespaço abrangido pelos dois primeiros autovetores da
                              matriz de correlação
    :type initialization: str.
    :param verbose: Parâmetro opcional para especificar verbosidade (0, 1 ou 2).
                    Em geral, é uma opção para produzir informações detalhadas 
                    de registro.
    :type verbose: int.
    """


som.train(data)

"""Treina o mapa usando os dados atuais no objeto Somoclu.
        :param data: Parâmetro opcional para fornecer dados de treinamento. Não é
                      necessário se os dados foram adicionados através do método
                     `update_data`.
        :type data: 2D numpy.array of float32.
        :param epochs: O número de epochs para treinar o mapa.
        :type epochs: int.
        :param radius0: O raio inicial no mapa onde a atualização acontece
                         em torno de uma melhor unidade correspondente(BMU). 
                         O valor padrão de 0 desencadeará um valor 
                         de min(n_columns, n_rows)/2.
        :type radius0: float.
        :param radiusN: O raio no mapa onde a atualização acontece em torno de uma
                        BMU na época final. Padrão: 1.
        :type radiusN: float.
        :param radiuscooling: A estratégia de "suavização" entre radius0 e radiusN:
                                   * "linear": Interpolação linear (padrão)
                                   * "exponential": Decaimento exponencial
        :param scale0: A taxa inicial de aprendizado. Valor padrão: 0,1.
        :type scale0: float.
        :param scaleN: A taxa de aprendizado na 'epoch' final. Padrão: 0,01
        :type scaleN: float.
        :param scalecooling: A estratégia de "suavização" entre scale0 and scaleN:
                                   * "linear": Interpolação linear (padrão)
                                   * "exponential": Decaimento exponencial
        :type scalecooling: str.
        """



som.cluster()

""" Classifica os neurônios, preenchendo a variável som.clusters, também seleciona
    as BMUs(neuônios que são exibidos no mapa) para cada entrada.
    
    Se os métodos de exibição/visualização forem chamados após este método e as cores
    não forem passadas para as funções de visualização, automaticamente serão 
    atribuídas cores as BMUs com base na classificação do cluster. Ou seja, as BMUs 
    serão coloridas de acordo com  a estrutura de agrupamento.
        :param algorithm: Parâmetro opcional para especificar um scikit-learn
                           algoritmo de agrupamento. O padrão é K-means com
                           oito clusters.
        :type filename: sklearn.base.ClusterMixin.
        """  
        
              
som.view_umatrix(bestmatches=True, labels=labels, filename='./mapa.png')
"""Plota a U-Matrix do mapa treinado.

      :param figsize: Parâmetro opcional para especificar o tamanho da figura.
      :type figsize: (int, int)
      :param colormap: Parâmetro opcional para especificar o mapa de cores a ser
                        usado.
      :type colormap: matplotlib.colors.Colormap
      :param colorbar: Parâmetro opcional para incluir um mapa de cores como legenda.
      :type colorbar: bool.
      :param bestmatches: Parâmetro opcional para plotar as BMUs
      :type bestmatches: bool.
      :param bestmatchcolors: Parâmetro opcional para especificar a cor de cada
                               bmu.
      :type bestmatchcolors: list of int.
      :param labels:Parâmetro opcional para especificar o rótulo de cada ponto.
      :type labels: list of str.
      :param zoom: Parâmetro opcional para ampliar uma região no mapa. o
                    as duas primeiras coordenadas da tupla são os limites da linha,
                    segunda tupla contém os limites da coluna.
      :type zoom: ((int, int), (int, int))
      :param filename: Se especificado, o gráfico não será mostrado mas salvo
                        neste arquivo.
      :type filename: str.
      """

np.savetxt("./clusters.csv", som.clusters, delimiter=",")
""" som.bmus possui as coordenadas das BMUs, que são as 
    células que conseguimos ver no mapa.
    
    som.clusters(arquivo cluster.csv) é o resultado do método som.cluster() e 
    possui a classificação de cada neurônio do mapa(ao total: (n_row * n_columns) 
    neurônios).
    
    Como a localização dos neurônios após o treinamento é fixa no mapa, através 
    das coordenadas das BMUs em som.bmus, é possível extrair as suas respectivas 
    classificações em som.clusters apenas em função de suas coordenadas.
    """
    
    

clusters = pd.read_csv('./clusters.csv')
id_classes = np.empty((len(data),2), dtype=int)
id_class = pd.DataFrame()
""" id_classes: array utilizado para armazenar as classes resultantes da cluesterização,
    será usado para salvar o arquivo classes.csv no formato:
        ID CLASSE 
        ..   ..
        ..   ..
        ..   ..
        ..   ..
"""


i=-1
for linha, coluna in som.bmus:
    i=i+1
    id_classes[i][0] = labels[i]                      #id
    id_classes[i][1] = som.clusters[linha][coluna]    #classe
        
output = pd.DataFrame(id_classes, columns=['ID', 'Classe'])
output.to_csv('./classes.csv', sep=',', index=False)



f= open("./bmus.txt","w+")
"""escreve as coordenadas de cada bmu para arquivo "bmus.txt"""

i=1
for x,y in som.bmus:
    print(("ID %d: (%d, %d)\n" % (i, x, y)), file=f)
    i=i+1