# MNIST-AlexNet
Treinamento e validação de uma rede neural AlexNet aplicada ao dataset MNIST usando PyTorch.

## Descrição

Este projeto implementa uma **rede neural do tipo AlexNet** para a tarefa de **classificação de dígitos manuscritos** no conjunto de dados **MNIST**.  

O objetivo é explorar o uso de **arquiteturas clássicas de deep learning** aplicadas a um dataset simples, avaliando o desempenho da rede e o impacto do **número de épocas na acurácia e no erro acumulado**.

## Tecnologias 

- **Python 3.7.12**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib**

# Estrutura

## Dataset e Dataloader

Primeiro, definimos um **transformador** configurado para converter as imagens em **tensores** com dimensões `(1, 100, 100)`.

Tanto os dados de treino quanto os de validação são provenientes do pacote **Torchvision**, onde criamos duas variáveis: `trainset` e `valset`.

Para agilizar o processo de treinamento, reduzimos a quantidade de imagens utilizadas.  
O padrão do conjunto MNIST é composto por **60 mil imagens de treino**, mas neste projeto utilizamos **20 mil imagens para o treinamento** e **1 mil para a validação**.

O nosso **Dataloader** possui um **batch size** pré-definido de **130 imagens**, permitindo uma **atualização dos gradientes mais suave** e estável durante o aprendizado.

```python

imagem_size = 100

transformadores_imagem =  transforms.Compose([transforms.Resize(size=[imagem_size,imagem_size]),transforms.ToTensor()])

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transformadores_imagem)  
valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transformadores_imagem)  

train_subset = Subset(trainset, range(20000)) 
val_subset = Subset(valset, range(1000)) 

trainloader = DataLoader(train_subset, batch_size=130, shuffle=True)
valloader = DataLoader(val_subset, batch_size=130, shuffle=True)

```

## Alexnet

Utilizaremos o modelo **AlexNet** presente no pacote **PyTorch**, com o pré-treinamento desativado.


```python

alexnet =  models.alexnet(pretrained =False)

```

```markdown
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
Ao analisar a estrutura da **AlexNet**, nos deparamos com alguns ajustes necessários:

- A rede foi originalmente projetada para **imagens RGB (3 canais)**, enquanto o dataset **MNIST** possui **imagens em escala de cinza (1 canal)**.  
- O número de **saídas (`out_features`)** da camada final é **1000**, diferente do número de **classes do MNIST (10)**.

Para viabilizar o projeto, realizamos as seguintes modificações na arquitetura da rede neural:

1. Alteramos as dimensões de entrada na **camada 0** do bloco `features` para aceitar apenas **1 canal**.  
2. Modificamos a **camada 6** do bloco `classifier`, ajustando o parâmetro `out_features` para **10 classes**.  
3. Incluímos um módulo **`LogSoftmax`**, responsável por converter as saídas em **probabilidades na escala logarítmica**.


```python
alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
alexnet.classifier[6] = nn.Linear(4096, numero_de_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim=1))
```

### Otimizador

Para acelerar a convergência da rede, utilizamos o **otimizador Adam**, amplamente empregado em projetos desse tipo.

Definimos uma taxa de aprendizado (`lr`) de **0.001** e um `weight_decay` de **0.0000**.

```python
otimizador = optim.Adam(alexnet.parameters(), lr=0.001, weight_decay=0.0000)
```


## Treino e validação 

Criamos 3 funções para o processo de validação e treinamento:

### Função `treinar_modelo(modelo, metrica_erro, otimizador_sgd)`

Essa função realiza o **processo de treinamento** da **Alexnet**, ajustando os pesos da rede a cada iteração sobre os dados de treino. 

O modelo é colocado em modo de treinamento (`modelo.train()`), e o loop percorre os lotes do `trainloader`. Os gradientes acumulados são zerados a cada iteração. Em seguida, o modelo realiza o **forward pass**, gerando as previsões, e o erro é calculado por meio do NNlosss. O **backpropagation** é então executado para atualizar os pesos através do otimizador Adam. 

A função retorna quatro valores:  

- `erro_treino`: erro total acumulado durante o treinamento;  
- `acuracia_treino`: acurácia média obtida;  
- `hist_acuracia`: lista com a acurácia de cada lote;  
- `hist_erro`: lista com o erro de cada lote.  

```python
def treinar_modelo (modelo,metrica_erro, otimizador_sgd):
    erro_treino , acuracia_treino = 0.0 , 0.0
    modelo.train()
    hist_acuracia = []
    hist_erro = []

    for i, (entradas, labels) in enumerate(trainloader):

        entradas = entradas.to(device)
        labels = labels.to(device)

        otimizador_sgd.zero_grad()
        saidas = modelo(entradas)
        erro = metrica_erro(saidas, labels)

        erro.backward()

        otimizador_sgd.step()
        erro_treino += erro.item() * entradas.size(0)

        valores_maximos, indices_dos_valores_maximos = torch.max(saidas.data, 1)
        predicoes_corretas = indices_dos_valores_maximos.eq(labels.data.view_as(indices_dos_valores_maximos))

  
        acuracia = torch.mean(predicoes_corretas.type(torch.FloatTensor))


        acuracia_treino += acuracia.item() * entradas.size(0)

        hist_acuracia.append(acuracia.item())
        hist_erro.append(erro.item())


        print("Treino - Lote número {:03d}, Erro: {:.4f}, Acurácia: {:.4f}".format(i, erro.item(), acuracia.item()))

    return erro_treino, acuracia_treino, hist_acuracia, hist_erro
```

### Função `validar_modelo(modelo, metrica_erro)`

Essa função executa o **processo de validação**, avaliando seu desempenho em dados não utilizados durante o treinamento, sem realizar atualizações nos pesos da rede.

O modelo é colocado em modo de avaliação (`modelo.eval()`), e o cálculo dos gradientes é desativado com o bloco `torch.no_grad()` para economizar memória e tempo de execução. Em seguida, o loop percorre os lotes do conjunto `valloader`. O modelo realiza o **forward pass** e o erro é calculado . Para cada lote, são identificadas as previsões corretas e calculada a **acurácia** correspondente.

A função retorna quatro valores:  

- `acuracia_validacao`: acurácia total obtida no conjunto de validação;  
- `erro_validacao`: erro total acumulado durante a validação;  
- `hist_acuracia`: lista com a acurácia de cada lote;  
- `hist_erro`: lista com o erro de cada lote. 


```python
def validar_modelo(modelo,metrica_erro):
    erro_validacao, acuracia_validacao = 0.0 , 0.0
    hist_acuracia = []
    hist_erro = []

    with torch.no_grad():
            
            modelo.eval()

            for j, (entradas, labels) in enumerate(valloader):
                entradas = entradas.to(device)
                labels = labels.to(device)

                saidas = modelo(entradas)


                erro = metrica_erro(saidas, labels)

                erro_validacao += erro.item() * entradas.size(0)

                valores_maximos, indices_dos_valores_maximos = torch.max(saidas.data, 1)
                predicoes_corretas = indices_dos_valores_maximos.eq(labels.data.view_as(indices_dos_valores_maximos))

                acuracia = torch.mean(predicoes_corretas.type(torch.FloatTensor))

                acuracia_validacao += acuracia.item() * entradas.size(0)

                hist_acuracia.append(acuracia.item())
                hist_erro.append(erro.item())

                print("Validação - Lote número: {:03d}, Erro: {:.4f}, Acurácia: {:.4f}".format(j, erro.item(), acuracia.item()))



    return acuracia_validacao , erro_validacao , hist_acuracia, hist_erro
```
### Função `treinar_e_validar(modelo, metrica_erro, otimizador_sgd, epocas=10)`

Executa o **ciclo completo de treinamento e validação** do modelo ao longo de várias épocas.  
A cada época, chama as funções `treinar_modelo` e `validar_modelo`, calcula as médias de erro e acurácia para ambos os conjuntos e armazena os resultados em históricos.  

# Resultados

Realizamos um treinamento com **4 épocas**, obtendo uma **acurácia final de validação de 97.90%** e um **erro de 0.0796**.  

O resultado é bastante expressivo, considerando o **pequeno número de épocas** e o fato de termos utilizado **apenas 20 mil imagens** do conjunto original do MNIST.

## Treino

**Acuracia Treinamento**
- <img src="output\102a053b-4fdb-451e-833a-d801a89a4360.png" alt="Saída" width="800"/>

**Erro Treinamento**
- <img src="output\f487baf5-3fe3-4890-9189-623a017975b9.png" alt="Saída" width="800"/>

## Validação

**Acuracia Validação**
- <img src="output\a4d31797-30d9-494d-b72f-8f59e9a9a4d3.png" alt="Saída" width="800"/>

**Erro  Validação**
- <img src="output\eff24fc2-4950-41e3-a982-b500d9ffda8e.png" alt="Saída" width="800"/>

## Matriz de Confusão 
<img src="output\ae4cea23-802d-4c95-9e83-c2ea85725253.png" alt="Saída" width="500"/>
