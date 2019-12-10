import csv

'''
def carregar_acessos():
    dados = []
    marcacoes = []

    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)

    for acessou_home, acessou_como_funciona, acessou_contato, comprou in leitor:

        dados.append([acessou_home, acessou_como_funciona, acessou_contato])
        marcacoes.append(comprou)

    return dados, marcacoes
'''

def carregar_acessos():
    X = []
    Y = []

    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)

    next(leitor)

    for acessou_home, acessou_como_funciona, acessou_contato, comprou in leitor:

        X.append([int(acessou_home), int(acessou_como_funciona), int(acessou_contato)])
        Y.append(int(comprou))

    return X, Y
