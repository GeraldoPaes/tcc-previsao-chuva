
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


def nested_cross_validation_grid_search(lista_modelos, X, y, k_folds_outer=5, k_folds_inner=5, rand_state=42):
    """
    Realiza validação cruzada aninhada com busca em grade para múltiplos modelos de classificação.

    Parâmetros:
    -----------
    lista_modelos : list of dict
        Lista de dicionários onde cada um contém:
        - 'nome_do_modelo': string com nome do modelo
        - 'estimador': objeto do classificador
        - 'parametros': dicionário com parâmetros para grid search
    X : pandas.DataFrame
        Features do conjunto de dados
    y : pandas.Series
        Rótulos das classes
    k_folds_outer : int, default=5
        Número de folds para validação externa
    k_folds_inner : int, default=5
        Número de folds para validação interna (grid search)
    rand_state : int, default=42
        Semente aleatória para reprodutibilidade

    Retorna:
    --------
    dict
        Dicionário com resultados detalhados para cada modelo
    """

    print(f"\n\n\n **** RESULTADO DOS MODELOS ****\n")

    resultados_gerais = {}
    classes_unicas = np.unique(y)
    n_classes = len(classes_unicas)

    print(f"Número de classes identificadas: {n_classes}")
    print(f"Classes: {classes_unicas}\n")

    # Validação inicial dos dados
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo número de amostras")
    
    for classe in classes_unicas:
        if sum(y == classe) < k_folds_outer:
            raise ValueError(f"Classe {classe} tem menos amostras ({sum(y == classe)}) que o número de folds ({k_folds_outer})")

    for mdl in lista_modelos:
        nome_do_modelo = mdl["nome_do_modelo"]
        estimador_base = mdl.get('estimador')
        parametros = mdl.get('parametros')

        # Métricas por fold
        metricas_por_fold = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'tempo_treino': []
        }

        print(f"Treinando modelo {nome_do_modelo} ", end="")

        cv_outer = StratifiedKFold(n_splits=k_folds_outer, shuffle=True, random_state=rand_state)
        best_model_params = []
        best_trained_models = []

        try:
            for fold_idx, (train_ix, test_ix) in enumerate(cv_outer.split(X, y), 1):
                print(f"{fold_idx}", end=".")

                # Separação dos dados
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  
                y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                # Configuração e execução do GridSearchCV
                grid_search = GridSearchCV(
                    estimator=estimador_base,
                    param_grid=parametros,
                    scoring='f1_weighted',
                    cv=StratifiedKFold(n_splits=k_folds_inner, shuffle=True, random_state=rand_state),
                    n_jobs=4,
                    verbose=0
                )

                # Treinamento com medição de tempo
                inicio_treino = time.time()
                grid_search.fit(X_train, y_train)
                tempo_treino = time.time() - inicio_treino

                # Clonagem e configuração do melhor modelo
                melhor_modelo = clone(grid_search.best_estimator_)
                melhor_modelo.set_params(**grid_search.best_params_)
                melhor_modelo.fit(X_train, y_train)

                # Armazenamento dos melhores parâmetros e modelos
                best_model_params.append(grid_search.best_params_)
                best_trained_models.append(melhor_modelo)

                # Avaliação do modelo
                y_pred = melhor_modelo.predict(X_test)
                
                # Cálculo das métricas
                metricas_por_fold['accuracy'].append(
                    metrics.accuracy_score(y_test, y_pred)
                )
                metricas_por_fold['precision'].append(
                    metrics.precision_score(y_test, y_pred, average='weighted')
                )
                metricas_por_fold['recall'].append(
                    metrics.recall_score(y_test, y_pred, average='weighted')
                )
                metricas_por_fold['f1'].append(
                    metrics.f1_score(y_test, y_pred, average='weighted')
                )
                metricas_por_fold['tempo_treino'].append(tempo_treino)

                # Matriz de confusão para o fold atual (opcional)
                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                print(f"\nMatriz de Confusão - Fold {fold_idx}:")
                print(conf_matrix)

        except Exception as e:
            print(f"\nErro durante o treinamento do modelo {nome_do_modelo}: {str(e)}")
            continue

        print("\n-- Processando resultados finais --\n")

        # Cálculo das estatísticas finais
        resultados_finais = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metricas_por_fold.items()
        }

        # Impressão dos resultados
        print(f"Resultados para {nome_do_modelo}:")
        print(f" - Acurácia   : {resultados_finais['accuracy']['mean']:.4f} +/- {resultados_finais['accuracy']['std']:.4f}")
        print(f" - Precisão   : {resultados_finais['precision']['mean']:.4f} +/- {resultados_finais['precision']['std']:.4f}")
        print(f" - Revocação  : {resultados_finais['recall']['mean']:.4f} +/- {resultados_finais['recall']['std']:.4f}")
        print(f" - F1-Score   : {resultados_finais['f1']['mean']:.4f} +/- {resultados_finais['f1']['std']:.4f}")
        print(f" - Tempo médio: {resultados_finais['tempo_treino']['mean']:.2f} segundos")
        print('=' * 50, '\n')

        # Armazenamento dos resultados
        resultados_gerais[nome_do_modelo] = {
            **{f"{metric}_mean": stats['mean'] for metric, stats in resultados_finais.items()},
            **{f"{metric}_std": stats['std'] for metric, stats in resultados_finais.items()},
            "metricas_por_fold": metricas_por_fold,
            "melhores_parametros": best_model_params,
            "melhores_modelos": best_trained_models
        }

    print(f"Processamento finalizado em {time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())}")

    return resultados_gerais
