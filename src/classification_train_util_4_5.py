
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn import metrics

from sklearn.base import clone


def nested_cross_validation_grid_search(lista_modelos, X, k_folds_outer=5, k_folds_inner=5, rand_state=42):
    print(f"\n\n\n **** RESULTADO DOS MODELOS + CURVAS ROC E PR ****\n")

    resultados_gerais = {}

    for mdl in lista_modelos:
        nome_do_modelo = mdl["nome_do_modelo"]
        estimador_base = mdl.get('estimador')
        parametros = mdl.get('parametros')

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        auc_score_list = []
        aucpr_score_list = []

        # Inicializar listas para curvas ROC e PR
        roc_fpr_list = []
        roc_tpr_list = []
        pr_precision_list = []
        pr_recall_list = []

        print(f"Treinando modelo {nome_do_modelo} ", end="")

        cv_outer = KFold(n_splits=k_folds_outer, shuffle=True, random_state=rand_state)

        tempos_de_treinamento = []
        best_model_params = []
        best_trained_models = []

        for train_ix, test_ix in cv_outer.split(X):
            print(".", end="")

            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]

            # Calcular a mediana nos dados de treino
            median_precip_train = X_train['PRECIP_NEXT_QUAD'].median()

            # Definir y_train e y_test com base na mediana calculada
            y_train = (X_train['PRECIP_NEXT_QUAD'] > median_precip_train).astype(int)
            y_test = (X_test['PRECIP_NEXT_QUAD'] > median_precip_train).astype(int)

            # Remover a coluna 'PRECIP_NEXT_QUAD' após definir y
            X_train = X_train.drop(columns=['PRECIP_NEXT_QUAD'])
            X_test = X_test.drop(columns=['PRECIP_NEXT_QUAD'])

            grid_search = GridSearchCV(estimador_base, parametros, 
                                       scoring='f1', 
                                       cv=KFold(n_splits=k_folds_inner, shuffle=True, random_state=17),
                                       n_jobs=4)
            
            tempo_treinamento = time.time()
            modelo_treinado = grid_search.fit(X_train, y_train)
            tempo_treinamento = time.time() - tempo_treinamento

            tempos_de_treinamento.append(tempo_treinamento)

            modelo_treinado = clone(grid_search.best_estimator_)
            modelo_treinado.set_params(**grid_search.best_params_)

            best_model_params.append(grid_search.best_params_)

            modelo_treinado = modelo_treinado.fit(X_train, y_train)
            
            best_trained_models.append(modelo_treinado)

            y_pred = modelo_treinado.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precisions = metrics.precision_score(y_test, y_pred)
            recalls = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)

            accuracy_list.append(accuracy)
            precision_list.append(precisions)
            recall_list.append(recalls)
            f1_score_list.append(f1)
            
            if hasattr(modelo_treinado, "predict_proba"):
                y_pred_proba = modelo_treinado.predict_proba(X_test)[:, 1]
                
                auc_score = metrics.roc_auc_score(y_test, y_pred_proba)
                auc_score_list.append(auc_score)

                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
                roc_fpr_list.append(fpr)
                roc_tpr_list.append(tpr)

                aucpr_score = metrics.average_precision_score(y_test, y_pred_proba)
                aucpr_score_list.append(aucpr_score)

                precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, y_pred_proba)
                pr_precision_list.append(precisions)
                pr_recall_list.append(recalls)
            else:
                print('x', end='')
                auc_score_list.append(0)
                aucpr_score_list.append(0)
                pr_precision_list.append([])
                pr_recall_list.append([])
        
        print("\n-- coletando e armazenando resultados --\n")

        accuracy_mean = np.mean(accuracy_list)
        accuracy_std = np.std(accuracy_list)
        precision_mean = np.mean(precision_list)
        precision_std = np.std(precision_list)
        recall_mean = np.mean(recall_list)
        recall_std = np.std(recall_list)
        f1_score_mean = np.mean(f1_score_list)
        f1_score_std = np.std(f1_score_list)

        auc_mean = np.mean(auc_score_list)
        auc_std = np.std(auc_score_list)
        aucpr_mean = np.mean(aucpr_score_list)
        aucpr_std = np.std(aucpr_score_list)

        print(f" - Acurácia   : {accuracy_mean:.4f} +/- {accuracy_std:.5f}")
        print(f" - Precisão   : {precision_mean:.4f} +/- {precision_std:.5f}")
        print(f" - Revocação  : {recall_mean:.4f} +/- {recall_std:.5f}")
        print(f" - F1 - Score : {f1_score_mean:.4f} +/- {f1_score_std:.5f}")
        print(f" - ROC - AUC  : {auc_mean:.4f} +/- {auc_std:.5f}")
        print(f" - PR - AUC   : {aucpr_mean:.4f} +/- {aucpr_std:.5f}")
        print(f" - Tempo médio de treinamento: {np.mean(tempos_de_treinamento):.2f} segundos\n")
        print('=' * 50, '\n')

        resultados_gerais[nome_do_modelo] = {
            "Acurácia_mean": accuracy_mean,
            "Acurácia_std": accuracy_std,
            "Precisão_mean": precision_mean,
            "Precisão_std": precision_std,
            "Revocação_mean": recall_mean,
            "Revocação_std": recall_std,
            "F1_score_mean": f1_score_mean,
            "F1_score_std": f1_score_std,
            "aucROC_mean": auc_mean,
            "aucROC_std": auc_std,
            "aucPR_mean": aucpr_mean,
            "aucPR_std": aucpr_std,
            "tempo_medio_treinamento": np.mean(tempos_de_treinamento),
            "F1_score_list": f1_score_list,
            "melhores_parametros": best_model_params, 
            "melhores_modelos": best_trained_models 
        }

    print("Terminado em", time.strftime('%d/%m/%Y %H:%M:%S', time.localtime()))

    return resultados_gerais
