import numpy as np
import pandas as pd
from scipy import interp

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, plot, iplot

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

from sklearn import linear_model, decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_hists2(disease, no_disease, features_for_hist):

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    def create_disease_trace(col, visible=False):
        return go.Histogram(
            x=disease[col],
            name='disease',
            marker = dict(color = colors[1]),
            visible=visible,
        )

    def create_no_disease_trace(col, visible=False):
        return go.Histogram(
            x=no_disease[col],
            name='no disease',
            marker = dict(color = colors[0]),
            visible = visible,
        )
    
    active_idx = 0
    traces_disease = [(create_disease_trace(col) if i != active_idx else create_disease_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]
    traces_no_disease = [(create_no_disease_trace(col) if i != active_idx else create_no_disease_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]
    data = traces_disease + traces_no_disease

    n_features = len(features_for_hist)
    steps = []
    for i in range(n_features):
        step = dict(
            method = 'restyle',  
            args = ['visible', [False] * len(data)],
            label = features_for_hist[i],
        )
        step['args'][1][i] = True # Toggle i'th trace to "visible"
        step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active = active_idx,
        currentvalue = dict(
            prefix = "Feature: ", 
            xanchor= 'center',
        ),
        pad = {"t": 50},
        steps = steps,
    )]

    layout = dict(
        sliders=sliders,
        yaxis=dict(
            title='#samples',
            automargin=True,
        ),
    )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='histogram_slider')
    
# Create function for plotting
def plot_hists(df, cols_to_plot):
    
    # Remove columns where there is only one value
    for col in cols_to_plot:
        if col not in df.columns:
            cols_to_plot.remove(col)
    
    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for col in cols_to_plot:
        fig.add_trace(
            go.Histogram(x=df[col], name=col)
        )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=cols_to_plot[i]
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Feature: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        yaxis=dict(
            title='#samples',
            automargin=True,
        )
    )

    fig.show()

def var_imp_plot(model, df):
    
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    
    column_names = df.columns

    imp = list(zip(column_names, feature_importance))
    x = sorted(imp, reverse=True, key= lambda x: x[1])[0:20][::-1]

    sorted_idx = np.argsort(feature_importance)[0:20]
    pos = np.arange(sorted_idx.shape[0])+0.5

    plt.subplot(1, 2, 2)
    plt.barh(pos, [a[1] for a in x], align='center')
    plt.yticks(pos, [a[0] for a in x])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
def print_grid_search(grid_search):
    
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print( "Best estimator:\n{}".format(grid_search.best_estimator_))
    
def print_model_scores(model, y_test, X_test_scaled, X_train_scaled, y_train):
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_predicted_test = model.predict(X_test_scaled)
    y_predicted_train = model.predict(X_train_scaled)
    
    print("Train Accuracy : %.4f " % (model.score(X_train_scaled, y_train)))
    print("Test Accuracy : %.4f " % (model.score(X_test_scaled, y_test)))
    
    print("Confusion matrix Train: ")
    print(confusion_matrix(y_train, y_predicted_train))
  
    print("Confusion matrix Test: ")
    print(confusion_matrix(y_test, y_predicted_test))
    
    unique_classes=[1,2,10,15]

    probabilities_test = model.predict_proba(X_test_scaled)
    probabilities_train = model.predict_proba(X_train_scaled)
    
    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=[1,2,10,15])
    y_train_binarized = label_binarize(y_train, classes=[1,2,10,15])
    n_classes = y_test_binarized.shape[1]   
    
    # Compute ROC curve and ROC area for each class
    fpr_train = dict()
    tpr_train = dict()
    roc_auc_train = dict()
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()
    for i in range(n_classes):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], probabilities_test[:, i])
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])
        
        fpr_train[i], tpr_train[i], _ = roc_curve(y_train_binarized[:, i], probabilities_train[:, i])
        roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])
    
    ## Calculate MultiClass AUC
    # First aggregate all false positive rates
    all_fpr_train = np.unique(np.concatenate([fpr_train[i] for i in range(n_classes)]))
    all_fpr_test = np.unique(np.concatenate([fpr_test[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr_train = np.zeros_like(all_fpr_train)
    for i in range(n_classes):
        mean_tpr_train += interp(all_fpr_train, fpr_train[i], tpr_train[i])
    mean_tpr_test = np.zeros_like(all_fpr_test)
    for i in range(n_classes):
        mean_tpr_test += interp(all_fpr_test, fpr_test[i], tpr_test[i])
    
    # Finally average it and compute AUC
    mean_tpr_train /= n_classes
    mean_tpr_test /= n_classes
    
    fpr_train["macro"] = all_fpr_train
    tpr_train["macro"] = mean_tpr_train
    roc_auc_train["macro"] = auc(fpr_train["macro"], tpr_train["macro"])
    print("AUC Train: {:.4f}".format(roc_auc_train['macro']))
    
    fpr_test["macro"] = all_fpr_test
    tpr_test["macro"] = mean_tpr_test
    roc_auc_test["macro"] = auc(fpr_test["macro"], tpr_test["macro"])
    print("AUC Test: {:.4f}".format(roc_auc_test['macro']))
    print ("\n")
    
    return({'Test_Accuracy': round(model.score(X_test_scaled, y_test), 4),
            'Train_Accuracy': round(model.score(X_train_scaled, y_train), 4),
            'Train AUC': round(roc_auc_train['macro'], 4),
            'Test AUC': round(roc_auc_test['macro'], 4)})
    
def y_by_classes(y_test, y_predicted):
    
    y_test_temp=y_test.reset_index()
    
    test=[]
    predicted=[]
    classes=[1,15,10,2]

    for c in range(len(classes)):
        test.append(list(y_test_temp[y_test_temp.Target == classes[c]].Target))
        predicted.append(list(y_predicted[y_test_temp[y_test_temp.Target == classes[c]].index.values]))
    
    return({'test':test, 'predicted':predicted})
        
        
def plot_model(X_train_scaled, y_train, X_test_scaled, y_test, clf):
    
    y_predicted = clf.predict(X_test_scaled)
    y_train_preds = clf.predict(X_train_scaled)
    
    unique_classes=[1,2,10,15]

    probabilities = clf.predict_proba(X_test_scaled)

    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=[1,2,10,15])
    n_classes = y_test_binarized.shape[1]

    print (clf)
    print ("\n Classification report : \n",classification_report(y_test,y_predicted))
    print("Test Accuracy   Score : {:.4f}".format(accuracy_score(y_test,y_predicted)))
    #confusion matrix
    conf_matrix = confusion_matrix(y_test,y_predicted)    
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarized[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    ## Calculate MultiClass AUC
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("Multi-Class Area Under the Curve: {:.4f}".format(roc_auc['macro']))
    print ("\n")

    print('Classwise Area Under the Curves')
    for i in range(len(unique_classes)):
        if i==0:
            auc_name = 'Normal- '
        else:
            auc_name = 'Disease Type '+str(i) + '- '
        
        temp_auc = round(roc_auc[i], 2)
        print(auc_name + str(temp_auc))
    print('\n')
    
    #plot confusion matrix
    trace1 = go.Heatmap(z = conf_matrix ,
                        x = ["No Disease", 'Disease Class 1', 'Disease Class 2', 'Disease Class 3'],
                        y = ["No Disease", 'Disease Class 1', 'Disease Class 2', 'Disease Class 3'],
                        showscale  = False,colorscale = "Picnic",
                        name = "matrix")

    #subplots
    fig = tls.make_subplots(rows=3, cols=2,
                            specs=[[{}, None], [{}, {}], [{}, {}]],
                                subplot_titles=('Confusion Matrix',
                                                'ROC 1',
                                                'ROC 2',
                                                'ROC 3',
                                                'ROC 4'))

    # fig = tls.make_subplots(rows=3, cols=2)
    fig.append_trace(trace1,1,1)

    for i in range(n_classes):
        trace2_temp = go.Scatter(x = fpr[i],y = tpr[i],
                                 name = "Roc : " + str(roc_auc[i]), 
                                 mode='lines+text',
                                 text=['AUC: '+str(round(roc_auc[i], 2))], 
                                 textposition='top right',
                                 textfont=dict(
                                     family="sans serif",
                                     size=18,
                                     color="DarkSeaGreen"),
                                 line = dict(color = ('rgb(22, 96, 167)'),
                                             width = 2))
        trace3_temp = go.Scatter(x = [0,1],y=[0,1],
                                 line = dict(color = ('rgb(205, 12, 24)'),
                                             width = 2,
                                             dash = 'dot'))
        if i==0:
            fig.append_trace(trace2_temp,2,1)
            fig.append_trace(trace3_temp,2,1)
        elif i==1:
            fig.append_trace(trace2_temp,2,2)
            fig.append_trace(trace3_temp,2,2)
        elif i==2:
            fig.append_trace(trace2_temp,3,1)
            fig.append_trace(trace3_temp,3,1)
        else:
            fig.append_trace(trace2_temp,3,2)
            fig.append_trace(trace3_temp,3,2)

    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    for i in [2,3,4,5]:
        fig["layout"]["xaxis"+str(i)].update(dict(title = "false positive rate"))
        fig["layout"]["yaxis"+str(i)].update(dict(title = "true positive rate"))

    iplot(fig)
    
    
#function  for pie plot for customer attrition types
def plot_pie(column, df):
    
    no_disease = df[df["Target"] == 1]
    disease1 = df[df["Target"] == 15]
    disease2 = df[df["Target"] == 10]
    disease3 = df[df["Target"] == 2]
    
    trace1 = go.Pie(values  = disease1[column].value_counts().values.tolist(),
                    labels  = disease1[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.23]),
                    name    = "Patients with Disease 1",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6)
    trace2 = go.Pie(values  = disease2[column].value_counts().values.tolist(),
                    labels  = disease2[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [.26,.48]),
                    name    = "Patients with Disease 1",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6)
    trace3 = go.Pie(values  = disease3[column].value_counts().values.tolist(),
                    labels  = disease3[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [.51,.73]),
                    name    = "Patients with Disease 1",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6)
    trace4 = go.Pie(values  = no_disease[column].value_counts().values.tolist(),
                    labels  = no_disease[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.76,1]),
                    hole    = .6,
                    name    = "Patients Without Disease" )


    layout = go.Layout(dict(title = column + " distribution by type of Arrhythmia ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "No Arrhythmia",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .06, y = .5),
                                           dict(text = "Type 1",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .365,y = .5),
                                           dict(text = "Type 2",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .62,y = .5),
                                           dict(text = "Type 3",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .91,y = .5)
                                          ]
                           )
                      )
    data = [trace1,trace2,trace3,trace4]
    fig  = go.Figure(data = data,layout = layout)
    iplot(fig)