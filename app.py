from flask import Flask, jsonify, request, render_template, send_from_directory, session, flash, redirect, url_for, session
import settings
from auto_machine_learning.automl.automl import automl_run
from pandas import read_csv,read_excel
import pandas as pd
from auto_machine_learning.automl.auto_model_trainer import auto_trainer
from auto_machine_learning.visualization import plot_2d,plot_3d
port = 5000
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(settings)
import time
import codecs
from uuid import uuid4
app.secret_key = 'my unobvious secret key'
from time import time

import warnings
warnings.filterwarnings("ignore")

name_holder = {
    'Linear Regression' : 'LiR',
    'Ridge Regression' : "RR",
    'Lasso Regression' : "LaR",
    'Decision Tree Regressor' : 'DTR',
    'Random Forest Regressor' : 'RFR',
    'AdaBoost Regressor' : 'ABR',
    'Extra Trees Regressor' : 'ETR',
    'Bagging Regressor' : 'BR',
    'Gradient Boosting Regressor' : 'GBR',
    'Logistic Regression' : 'LoR',
    'Random Forest Classifier' : 'RFC',
    'AdaBoost Classifier' : 'ABC',
    'Bagging Classifier' : 'BC',
    'Gradient Boosting Classifier' : 'GBC',
    'Extra Trees Classifier' : 'ETC',
    'Decision Tree Classifier' : 'DTC',
    'No HPO':'No HPO',
    'Grid Search':'GS',
    'Random Search':'RS',
    'Bayesian Optimization':'BO',
    'No Feature Engineering' : 'No FE',
    'ANOVA' : 'ANOVA',
    'Correlation Method' : 'Corr',
    'Pricipal Component Analysis' : 'PCA',
    'Select From Model' : 'SFM'
}

column_holder={
    'Meta Layer Model':'Meta Layer Model',
    'Base Layer Models':'Base Layer Models',
    'r2':'R2 Score',
    'rmse':'RMSE',
    'mae':'MAE',
    'accuracy':'Accuracy',
    'precision':'Precsion',
    'precision_micro':'Precsion Micro',
    'precision_macro':'Precison Macro',
    'recall':'Recall',
    'recall_micro':'Recall Micro',
    'recall_macro':'Recall Macro',
    'f1':'F1 Score',
    'f1_micro':'F1 Score Micro',
    'f1_macro':'F1 Score Macro',
    'Estimator':'Estimator',
    'Feature Engineering Method':'Feature Engineering Method',
    'Hyperparameter Optimization Method':'Hyperparameter Optimization Method'
}

@app.route('/')
def home():
    session['uid']=str(uuid4())
    return render_template('index.html')

@app.route('/ensemble')
def ensemble():
    if 'message-ensemble' in session:
        message=session['message-ensemble']
        session.pop('message-ensemble',None)
        return render_template('automl.html',message = message)
    return render_template('automl.html')

@app.route('/process', methods=['POST'])
def process():
    print(request.form)
    print(request.files)
    label = request.form['label']
    task = request.form['task']
    formdata = dict(list(request.form.lists()))
    uploaded_file = request.files['dataset']
    dataset_path = './data/{}'.format(uploaded_file.filename or 'dataset.csv')
    uploaded_file.save(dataset_path)
    dataset = read_csv(dataset_path)
    pickle_path = './models/model.pickle'
    excel_path = './excel_files/excel'
    regression_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor']
    classification_models = ['LogisticRegression','RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']
    model_list = classification_models if task == 'classification' else regression_models
    base_layer_models = formdata.get('base-layer', [])
    meta_models = formdata.get('meta-layer', [])
    sortby = formdata.get('metric-sortby')
    start = time()
    try:
        _, stats, list_of_new_features = automl_run(dataset, label, task,
            base_layer_models = base_layer_models if formdata['settings'][0] == 'custom' else model_list,
            meta_layer_models = meta_models if formdata['settings'][0] == 'custom' else model_list,
            download_model = pickle_path,
            metric=formdata['metric'][0],
            sortby=sortby[0],
            excel_file=excel_path
        )
    except:
        session['message-ensemble'] = 'Please check the label given and make sure the csv file is valid!'
        return redirect(url_for('.ensemble'))
        #return render_template('automl.html', message='Please check the label given and make sure the csv file is valid!')
    time_taken = time()-start
    unit = 'seconds'
    if time_taken>120:
        time_taken = time_taken/60
        unit = 'minutes'

    time_taken = round(time_taken,4)
    time_taken = "{} ".format(time_taken)+unit
    sortby[0]=column_holder[sortby[0]]
    myfile=read_excel(excel_path+'.xlsx')
    
    for ind in myfile.index:
        myfile['Meta Layer Model'][ind]=name_holder[myfile['Meta Layer Model'][ind]]
        x= myfile['Base Layer Models'][ind].split(", ")
        for sp in range(len(x)):
            x[sp]=name_holder[x[sp]]
        myfile['Base Layer Models'][ind]=', '.join(x)

    #print(sortby[0])
    plot_2d.bar_2d(myfile,Y=sortby[0],X='Meta Layer Model',groups=['Base Layer Models'],file_name='2dplot',download_png='2dplot',height=500,width=None)
    f = codecs.open("./2dplot.html",'r')
    graph_2d=f.read()
    graph_3d=None
    if len(list(pd.unique(stats['Base Layer Models'])))==1 or len(list(pd.unique(stats['Meta Layer Model'])))==1:
        graph_3d=None
    else:
        plot_3d.surface_3d(myfile, Z=sortby[0],  X='Meta Layer Model', Y=['Base Layer Models'],file_name='3d',height=750,width=None)
        f = codecs.open("./3d.html",'r')
        graph_3d=f.read()
    
    #print(sortby[0])

    metric_to_show = stats.iloc[0][formdata['metric-sortby'][0]]
    # return send_from_directory(filename=pickle_path+'.sav', directory='.')
    # return send_from_directory('.', pickle_path+'.sav', as_attachment=True)





    print(stats)
    print(list_of_new_features)
    # print(type(stats))
    return render_template('results.html', excel_path=excel_path+'.xlsx', model_path=pickle_path+'.sav', stats=stats, metric_to_show = metric_to_show, metric = formdata['metric-sortby'][0],graph_2d=graph_2d,graph_3d=graph_3d, task=task, time_taken = time_taken,list_of_new_features=list_of_new_features)

@app.route('/result-generator')
def result_gen():
    if 'message-autotrainer' in session:
        message=session['message-autotrainer']
        session.pop('message-autotrainer',None)
        return render_template('result_gen.html', message=message)
    return render_template('result_gen.html')

@app.route('/process-result-gen', methods=['POST'])
def process_result_gen():
    print(request.form)
    label = request.form['label']
    task = request.form['task']
    formdata = dict(list(request.form.lists()))
    # return formdata
    uploaded_file = request.files['dataset']
    dataset_path = './data/{}'.format(uploaded_file.filename or 'dataset.csv')
    uploaded_file.save(dataset_path)
    dataset = read_csv(dataset_path)
    pickle_path = './models/model.pickle'
    excel_path = './excel_files/excel'
    print(formdata)
    models = formdata.get('models', None)
    feature_engineering_methods = formdata.get('fe_method')
    hpo_methods = formdata.get('hpo_method')
    models = formdata.get('models')
    sortby = formdata.get('metric')
    threshold = float(request.form['threshold'])
    max_evals = int(request.form['max_evals'])
    test_size = float(request.form['test_size'])
    #print(sortby)
    start = time()
    try:
        model,stats,list_of_new_features = auto_trainer(dataset,label,task, feature_engineering_methods=feature_engineering_methods, 
                                        hpo_methods=hpo_methods, 
                                        models=models, 
                                        sortby = sortby[0], 
                                        download_model = pickle_path, 
                                        excel_file=excel_path,
                                        threshold=threshold,
                                        max_evals=max_evals,
                                        test_size=test_size)
    except:
        session['message-autotrainer'] = 'Please check the label given and make sure the csv file is valid!'
        return redirect(url_for('.result_gen'))
    #print(stats.head())
    time_taken = time()-start
    unit = 'seconds'
    if time_taken>120:
        time_taken = time_taken/60
        unit = 'minutes'

    time_taken = round(time_taken,4)
    time_taken = "{} ".format(time_taken)+unit
    sortby[0]=column_holder[sortby[0]]
    myfile=read_excel(excel_path+'.xlsx')
    for ind in myfile.index:
        myfile['Estimator'][ind]=name_holder[myfile['Estimator'][ind]]
        myfile['Feature Engineering Method'][ind]=name_holder[myfile['Feature Engineering Method'][ind]]
        myfile['Hyperparameter Optimization Method'][ind]=name_holder[myfile['Hyperparameter Optimization Method'][ind]]
    plot_2d.bar_2dsubplot(myfile,Y=sortby[0],plots=['Estimator','Feature Engineering Method','Hyperparameter Optimization Method'],file_name='2dsubplot',download_png='2dsubplot',height=1500,width=None)
    f = codecs.open("./2dsubplot.html",'r')
    graph_2d=f.read()
    graph_3d=None

    if  len(list(pd.unique(stats['Estimator'])))==1 or (len(list(pd.unique(stats['Feature Engineering Method'])))==1 and len(list(pd.unique(stats['Hyperparameter Optimization Method'])))==1):
        graph_3d=None
        #print(len(list(pd.unique(stats['Estimator']))),len(list(pd.unique(stats['Feature Engineering Method']))), len(list(pd.unique(stats['Hyperparameter Optimization Method']))) )
    else:
        myfile.rename(columns={'Feature Engineering Method': 'FE','Hyperparameter Optimization Method':'HPO'}, inplace=True)
        plot_3d.surface_3d(myfile, Z=sortby[0],  X='Estimator', Y=['FE','HPO'],file_name='3d',height=750,width=None)
        f = codecs.open("./3d.html",'r')
        graph_3d=f.read()

    metric_to_show = stats.iloc[0][sortby[0]]
    print(list_of_new_features)
    stats.drop('Selected Features',axis='columns',inplace=True)
    return render_template('results.html', excel_path=excel_path+'.xlsx', model_path=pickle_path+'.sav', stats=stats, metric_to_show = metric_to_show, metric = sortby,graph_2d=graph_2d,graph_3d=graph_3d,task=task, time_taken=time_taken,list_of_new_features=list_of_new_features)

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('.', path)

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/documentation')
def documentation():
    
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)
