from flask import Flask, jsonify, request, render_template, send_from_directory, session
import settings
from auto_machine_learning.automl.automl import automl_run
from pandas import read_csv,read_excel
from auto_machine_learning.automl.auto_model_trainer import auto_trainer
from auto_machine_learning.visualization import plot_2d,plot_3d
port = 5000
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(settings)
import time
import codecs
from uuid import uuid4
app.secret_key = 'my unobvious secret key'

name_holder = {
    'Linear Regression' : 'LiR',
    'Ridge Regression' : "RR",
    'Lasso Regression' : "LaR",
    'Decision Tree Regressor' : 'DTR',
    'Random Forest Regressor' : 'RFR',
    'AdaBoostRegressor' : 'ABR',
    'Extra Trees Regressor' : 'ETR',
    'Bagging Regressor' : 'BG',
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
    return render_template('home.html')

@app.route('/ensemble')
def ensemble():
    return render_template('automl.html')

@app.route('/process', methods=['POST'])
def process():
    print(request.form)
    print(request.files)
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
    regression_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor']
    classification_models = ['LogisticRegression','RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']
    model_list = classification_models if task == 'classification' else regression_models
    base_layer_models = formdata.get('base-layer', [])
    meta_models = formdata.get('meta-layer', [])
    sortby = formdata.get('metric-sortby')
    stats, _ = automl_run(dataset, label, task,
        base_layer_models = base_layer_models if formdata['settings'][0] == 'custom' else model_list,
        meta_layer_models = meta_models if formdata['settings'][0] == 'custom' else model_list,
        download_model = pickle_path,
        metric=formdata['metric'][0],
        sortby=sortby[0],
        excel_file=excel_path
    )
    # print(stats)
    # for i in stats:
    #     print('here',i)
    # print()

    sortby[0]=column_holder[sortby[0]]
    myfile=read_excel(excel_path+'.xlsx')
    
    for ind in myfile.index:
        myfile['Meta Layer Model'][ind]=name_holder[myfile['Meta Layer Model'][ind]]
        x= myfile['Base Layer Models'][ind].split(", ")
        for sp in range(len(x)):
            x[sp]=name_holder[x[sp]]
        myfile['Base Layer Models'][ind]=', '.join(x)


    plot_2d.bar_2d(myfile,Y=sortby[0],X='Meta Layer Model',groups=['Base Layer Models'],file_name='2d.html',height=None,width=None)
    plot_3d.surface_3d(myfile, Z=sortby[0],  X='Meta Layer Model', Y=['Base Layer Models'],file_name='3d.html',height=750,width=None)
    
    


    metric_to_show = stats.iloc[0][formdata['metric-sortby'][0]]
    # return send_from_directory(filename=pickle_path+'.sav', directory='.')
    # return send_from_directory('.', pickle_path+'.sav', as_attachment=True)

    f = codecs.open("./2d.html",'r')
    graph_2d=f.read()

    f = codecs.open("./3d.html",'r')
    graph_3d=f.read()

    print(stats)
    # print(type(stats))
    return render_template('results.html', excel_path=excel_path+'.xlsx', model_path=pickle_path+'.sav', stats=stats, metric_to_show = metric_to_show, metric = formdata['metric-sortby'][0],graph_2d=graph_2d,graph_3d=graph_3d, task=task)

@app.route('/result-generator')
def result_gen():
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
    print(sortby)
    stats, model = auto_trainer(dataset,label,task, feature_engineering_methods=feature_engineering_methods, 
                                    hpo_methods=hpo_methods, 
                                    models=models, 
                                    sortby = sortby[0], 
                                    download_model = pickle_path, 
                                    excel_file=excel_path,
                                    threshold=threshold,
                                    max_evals=max_evals,
                                    test_size=test_size)
    print(stats.head())

    sortby[0]=column_holder[sortby[0]]
    myfile=read_excel(excel_path+'.xlsx')
    for ind in myfile.index:
        myfile['Estimator'][ind]=name_holder[myfile['Estimator'][ind]]
        myfile['Feature Engineering Method'][ind]=name_holder[myfile['Feature Engineering Method'][ind]]
        myfile['Hyperparameter Optimization Method'][ind]=name_holder[myfile['Hyperparameter Optimization Method'][ind]]
    plot_2d.bar_2dsubplot(myfile,Y=sortby[0],plots=['Estimator','Feature Engineering Method','Hyperparameter Optimization Method'],file_name='2d.html',height=1500,width=None)
    f = codecs.open("./2d.html",'r')
    graph_2d=f.read()

    plot_3d.surface_3d(myfile, Z=sortby[0],  X='Estimator', Y=['Feature Engineering Method','Hyperparameter Optimization Method'],file_name='3d.html',height=750,width=None)
    f = codecs.open("./3d.html",'r')
    graph_3d=f.read()
    metric_to_show = stats.iloc[0][sortby[0]]
    return render_template('results.html', excel_path=excel_path+'.xlsx', model_path=pickle_path+'.sav', stats=stats, metric_to_show = metric_to_show, metric = sortby,graph_2d=graph_2d,graph_3d=graph_3d)

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('.', path)


@app.route('/trial')
def trial():
    return render_template('trial1.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)
