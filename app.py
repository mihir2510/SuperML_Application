from flask import Flask, jsonify, request, render_template, send_from_directory
import settings
from auto_machine_learning.AutoML.automl import automl_run
from pandas import read_csv
from auto_machine_learning.AutoML.auto_model_trainer import auto_trainer
port = 5000
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(settings)

@app.route('/')
def home():
    # return {
    #     'this': 'works'
    # }
    return render_template('index.html')

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
    stats, _ = automl_run(dataset, label, task,
        base_layer_models = base_layer_models if formdata['settings'][0] == 'custom' else model_list,
        meta_layer_models = meta_models if formdata['settings'][0] == 'custom' else model_list,
        download_model = pickle_path,
        metric=formdata['metric'][0],
        sortby=formdata['metric-sortby'][0],
        excel_file=excel_path
    )
    # print(stats)
    # for i in stats:
    #     print('here',i)
    # print()
    metric_to_show = stats.iloc[0][formdata['metric-sortby'][0]]
    # return send_from_directory(filename=pickle_path+'.sav', directory='.')
    # return send_from_directory('.', pickle_path+'.sav', as_attachment=True)
    return render_template('results.html', excel_path=excel_path+'.xlsx', model_path=pickle_path+'.sav', stats=stats, metric_to_show = metric_to_show, metric = formdata['metric-sortby'][0])
    # return {
    #     'this': 'works'
    # }

@app.route('/result-generator')
def result_gen():
    return render_template('result_gen.html')

@app.route('/process-result-gen', methods=['POST'])
def process_result_gen():
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
    print(sortby)
    stats, model = auto_trainer(dataset,label,task, feature_engineering_methods=feature_engineering_methods, hpo_methods=hpo_methods, models=models, sortby = sortby[0])
    print(stats.head())

    return 'data-submitted'

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(port=port, debug=True)
