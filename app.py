from flask import Flask, jsonify, request, render_template, send_from_directory
import settings
from auto_machine_learning.AutoML.automl import automl_run
from pandas import read_csv
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
    uploaded_file = request.files['dataset']
    dataset_path = './data/{}'.format(uploaded_file.filename or 'dataset.csv')
    dataset = read_csv(dataset_path)
    pickle_path = 'model.pickle'
    # automl_run(dataset, label, task, base_layer_models=[
    #     'LogisticRegression',
    #     'DecisionTreeClassifier',
    #     'RandomForestClassifier', 
    #     'GradientBoostingClassifier',
    #     'ExtraTreesClassifier',
    #     'AdaBoostClassifier'
    #     ], 
    #     download_model = pickle_path,
    #     meta_layer_models = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier']
    # )
    # return send_from_directory(filename=pickle_path+'.sav', directory='.')
    print(pickle_path+'.sav')
    return send_from_directory('.', pickle_path+'.sav', as_attachment=True)
    # return {
    #     'this': 'works'
    # }


if __name__ == '__main__':
    app.run(port=port, debug=True)