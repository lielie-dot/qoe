from flask import Flask
from flask import render_template
from flask import request
from learning import *
data="./data/Challenge_ech2_Enock.csv"

app = Flask(__name__)


@app.route('/',methods= ['POST','GET'])
def main():

    return render_template('indexbis.html')
     

@app.route('/Résultats',methods= ['POST','GET'])
def results():
   if request.method == 'POST':
        coupure= request.form.get('coupure')
        paging_without_response_nb= request.form.get('paging_without_response_nb')
        reiter_call_interval_nb= request.form.get('reiter_call_interval_nb')
        call_est_bad_setupduration_nb=request.form.get('call_est_bad_setupduration_nb')
        
        classe,valeur=unsupervised_supervised_training_given_prediction(data,[int(coupure)],[int(paging_without_response_nb)],[int(reiter_call_interval_nb)],[int(call_est_bad_setupduration_nb)])
        
        if classe==0:
            phrase="Le QoE est mauvais"

        else:
            phrase="Le QoE est bon"
        
        
        return render_template('prediction.html',interpretation=phrase,score="Le score est de : "+str(valeur)+" %")
   else:
        return render_template('prediction.html',score=" Vous devez renseiger l'intégralité des champs")
 



@app.route('/Visualisation',methods= ['POST','GET'])
def visualize():

    if request.method == 'POST':
         return render_template('indexbis.html')   
    
    else:
        
        visualize_training_steps(data)
        return render_template('visualisation.html')
    
   

@app.route('/Tests',methods= ['POST','GET'])
def test():
    if request.method=='POST':
        if request.form.get('Données'):
            Données=request.form.get('Données')
            data_test="./data/"+Données
            rmse_found=unsupervised_supervised_training_given_test(data,data_test)
        return render_template('test.html',rmse="RMSE : "+str(rmse_found))
    

    return render_template('test.html',rmse="RMSE : ")
    
  