from flask import Flask, redirect, render_template, request
from openai_api import get_openai_response
from semantic_similarity_bert import get_semantic_prediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_sqlalchemy import SQLAlchemy
import torch
import joblib
from database_commands.create_tables import app, mysql, Patient, Doctor, PatientDoctor, ClinicalInfo, Treatment
from datetime import datetime
from scipy.special import expit
import numpy as np

app = Flask(__name__, template_folder="templates")
app.config["SQLALCHEMY_DATABASE_URI"]="mysql://root:password123@localhost/oncological_treatment"

mysql.init_app(app) 

def get_classification_prediction(symptoms, model, tokenizer):
    optimal_thresholds = joblib.load('optimal_thresholds.pkl')
    print(optimal_thresholds)
    label_encoder = joblib.load('label_encoder.pkl')
    inputs = tokenizer(symptoms, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = expit(logits.detach().cpu().numpy())
    predictions = (probabilities > optimal_thresholds).astype(int) 
    print(probabilities) 
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    
def parse_int(int_str):
    return int(int_str) if int_str and int_str.strip().isdigit() else None 

def parse_float(float_str):
    try:
        return float(float_str.strip()) if float_str and float_str.strip() else None
    except ValueError:
        return None

@app.route('/')
def home():

    return render_template("index.html")

@app.route('/form')
def form():
    mode = request.args.get('mode')
    return render_template("form.html", mode=mode)

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    existing_patient = Patient.query.filter_by(patient_id=request.form.get('patient_id')).first()
    if not existing_patient:
        new_patient = Patient(
            patient_id=parse_int(request.form.get('patient_id')),
            name=request.form.get('name'),
            age=parse_int(request.form.get('age')),
            sex=request.form.get('sex')
        )
        mysql.session.add(new_patient)
        mysql.session.flush() 
        patient_id = new_patient.id
    else:
        patient_id = existing_patient.id 

    existing_doctor = Doctor.query.filter_by(name=request.form.get('primary_oncologist')).first()
    if not existing_doctor:
        new_doctor = Doctor(
            name=request.form.get('primary_oncologist'),
            specialty=request.form.get('specialty_of_physician'),
            contact_info=request.form.get('physician_contact_info')
        )
        mysql.session.add(new_doctor)
        mysql.session.flush() 
        doctor_id = new_doctor.id
    else:
        doctor_id = existing_doctor.id
        
    if not PatientDoctor.query.filter_by(patient_id=patient_id, doctor_id=doctor_id).first():
        patient_doctor = PatientDoctor(patient_id=patient_id, doctor_id=doctor_id)
        mysql.session.add(patient_doctor)
        
    if request.form.get("ki_67").endswith('%'):
        ki_67_value = float(request.form.get("ki_67").strip('%')) 
    else:
        ki_67_value = float(request.form.get("ki_67"))
    if ki_67_value > 30:
        ki_67_categorised = 3
    elif ki_67_value < 5:
        ki_67_categorised = 1
    elif 5 <= ki_67_value <= 30:
        ki_67_categorised = 2
    else:
        ki_67_categorised = 0  
        
    tnm_tumour = request.form.get("clinical_stage")[(request.form.get("clinical_stage")).find('T')+1:(request.form.get("clinical_stage")).find('N')]
    text = f"Tumor size: {parse_float(request.form.get('tumour_size'))}, Grade: {parse_int(request.form.get('tumour_grade'))}, \
        Node status: {parse_int(request.form.get('node_status'))}, Metastasis: {parse_int(request.form.get('metastasis'))}, \
        TNM: {tnm_tumour}, ER: {request.form.get('er_status')},  \
        PR: {request.form.get('pr_status')}, HER2: {request.form.get('her2_status')}, \
        Ki-67: {ki_67_categorised}" 
    
    new_clinical_info = ClinicalInfo(
            patient_id=patient_id,
            doctor_id=doctor_id,
            histological_subtype=request.form.get("histological_subtype"),
            tumour_size=parse_float(request.form.get("tumour_size")),
            second_tumour_size=parse_float(request.form.get("second_tumour_size")),
            third_tumour_size=parse_float(request.form.get("third_tumour_size")),
            tumour_grade=parse_int(request.form.get("tumour_grade")),
            node_status=parse_int(request.form.get("node_status")),
            metastasis=parse_int(request.form.get("metastasis")),
            tnm_tumour = parse_int(request.form.get("metastasis")),
            clinical_stage=request.form.get("clinical_stage"),
            er_status=request.form.get("er_status"),
            pr_status=request.form.get("pr_status"),
            her2_status=request.form.get("her2_status"),
            ki_67=request.form.get("ki_67"),
            ki_67_categorised = ki_67_categorised,
            date_of_diag=parse_date(request.form.get("date_of_diag")),
            ct_findings=request.form.get("ct_findings"),
            mri_findings=request.form.get("mri_findings"),
            pet_findings=request.form.get("pet_findings"),
            ultrasound_findings=request.form.get("ultrasound_findings"),
            xray_findings=request.form.get("xray_findings"),
            mammogram_findings=request.form.get("mammogram_findings"),
            text = text
        )
    mysql.session.add(new_clinical_info)
    mysql.session.flush()
    print(request.form.get("mode"))
    if request.form.get("mode") == "add-patient":
        new_treatment = Treatment(info_id=new_clinical_info.id, 
                                  final_treatment_plan=request.form.get("final_treatment_plan").title(),
                                  method="Entry")
        mysql.session.add(new_treatment)
        mysql.session.commit()
        return redirect('/')
    mysql.session.commit()
    symptoms=text
    openai_pred = get_openai_response(symptoms)
    model = AutoModelForSequenceClassification.from_pretrained("models/bert_model/text_classification")
    tokenizer = AutoTokenizer.from_pretrained("models/bert_model/text_classification")

    text_classification_pred = get_classification_prediction(symptoms, model, tokenizer)
    semantic_similarity_pred = get_semantic_prediction(symptoms)
    return render_template('diagnosis.html', openai_pred=openai_pred, text_classification_pred=', '.join(text_classification_pred[0]), semantic_similarity_pred=semantic_similarity_pred, info_id=new_clinical_info.id)

@app.route('/save-treatment', methods=['POST'])
def save_treatment():
    new_treatment = Treatment(info_id=request.form.get("info_id"),
        final_treatment_plan=request.form.get("final_treatment_plan"),
        method=request.form.get("method"))
    mysql.session.add(new_treatment)
    mysql.session.commit()

    return redirect('/')

@app.route('/view-database')
def view_database():
    data = (
        mysql.session.query(
            Patient.__table__.c,
            *list(Doctor.__table__.c)[1:],
            *list(ClinicalInfo.__table__.c)[3:],
            *list(Treatment.__table__.c)[2:]
        )
        .join(PatientDoctor, Patient.id == PatientDoctor.patient_id)  
        .join(Doctor, Doctor.id == PatientDoctor.doctor_id)  
        .join(ClinicalInfo, ClinicalInfo.patient_id == Patient.id)  
        .join(Treatment, Treatment.info_id == ClinicalInfo.id)
        .all()
    )
    print(data)
    columns = (
        (Patient.__table__.columns.keys()) +
        (Doctor.__table__.columns.keys()[1:]) +
        (ClinicalInfo.__table__.columns.keys()[3:]) +
        (Treatment.__table__.columns.keys()[2:])
    )
    print(data)

    return render_template('view-database.html', data=data, columns=columns)

if __name__ =="__main__":
    app.run(debug=True)