from flask import Flask
from flask_sqlalchemy import SQLAlchemy

mysql = SQLAlchemy() 
app = Flask(__name__, template_folder="templates")
app.config["SQLALCHEMY_DATABASE_URI"]="mysql://root:password123@localhost/oncological_treatment"

mysql.init_app(app) 

class Patient(mysql.Model):
    __tablename__ = 'patients'
    id = mysql.Column(mysql.Integer, primary_key=True, autoincrement=True)
    patient_id = mysql.Column(mysql.Integer, unique=True)
    name = mysql.Column(mysql.String(50))
    age = mysql.Column(mysql.Integer, nullable=True)
    sex = mysql.Column(mysql.String(10))

    doctors = mysql.relationship('PatientDoctor', back_populates='patient')
    clinical_info = mysql.relationship('ClinicalInfo', back_populates='patient')

class Doctor(mysql.Model):
    __tablename__ = 'doctors'
    id = mysql.Column(mysql.Integer, primary_key=True, autoincrement=True)
    name = mysql.Column(mysql.String(50), nullable=False)
    specialty = mysql.Column(mysql.String(100))
    contact_info = mysql.Column(mysql.String(100))

    patients = mysql.relationship('PatientDoctor', back_populates='doctor')
    clinical_info = mysql.relationship('ClinicalInfo', back_populates='doctor')

class PatientDoctor(mysql.Model):
    __tablename__ = 'patient_doctor'
    id = mysql.Column(mysql.Integer, primary_key=True, autoincrement=True)
    patient_id = mysql.Column(mysql.Integer, mysql.ForeignKey('patients.id'), nullable=False)
    doctor_id = mysql.Column(mysql.Integer, mysql.ForeignKey('doctors.id'), nullable=False)

    patient = mysql.relationship('Patient', back_populates='doctors')
    doctor = mysql.relationship('Doctor', back_populates='patients')

class ClinicalInfo(mysql.Model):
    __tablename__ = 'clinical_info'
    id = mysql.Column(mysql.Integer, primary_key=True, autoincrement=True)
    patient_id = mysql.Column(mysql.Integer, mysql.ForeignKey('patients.id'), nullable=False)
    doctor_id = mysql.Column(mysql.Integer, mysql.ForeignKey('doctors.id'), nullable=False)
    histological_subtype = mysql.Column(mysql.String(50))
    tumour_size = mysql.Column(mysql.Float, nullable=True)
    second_tumour_size = mysql.Column(mysql.Float, nullable=True)
    third_tumour_size = mysql.Column(mysql.Float, nullable=True)
    tumour_grade = mysql.Column(mysql.Integer, nullable=True)
    node_status =  mysql.Column(mysql.Integer, nullable=True)
    metastasis =  mysql.Column(mysql.Integer, nullable=True)
    tnm_tumour = mysql.Column(mysql.String(5), nullable=True)
    clinical_stage = mysql.Column(mysql.String(50))
    er_status = mysql.Column(mysql.String(10))
    pr_status = mysql.Column(mysql.String(10))
    her2_status = mysql.Column(mysql.String(10))
    ki_67 = mysql.Column(mysql.String(10))
    ki_67_categorised = mysql.Column(mysql.String(5))
    date_of_diag = mysql.Column(mysql.Date)
    ct_findings = mysql.Column(mysql.String(500))
    mri_findings = mysql.Column(mysql.String(500))
    pet_findings = mysql.Column(mysql.String(500))
    ultrasound_findings = mysql.Column(mysql.String(500))
    xray_findings = mysql.Column(mysql.String(500))
    mammogram_findings = mysql.Column(mysql.String(500))
    text = mysql.Column(mysql.String(500))
    
    patient = mysql.relationship('Patient', back_populates='clinical_info')
    doctor = mysql.relationship('Doctor', back_populates='clinical_info')
    treatment_plan = mysql.relationship('Treatment', back_populates='clinical_info')
    
class Treatment(mysql.Model):
    __tablename__ = 'treatment_plan'
    id = mysql.Column(mysql.Integer, primary_key=True, autoincrement=True)
    info_id = mysql.Column(mysql.Integer, mysql.ForeignKey('clinical_info.id'), nullable=False)
    final_treatment_plan = mysql.Column(mysql.Text)
    method = mysql.Column(mysql.String(30))
    
    clinical_info = mysql.relationship('ClinicalInfo', back_populates='treatment_plan')
  
with app.app_context():
    mysql.create_all()  