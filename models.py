from . import db
from flask_login import UserMixin







class Cheque(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    nom = db.Column(db.String(100))
    montant = db.Column(db.String(100))
    montantnum = db.Column(db.String(100))
    accountowner = db.Column(db.String(100))
    dest = db.Column(db.String(100))
    date = db.Column(db.String(100))
    rib = db.Column(db.String(100))

    user_id = db.Column(db.Integer,db.ForeignKey('user.id')) #foreign Key
   

class User(db.Model, UserMixin):
    id = db.Column(db.Integer,primary_key=True)
    fullname = db.Column(db.String(100))
    email = db.Column(db.String(100),unique =True)
    pwd = db.Column(db.String(150))

    #cheques = db.relationship('Cheque') #so that a user can access all his cheques

