from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask import current_app
from os import path
from flask_login import LoginManager
from flask import Blueprint
db = SQLAlchemy()
db_name = "database.db"

chatbot = Blueprint('chatbot', __name__)

def create_app():
    app = Flask(__name__,static_url_path='/static')
    app.secret_key="nada"
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_name}'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] =False
    db.init_app(app)

    

    from .views import views
    from .auth import auth

    app.register_blueprint(views,url_prefix='/')
    app.register_blueprint(auth,url_prefix='/')
    app.register_blueprint(chatbot, url_prefix='/chatbot')


    from .models import User
    
    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view='auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def create_database(app):
    with app.app_context():
        if not path.exists('pfa_flask/' + db_name):
            db.create_all()
            print('Created Database !')



