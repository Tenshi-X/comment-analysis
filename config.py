import os

class Config:
    SECRET_KEY = 'supersecretkey'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'instance', 'uploads')
    ALLOWED_EXTENSIONS = {'csv'}
