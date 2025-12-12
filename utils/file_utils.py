import os
from werkzeug.utils import secure_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def save_uploaded_file(file, upload_folder):
    # Pastikan upload_folder adalah folder, bukan file
    if not os.path.isdir(upload_folder):
        # Jika path ada dan bukan folder, hapus dulu
        if os.path.exists(upload_folder):
            os.remove(upload_folder)
        os.makedirs(upload_folder, exist_ok=True)
    else:
        os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    return filepath
