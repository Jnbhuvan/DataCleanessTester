from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_numpy_types(obj):
    """
    Convert numpy data types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def dataset_purity(df):
    """
    Calculate dataset purity metrics including missing values, duplicates, outliers, and overall purity score
    """
    total_cells = df.shape[0] * df.shape[1]

    # 1. Missing values
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / total_cells) * 100

    # 2. Duplicate rows
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100

    # 3. Data type consistency (basic check)
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns
    dtype_consistency = ((len(numeric_cols) + len(cat_cols)) / df.shape[1]) * 100

    # 4. Outlier detection (IQR method for numeric cols)
    outlier_count = 0
    total_numeric = 0
    for col in numeric_cols:
        # Handle columns with all NaN values
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero
                outlier_count += ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            total_numeric += len(df[col])

    outlier_pct = (outlier_count / total_numeric) * 100 if total_numeric > 0 else 0

    # Combine into purity score (weights adjustable)
    purity_score = 100 - (0.4*missing_pct + 0.3*duplicate_pct + 0.2*outlier_pct)

    result = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": total_cells,
        "missing_values": missing_count,
        "missing_percentage": round(missing_pct, 2),
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": round(duplicate_pct, 2),
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_pct, 2),
        "data_type_consistency": round(dtype_consistency, 2),
        "purity_score": max(0, round(purity_score, 2)),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(cat_cols)
    }
    
    # Convert numpy types to native Python types
    return convert_numpy_types(result)

def process_csv(file_path):
    """
    Process the CSV file using the dataset_purity function
    """
    try:
        # Read the CSV file with error handling for mixed types
        df = pd.read_csv(file_path, low_memory=False)
        
        # Get basic file info
        basic_info = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'first_five_rows': df.head().fillna('NULL').to_dict('records'),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Get purity metrics
        purity_metrics = dataset_purity(df)
        
        # Combine results and convert all numpy types
        result = convert_numpy_types({**basic_info, 'purity_metrics': purity_metrics})
        
        return result
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is a CSV
    if file and allowed_file(file.filename):
        try:
            # Secure the filename and save it
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the CSV file
            result = process_csv(file_path)
            
            # Clean up: remove the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up in case of error
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)