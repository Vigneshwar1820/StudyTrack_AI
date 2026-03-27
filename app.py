from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
import os
import sqlite3
from functools import wraps
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'studytrack_secret_key_2024_enhanced_admin'

# Custom Jinja2 filter for JSON parsing - MUST BE BEFORE ANY ROUTES
@app.template_filter('from_json')
def from_json_filter(value):
    """Custom Jinja2 filter to parse JSON strings"""
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except (ValueError, TypeError, json.JSONDecodeError):
        return []

# Explicitly add the filter to Jinja environment
app.jinja_env.filters['from_json'] = from_json_filter

# Load dataset
try:
    DATASET = pd.read_csv("StudentPerformanceFactors (1).csv")
    print(f"✓ Dataset loaded: {len(DATASET)} students")
    
    if 'Student_ID' not in DATASET.columns:
        DATASET['Student_ID'] = ['STU' + str(i).zfill(4) for i in range(1, len(DATASET) + 1)]
    
    print(f"✓ Student IDs: {DATASET['Student_ID'].min()} to {DATASET['Student_ID'].max()}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    DATASET = None

# Database setup
def init_db():
    conn = sqlite3.connect('studytrack.db')
    c = conn.cursor()
    
    # Users table with role and assigned_student_id
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT DEFAULT 'student',
                  assigned_student_id TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Analysis history table
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  student_id TEXT,
                  student_name TEXT,
                  hours_studied REAL,
                  attendance REAL,
                  sleep_hours REAL,
                  previous_scores REAL,
                  tutoring_sessions INTEGER,
                  physical_activity REAL,
                  cluster INTEGER,
                  cluster_label TEXT,
                  predicted_score REAL,
                  analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  admin_notified INTEGER DEFAULT 0,
                  admin_viewed INTEGER DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  analysis_id INTEGER,
                  user_id INTEGER,
                  student_id TEXT,
                  recommendations TEXT,
                  status TEXT DEFAULT 'pending',
                  admin_response TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  responded_at TIMESTAMP,
                  FOREIGN KEY (analysis_id) REFERENCES analysis_history (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Goals table
    c.execute('''CREATE TABLE IF NOT EXISTS goals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  goal_type TEXT,
                  target_value REAL,
                  current_value REAL,
                  deadline DATE,
                  status TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Chat history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  message TEXT,
                  response TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create default admin if doesn't exist
    try:
        admin_password = generate_password_hash('admin123')
        c.execute('INSERT OR IGNORE INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
                 ('admin', 'admin@studytrack.com', admin_password, 'admin'))
    except:
        pass
    
    conn.commit()
    conn.close()

init_db()

# Load models
try:
    kmeans = joblib.load("studytrack_kmeans_model.pkl")
    scaler = joblib.load("studytrack_scaler.pkl")
    print("✓ Models loaded successfully")
except:
    print("✗ Warning: Model files not found")
    kmeans = None
    scaler = None

# Train score predictor if not exists
try:
    from sklearn.linear_model import LinearRegression
    score_predictor = joblib.load("score_predictor.pkl")
    print("✓ Score predictor loaded")
except:
    print("⚠ Training score prediction model...")
    try:
        if DATASET is not None:
            df = DATASET.copy()
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df.columns and col not in ['Exam_Score', 'Student_ID']:
                    df[col] = le.fit_transform(df[col].astype(str))
            
            X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                    'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']]
            y = df['Exam_Score']
            
            score_predictor = LinearRegression()
            score_predictor.fit(X, y)
            joblib.dump(score_predictor, "score_predictor.pkl")
            print("✓ Score predictor trained and saved")
    except Exception as e:
        print(f"✗ Could not train score predictor: {e}")
        score_predictor = None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def get_db():
    conn = sqlite3.connect('studytrack.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_student_data(student_id):
    """Get student data from dataset by Student ID"""
    if DATASET is None:
        return None
    
    student = DATASET[DATASET['Student_ID'] == student_id]
    if student.empty:
        return None
    
    return student.iloc[0].to_dict()

def label_cluster(cluster_id):
    """Label clusters based on performance"""
    labels = {
        0: "Needs Improvement",
        1: "Average Performer", 
        2: "High Performer"
    }
    return labels.get(cluster_id, "Unknown")

def recommend_study_habits(cluster_label, student_data):
    """Generate personalized recommendations"""
    hours = student_data.get('hours_studied', 0)
    attendance = student_data.get('attendance', 0)
    sleep = student_data.get('sleep_hours', 0)
    
    recommendations = {
        "Needs Improvement": {
            "title": "Needs Improvement",
            "color": "#ef4444",
            "icon": "⚠️",
            "description": "Don't worry! With dedication and the right strategies, you can significantly improve.",
            "recommendations": [
                f"📚 Increase study hours from {hours} to at least 5-6 hours daily",
                f"📅 Improve attendance from {attendance}% to 90%+ for better learning",
                "⏰ Follow a fixed study schedule and stick to it consistently",
                "📖 Revise basics regularly and clear doubts immediately",
                f"😴 Get adequate sleep - aim for 7-8 hours ({max(8-sleep, 0):.0f} more hours needed)",
                "👥 Join study groups for peer learning and motivation",
                "🎯 Break study sessions into focused 25-minute intervals (Pomodoro)"
            ]
        },
        "Average Performer": {
            "title": "Average Performer",
            "color": "#f59e0b",
            "icon": "📈",
            "description": "You're doing well! A few strategic improvements can help you reach the next level.",
            "recommendations": [
                f"✅ Maintain your current {hours} hours of daily study",
                "📝 Add weekly revision sessions for better retention",
                "🔄 Practice mock tests and previous year papers regularly",
                "⏱️ Improve time management skills for efficiency",
                "🎯 Focus on strengthening weak subjects systematically",
                "👨‍🏫 Teach concepts to others to deepen understanding",
                "🏆 Set specific, measurable goals for continuous improvement"
            ]
        },
        "High Performer": {
            "title": "High Performer",
            "color": "#10b981",
            "icon": "🌟",
            "description": "Excellent work! Keep up the great habits and continue challenging yourself.",
            "recommendations": [
                f"🚀 Maintain your excellent {hours} hours of focused study",
                "🎓 Focus on advanced topics and competitive exam preparation",
                "👥 Help peers through mentoring or leading study groups",
                "⚡ Work on performance optimization techniques",
                "📚 Explore additional resources and research papers",
                f"💪 Keep up your {attendance}% attendance and {sleep} hours sleep routine",
                "🏅 Participate in academic competitions and olympiads",
                "🎯 Set ambitious goals for continuous growth"
            ]
        }
    }
    return recommendations.get(cluster_label, recommendations["Average Performer"])

def generate_coach_response(message):
    """Simple rule-based AI coach"""
    message = message.lower()
    
    responses = {
        'study': "Great question! I recommend studying in 25-minute focused sessions (Pomodoro technique) with 5-minute breaks. What subject are you working on?",
        'motivation': "Remember why you started! Every small step counts. Break your goals into tiny achievements and celebrate each one. You've got this! 💪",
        'time': "Time management is key! Try the Eisenhower Matrix: prioritize tasks by urgency and importance. Focus on important tasks first.",
        'exam': "Exam preparation tips: 1) Start early, 2) Practice past papers, 3) Create summary notes, 4) Get enough sleep, 5) Stay positive!",
        'score': "To improve scores: Review mistakes, strengthen weak areas, practice regularly, seek help when stuck, and maintain consistency.",
        'sleep': "Sleep is crucial for learning! Aim for 7-8 hours. Your brain consolidates information during sleep. Quality rest = better performance.",
        'help': "I'm here to help! Ask me about: study techniques, motivation, time management, exam prep, improving scores, or any academic concerns.",
    }
    
    for keyword, response in responses.items():
        if keyword in message:
            return response
    
    return "That's a great question! Can you tell me more about what specific area you need help with? (study techniques, motivation, time management, etc.)"
@app.route('/admin/recommendation/<int:rec_id>/delete', methods=['POST'])
@admin_required
def delete_recommendation(rec_id):
    conn = get_db()
    rec = conn.execute('SELECT * FROM recommendations WHERE id = ?', (rec_id,)).fetchone()
    
    if rec:
        conn.execute('DELETE FROM recommendations WHERE id = ?', (rec_id,))
        conn.commit()
        flash('Recommendation deleted successfully!', 'success')
    else:
        flash('Recommendation not found.', 'error')
    
    conn.close()
    return redirect(request.referrer or url_for('admin_recommendations'))

@app.route('/admin/analysis/<int:analysis_id>/delete', methods=['POST'])
@admin_required
def admin_delete_analysis(analysis_id):
    conn = get_db()
    analysis = conn.execute('SELECT * FROM analysis_history WHERE id = ?', (analysis_id,)).fetchone()
    
    if analysis:
        conn.execute('DELETE FROM recommendations WHERE analysis_id = ?', (analysis_id,))
        conn.execute('DELETE FROM analysis_history WHERE id = ?', (analysis_id,))
        conn.commit()
        flash('Analysis deleted successfully!', 'success')
    else:
        flash('Analysis not found.', 'error')
    
    conn.close()
    return redirect(request.referrer or url_for('admin_analytics'))

@app.route('/admin/student/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_student(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ? AND role = "student"', (user_id,)).fetchone()
    
    if user:
        # Delete all related data
        conn.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
        conn.execute('DELETE FROM goals WHERE user_id = ?', (user_id,))
        
        analyses = conn.execute('SELECT id FROM analysis_history WHERE user_id = ?', (user_id,)).fetchall()
        analysis_ids = [a['id'] for a in analyses]
        
        if analysis_ids:
            placeholders = ','.join('?' * len(analysis_ids))
            conn.execute(f'DELETE FROM recommendations WHERE analysis_id IN ({placeholders})', analysis_ids)
        
        conn.execute('DELETE FROM recommendations WHERE user_id = ?', (user_id,))
        conn.execute('DELETE FROM analysis_history WHERE user_id = ?', (user_id,))
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        conn.commit()
        flash(f'Student "{user["username"]}" and all related data deleted!', 'success')
    else:
        flash('Student not found.', 'error')
    
    conn.close()
    return redirect(url_for('admin_dashboard'))
# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        student_id = request.form.get('student_id', '').strip().upper()
        
        # Validate student ID exists in dataset
        if student_id and DATASET is not None:
            if student_id not in DATASET['Student_ID'].values:
                flash('Invalid Student ID. Please select from the available list.', 'error')
                return render_template('register.html', student_ids=sorted(DATASET['Student_ID'].tolist()) if DATASET is not None else [])
        
        conn = get_db()
        try:
            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, email, password, role, assigned_student_id) VALUES (?, ?, ?, ?, ?)',
                        (username, email, hashed_password, 'student', student_id if student_id else None))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
        finally:
            conn.close()
    
    student_ids = sorted(DATASET['Student_ID'].tolist()) if DATASET is not None else []
    return render_template('register.html', student_ids=student_ids)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['assigned_student_id'] = user['assigned_student_id']
            flash(f'Welcome back, {username}!', 'success')
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect admins to their own dashboard
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db()
    
    # Get analysis history - only for this user's assigned student ID
    history = conn.execute('''SELECT * FROM analysis_history 
                             WHERE user_id = ? 
                             ORDER BY analysis_date DESC LIMIT 10''',
                          (session['user_id'],)).fetchall()
    
    # Get goals
    goals = conn.execute('''SELECT * FROM goals 
                           WHERE user_id = ? 
                           ORDER BY created_at DESC''',
                        (session['user_id'],)).fetchall()
    
    # Get pending recommendations count
    pending_recs = conn.execute('''SELECT COUNT(*) as count FROM recommendations 
                                  WHERE user_id = ? AND status = 'pending' ''',
                               (session['user_id'],)).fetchone()
    
    conn.close()
    
    return render_template('dashboard.html', history=history, goals=goals, 
                         pending_count=pending_recs['count'] if pending_recs else 0)

# ADMIN ROUTES
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db()
    
    # Get all pending recommendations
    pending = conn.execute('''SELECT r.*, u.username, a.predicted_score, a.cluster_label
                             FROM recommendations r
                             JOIN users u ON r.user_id = u.id
                             JOIN analysis_history a ON r.analysis_id = a.id
                             WHERE r.status = 'pending'
                             ORDER BY r.created_at DESC''').fetchall()
    
    # Get all users (students)
    students = conn.execute('''SELECT u.*, 
                              COUNT(DISTINCT a.id) as analysis_count,
                              AVG(a.predicted_score) as avg_score
                              FROM users u
                              LEFT JOIN analysis_history a ON u.id = a.user_id
                              WHERE u.role = 'student'
                              GROUP BY u.id
                              ORDER BY u.created_at DESC''').fetchall()
    
    # Get statistics
    stats = {
        'total_students': conn.execute('SELECT COUNT(*) as count FROM users WHERE role="student"').fetchone()['count'],
        'total_analyses': conn.execute('SELECT COUNT(*) as count FROM analysis_history').fetchone()['count'],
        'pending_recommendations': len(pending),
        'avg_predicted_score': conn.execute('SELECT AVG(predicted_score) as avg FROM analysis_history').fetchone()['avg'] or 0
    }
    
    conn.close()
    
    return render_template('admin_dashboard.html', pending=pending, students=students, stats=stats)

@app.route('/admin/send_review/<int:rec_id>', methods=['POST'])
@admin_required
def admin_send_review(rec_id):
    """Allow admin to send review message for any analysis"""
    conn = get_db()
    
    admin_response = request.form.get('admin_response')
    status = request.form.get('status', 'reviewed')
    
    if not admin_response:
        flash('Please enter a review message.', 'error')
        return redirect(url_for('admin_analytics'))
    
    conn.execute('''UPDATE recommendations 
                   SET status = ?, admin_response = ?, responded_at = ?
                   WHERE id = ?''',
                (status, admin_response, datetime.now(), rec_id))
    conn.commit()
    conn.close()
    
    flash('Review message sent successfully!', 'success')
    return redirect(url_for('admin_analytics'))

@app.route('/admin/recommendations')
@admin_required
def admin_recommendations():
    conn = get_db()
    
    # Get all recommendations with details
    recommendations = conn.execute('''SELECT r.*, u.username, u.email, 
                                     a.student_id, a.predicted_score, a.cluster_label,
                                     a.hours_studied, a.attendance, a.sleep_hours
                                     FROM recommendations r
                                     JOIN users u ON r.user_id = u.id
                                     JOIN analysis_history a ON r.analysis_id = a.id
                                     ORDER BY r.created_at DESC''').fetchall()
    
    conn.close()
    
    return render_template('admin_recommendations.html', recommendations=recommendations)

@app.route('/admin/recommendation/<int:rec_id>', methods=['GET', 'POST'])
@admin_required
def admin_view_recommendation(rec_id):
    conn = get_db()
    
    if request.method == 'POST':
        admin_response = request.form.get('admin_response')
        status = request.form.get('status', 'reviewed')
        
        conn.execute('''UPDATE recommendations 
                       SET status = ?, admin_response = ?, responded_at = ?
                       WHERE id = ?''',
                    (status, admin_response, datetime.now(), rec_id))
        conn.commit()
        flash('Response sent successfully!', 'success')
        return redirect(url_for('admin_recommendations'))
    
    # Get recommendation details
    rec = conn.execute('''SELECT r.*, u.username, u.email,
                         a.* 
                         FROM recommendations r
                         JOIN users u ON r.user_id = u.id
                         JOIN analysis_history a ON r.analysis_id = a.id
                         WHERE r.id = ?''', (rec_id,)).fetchone()
    
    conn.close()
    
    if not rec:
        flash('Recommendation not found.', 'error')
        return redirect(url_for('admin_recommendations'))
    
    return render_template('admin_view_recommendation.html', rec=rec)

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    """Admin analytics showing ALL analyses with their review status"""
    conn = get_db()
    
    # Get all analyses with their recommendation status and user info
    history = conn.execute('''SELECT ah.*, u.username, u.role, u.email,
                             r.status as rec_status, r.id as rec_id,
                             r.admin_response
                             FROM analysis_history ah
                             JOIN users u ON ah.user_id = u.id
                             LEFT JOIN recommendations r ON ah.id = r.analysis_id
                             ORDER BY ah.analysis_date DESC''').fetchall()
    
    conn.close()
    
    return render_template('admin_analytics.html', history=history)

@app.route('/admin/student/<int:user_id>')
@admin_required
def admin_view_student(user_id):
    conn = get_db()
    
    # Get student details
    student = conn.execute('SELECT * FROM users WHERE id = ? AND role = "student"', 
                          (user_id,)).fetchone()
    
    if not student:
        flash('Student not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    # Get student's analysis history
    analyses = conn.execute('''SELECT * FROM analysis_history 
                              WHERE user_id = ? 
                              ORDER BY analysis_date DESC''',
                           (user_id,)).fetchall()
    
    # Get student's recommendations
    recommendations = conn.execute('''SELECT * FROM recommendations 
                                     WHERE user_id = ? 
                                     ORDER BY created_at DESC''',
                                  (user_id,)).fetchall()
    
    conn.close()
    
    return render_template('admin_student_profile.html', 
                         student=student, analyses=analyses, recommendations=recommendations)

# STUDENT ROUTES
@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    conn = get_db()
    
    analysis = conn.execute('SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
                          (analysis_id, session['user_id'])).fetchone()
    
    if analysis:
        conn.execute('DELETE FROM analysis_history WHERE id = ?', (analysis_id,))
        conn.execute('DELETE FROM recommendations WHERE analysis_id = ?', (analysis_id,))
        conn.commit()
        flash('Analysis deleted successfully!', 'success')
    else:
        flash('Analysis not found or unauthorized.', 'error')
    
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/input')
@login_required
def input_form():
    is_admin = session.get('role') == 'admin'
    
    if is_admin:
        # Admin can see all student IDs
        student_ids = sorted(DATASET['Student_ID'].tolist()) if DATASET is not None else []
    else:
        # Regular user can only see their assigned student ID
        assigned_id = session.get('assigned_student_id')
        student_ids = [assigned_id] if assigned_id else []
    
    return render_template('input_student_id.html', student_ids=student_ids, is_admin=is_admin)

@app.route('/get_student_data/<student_id>')
@login_required
def get_student_data_api(student_id):
    # Check authorization
    is_admin = session.get('role') == 'admin'
    assigned_id = session.get('assigned_student_id')
    
    if not is_admin and student_id != assigned_id:
        return jsonify({'success': False, 'error': 'Unauthorized access to this student ID'})
    
    student_data = get_student_data(student_id)
    
    if student_data is None:
        return jsonify({'success': False, 'error': 'Student ID not found'})
    
    return jsonify({
        'success': True,
        'data': {
            'student_id': student_id,
            'hours_studied': float(student_data.get('Hours_Studied', 0)),
            'attendance': float(student_data.get('Attendance', 0)),
            'sleep_hours': float(student_data.get('Sleep_Hours', 0)),
            'previous_scores': float(student_data.get('Previous_Scores', 0)),
            'tutoring_sessions': int(student_data.get('Tutoring_Sessions', 0)),
            'physical_activity': float(student_data.get('Physical_Activity', 0))
        }
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        student_id = request.form.get('student_id', '').strip().upper()
        
        if not student_id:
            flash('Please enter a Student ID', 'error')
            return redirect(url_for('input_form'))
        
        # Check authorization for regular users
        is_admin = session.get('role') == 'admin'
        assigned_id = session.get('assigned_student_id')
        
        if not is_admin and student_id != assigned_id:
            flash('You can only analyze your assigned Student ID.', 'error')
            return redirect(url_for('input_form'))
        
        student_data = get_student_data(student_id)
        
        if student_data is None:
            flash(f'Student ID "{student_id}" not found in database.', 'error')
            return redirect(url_for('input_form'))
        
        # Extract features
        hours_studied = float(student_data.get('Hours_Studied', 0))
        attendance = float(student_data.get('Attendance', 0))
        sleep_hours = float(student_data.get('Sleep_Hours', 0))
        previous_scores = float(student_data.get('Previous_Scores', 0))
        tutoring_sessions = int(student_data.get('Tutoring_Sessions', 0))
        physical_activity = float(student_data.get('Physical_Activity', 0))
        
        input_features = np.array([[
            hours_studied, attendance, sleep_hours,
            previous_scores, tutoring_sessions, physical_activity
        ]])
        
        # Predict cluster
        if kmeans and scaler:
            scaled_features = scaler.transform(input_features)
            cluster = int(kmeans.predict(scaled_features)[0])
            cluster_label = label_cluster(cluster)
        else:
            if previous_scores >= 80:
                cluster, cluster_label = 2, "High Performer"
            elif previous_scores >= 60:
                cluster, cluster_label = 1, "Average Performer"
            else:
                cluster, cluster_label = 0, "Needs Improvement"
        
        # Predict score
        if score_predictor:
            predicted_score = float(score_predictor.predict(input_features)[0])
            predicted_score = min(100, max(0, predicted_score))
        else:
            predicted_score = previous_scores
        
        # Generate recommendations
        student_info = {
            'hours_studied': hours_studied,
            'attendance': attendance,
            'sleep_hours': sleep_hours
        }
        recommendations = recommend_study_habits(cluster_label, student_info)
        
        # Save to database
        conn = get_db()
        cursor = conn.execute('''INSERT INTO analysis_history 
                       (user_id, student_id, student_name, hours_studied, attendance, sleep_hours,
                        previous_scores, tutoring_sessions, physical_activity, 
                        cluster, cluster_label, predicted_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session['user_id'], student_id, f"Student {student_id}", hours_studied, attendance,
                     sleep_hours, previous_scores, tutoring_sessions, physical_activity,
                     cluster, cluster_label, predicted_score))
        analysis_id = cursor.lastrowid
        
        # Create recommendation entry for BOTH admin and students
        rec_text = json.dumps(recommendations['recommendations'])
        
        if is_admin:
            # Admin analysis - mark as 'reviewed' (admin-performed)
            conn.execute('''INSERT INTO recommendations 
                           (analysis_id, user_id, student_id, recommendations, status, admin_response, responded_at)
                           VALUES (?, ?, ?, ?, 'reviewed', 'Analysis performed by admin', ?)''',
                        (analysis_id, session['user_id'], student_id, rec_text, datetime.now()))
        else:
            # Student analysis - mark as 'pending' for admin review
            conn.execute('''INSERT INTO recommendations 
                           (analysis_id, user_id, student_id, recommendations, status)
                           VALUES (?, ?, ?, ?, 'pending')''',
                        (analysis_id, session['user_id'], student_id, rec_text))
        
        conn.commit()
        conn.close()
        
        session['current_analysis_id'] = analysis_id
        
        if is_admin:
            flash('Analysis complete! Added to analytics.', 'success')
        else:
            flash('Analysis complete! Recommendations sent to admin for review.', 'success')
        
        return redirect(url_for('feedback'))
        
    except Exception as e:
        flash(f'Error processing student data: {str(e)}', 'error')
        return redirect(url_for('input_form'))

@app.route('/feedback')
@login_required
def feedback():
    if 'current_analysis_id' not in session:
        flash('No analysis found. Please analyze a student first.', 'warning')
        return redirect(url_for('input_form'))

    conn = get_db()

    # 🔹 Fetch analysis
    analysis = conn.execute(
        'SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
        (session['current_analysis_id'], session['user_id'])
    ).fetchone()

    if not analysis:
        conn.close()
        flash('Analysis not found.', 'error')
        return redirect(url_for('input_form'))

    # 🔹 Fetch user (safe – no column assumptions)
    user = conn.execute(
        'SELECT * FROM users WHERE id = ?',
        (session['user_id'],)
    ).fetchone()

    conn.close()

    # 🔹 Safely resolve username & email
    username = (
        user['username']
        if user and 'username' in user.keys()
        else user['email'].split('@')[0]
        if user and 'email' in user.keys()
        else analysis['student_id']
    )

    email = (
        user['email']
        if user and 'email' in user.keys()
        else 'Not Available'
    )

    # 🔹 Recommendations
    student_info = {
        'hours_studied': analysis['hours_studied'],
        'attendance': analysis['attendance'],
        'sleep_hours': analysis['sleep_hours']
    }

    recommendations = recommend_study_habits(
        analysis['cluster_label'],
        student_info
    )

    # 🔹 Final payload
    feedback_data = {
        'student_id': analysis['student_id'],
        'username': username,
        'email': email,
        'hours_studied': analysis['hours_studied'],
        'attendance': analysis['attendance'],
        'sleep_hours': analysis['sleep_hours'],
        'previous_scores': analysis['previous_scores'],
        'tutoring_sessions': analysis['tutoring_sessions'],
        'physical_activity': analysis['physical_activity'],
        'cluster': analysis['cluster'],
        'cluster_label': analysis['cluster_label'],
        'predicted_score': analysis['predicted_score'],
        'recommendations': recommendations,
        'is_admin': session.get('role') == 'admin'
    }

    return render_template('feedback.html', **feedback_data)



@app.route('/my_recommendations')
@login_required
def my_recommendations():
    # Redirect admins - they don't need to view their own recommendations
    if session.get('role') == 'admin':
        flash('Admins can view all recommendations from the admin dashboard.', 'info')
        return redirect(url_for('admin_recommendations'))
    
    conn = get_db()
    
    recommendations = conn.execute('''SELECT r.*, a.student_id, a.predicted_score, a.cluster_label
                                     FROM recommendations r
                                     JOIN analysis_history a ON r.analysis_id = a.id
                                     WHERE r.user_id = ?
                                     ORDER BY r.created_at DESC''',
                                  (session['user_id'],)).fetchall()
    
    conn.close()
    
    # Parse the recommendations JSON for each record
    recommendations_list = []
    for rec in recommendations:
        rec_dict = dict(rec)
        # Parse the JSON string to get the actual recommendations list
        try:
            if rec_dict['recommendations']:
                rec_dict['recommendations_list'] = json.loads(rec_dict['recommendations'])
            else:
                rec_dict['recommendations_list'] = []
        except (json.JSONDecodeError, TypeError):
            rec_dict['recommendations_list'] = []
        
        recommendations_list.append(rec_dict)
    
    return render_template('student_recommendations.html', recommendations=recommendations_list)

@app.route('/analytics')
@login_required
def analytics():
    """User analytics - only their own data"""
    # Redirect admins to admin analytics
    if session.get('role') == 'admin':
        return redirect(url_for('admin_analytics'))
    
    conn = get_db()
    history = conn.execute('''SELECT * FROM analysis_history 
                             WHERE user_id = ? 
                             ORDER BY analysis_date ASC''',
                          (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('analytics.html', history=history)

@app.route('/goals', methods=['GET', 'POST'])
@login_required
def goals():
    # Redirect admins away from student features
    if session.get('role') == 'admin':
        flash('This feature is for students only.', 'info')
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db()
    
    if request.method == 'POST':
        goal_type = request.form['goal_type']
        target_value = float(request.form['target_value'])
        current_value = float(request.form.get('current_value', 0))
        deadline = request.form['deadline']
        
        conn.execute('''INSERT INTO goals 
                       (user_id, goal_type, target_value, current_value, deadline, status)
                       VALUES (?, ?, ?, ?, ?, 'active')''',
                    (session['user_id'], goal_type, target_value, current_value, deadline))
        conn.commit()
        flash('Goal created successfully!', 'success')
    
    goals_list = conn.execute('''SELECT * FROM goals 
                                WHERE user_id = ? 
                                ORDER BY created_at DESC''',
                            (session['user_id'],)).fetchall()
    
    # Convert Row objects to dictionaries and add progress percentage
    goals_with_progress = []
    for goal in goals_list:
        goal_dict = dict(goal)
        if goal_dict['target_value'] > 0:
            progress = min(100, (goal_dict['current_value'] / goal_dict['target_value']) * 100)
            goal_dict['progress_percent'] = round(progress, 1)
            # Auto-complete goals that reached target
            if progress >= 100 and goal_dict['status'] != 'completed':
                conn.execute('UPDATE goals SET status = "completed" WHERE id = ?', (goal_dict['id'],))
                conn.commit()
                goal_dict['status'] = 'completed'
        else:
            goal_dict['progress_percent'] = 0
        goals_with_progress.append(goal_dict)
    
    conn.close()
    
    return render_template('goals.html', goals=goals_with_progress)


@app.route('/goals/<int:goal_id>/update', methods=['POST'])
@login_required
def update_goal(goal_id):
    if session.get('role') == 'admin':
        flash('This feature is for students only.', 'info')
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db()
    
    # Check if this is a full edit or quick update
    goal_type = request.form.get('goal_type')
    
    if goal_type:  # Full edit with all fields
        target_value = float(request.form['target_value'])
        current_value = float(request.form['current_value'])
        deadline = request.form['deadline']
        
        # Verify goal belongs to user
        goal = conn.execute('SELECT * FROM goals WHERE id = ? AND user_id = ?',
                           (goal_id, session['user_id'])).fetchone()
        
        if goal:
            # Calculate new status
            progress = (current_value / target_value * 100) if target_value > 0 else 0
            new_status = 'completed' if progress >= 100 else 'active'
            
            conn.execute('''UPDATE goals 
                           SET goal_type = ?, target_value = ?, current_value = ?, 
                               deadline = ?, status = ?
                           WHERE id = ?''',
                        (goal_type, target_value, current_value, deadline, new_status, goal_id))
            conn.commit()
            flash('Goal updated successfully!', 'success')
        else:
            flash('Goal not found.', 'error')
    else:  # Quick update (only current_value)
        current_value = float(request.form['current_value'])
        
        # Get goal to calculate new status
        goal = conn.execute('SELECT * FROM goals WHERE id = ? AND user_id = ?',
                           (goal_id, session['user_id'])).fetchone()
        
        if goal:
            target_value = goal['target_value']
            progress = (current_value / target_value * 100) if target_value > 0 else 0
            new_status = 'completed' if progress >= 100 else 'active'
            
            conn.execute('UPDATE goals SET current_value = ?, status = ? WHERE id = ?',
                        (current_value, new_status, goal_id))
            conn.commit()
            
            if new_status == 'completed':
                flash('🎉 Congratulations! Goal completed!', 'success')
            else:
                flash('Goal progress updated!', 'success')
        else:
            flash('Goal not found.', 'error')
    
    conn.close()
    return redirect(url_for('goals'))


@app.route('/goals/<int:goal_id>/delete', methods=['POST'])
@login_required
def delete_goal(goal_id):
    if session.get('role') == 'admin':
        flash('This feature is for students only.', 'info')
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db()
    
    # Verify goal belongs to user
    goal = conn.execute('SELECT * FROM goals WHERE id = ? AND user_id = ?',
                       (goal_id, session['user_id'])).fetchone()
    
    if goal:
        conn.execute('DELETE FROM goals WHERE id = ?', (goal_id,))
        conn.commit()
        flash('Goal deleted successfully!', 'success')
    else:
        flash('Goal not found.', 'error')
    
    conn.close()
    return redirect(url_for('goals'))

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    # Redirect admins - chat is for students
    if session.get('role') == 'admin':
        flash('AI Coach is available for students only.', 'info')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        user_message = request.json.get('message', '')
        response = generate_coach_response(user_message)
        
        conn = get_db()
        conn.execute('''INSERT INTO chat_history (user_id, message, response)
                       VALUES (?, ?, ?)''',
                    (session['user_id'], user_message, response))
        conn.commit()
        conn.close()
        
        return jsonify({'response': response})
    
    conn = get_db()
    history = conn.execute('''SELECT * FROM chat_history 
                             WHERE user_id = ? 
                             ORDER BY timestamp DESC LIMIT 50''',
                          (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('chat.html', history=reversed(list(history)))

if __name__ == '__main__':
    app.run(debug=True, port=5000)