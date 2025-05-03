from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import mysql.connector
import os
import bcrypt
import secrets
from datetime import datetime, timedelta
# test.py에서 가져온 임포트
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from flask_cors import CORS
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging
from logging.handlers import RotatingFileHandler
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 로깅 설정
if not os.path.exists("logs"):
    os.mkdir("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/app.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__,
            static_folder='static',      
            static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))
CORS(app)  # 모든 라우트에 CORS 허용

# MySQL 연결 설정
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='your_mysql_password',  # 실제 MySQL 비밀번호로 변경
        database='focusmate'
    )

# 데이터베이스 초기화
def init_db():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='your_mysql_password'  # 실제 MySQL 비밀번호로 변경
        )
        cursor = conn.cursor()
        
        # 데이터베이스 생성
        cursor.execute("CREATE DATABASE IF NOT EXISTS focusmate")
        cursor.execute("USE focusmate")
        
        # 사용자 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            email VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            user_type ENUM('student', 'teacher') NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL
        )
        """)
        
        # 학습 세션 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP NULL,
            duration INT DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # 감정 분석 결과 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotion_analyses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            session_id INT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            emotion VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (session_id) REFERENCES study_sessions(id)
        )
        """)
        
        conn.commit()
        print("데이터베이스 초기화 완료")
    except mysql.connector.Error as err:
        print(f"데이터베이스 초기화 오류: {err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# InsightFace 모델 초기화
face_analyzer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# 감정 매핑
EMOTION_MAPPING = {
    0: "neutral",  # 중립
    1: "focused",  # 집중
    2: "interested",  # 흥미
    3: "anxious",  # 불안
    4: "tired",  # 지침
}

# 학습 관련 감정 상태 데이터베이스
emotion_info = {
    "focused": """
    집중 상태가 감지되었습니다.
    이는 학습에 가장 이상적인 상태입니다. 현재 뇌가 새로운 정보를 받아들이고 처리하기에 최적의 컨디션입니다.
    
    학습 제안:
    - 지금이 복잡한 개념을 이해하거나 문제 해결하기 좋은 시간입니다
    - 25분 집중 후 5분 휴식의 포모도로 기법을 활용해보세요
    - 현재의 집중력을 유지하면서 학습 목표를 달성해보세요
    """,
    "interested": """
    흥미로운 상태가 감지되었습니다.
    호기심과 학습 동기가 높은 상태입니다. 이런 감정은 새로운 지식 습득에 매우 효과적입니다.
    
    학습 제안:
    - 현재의 흥미를 활용해 새로운 주제를 탐험해보세요
    - 관심 있는 주제부터 시작해 연관 개념으로 확장해보세요
    - 능동적 학습 방법(프로젝트, 실험 등)을 시도해보세요
    """,
    "anxious": """
    불안한 상태가 감지되었습니다.
    시험이나 과제에 대한 압박감을 느끼고 있을 수 있습니다. 이는 자연스러운 반응이지만 관리가 필요합니다.
    
    학습 제안:
    - 잠시 심호흡하고 마음을 진정시켜보세요
    - 학습 내용을 작은 단위로 나누어 하나씩 진행해보세요
    - 필요하다면 10분간 가벼운 스트레칭을 해보세요
    """,
    "tired": """
    지친 상태가 감지되었습니다.
    피로가 쌓여 학습 효율이 저하될 수 있는 상태입니다. 잠시 휴식이 필요할 수 있습니다.
    
    학습 제안:
    - 15-20분 정도의 짧은 낮잠을 고려해보세요
    - 가벼운 운동이나 산책으로 기분 전환을 해보세요
    - 물을 마시고 간단한 간식으로 에너지를 보충해보세요
    """,
    "neutral": """
    중립적인 상태가 감지되었습니다.
    안정적이고 균형 잡힌 감정 상태입니다. 차분히 학습을 시작하기 좋은 상태입니다.
    
    학습 제안:
    - 우선순위가 높은 과제부터 차근차근 시작해보세요
    - 학습 계획을 세우고 목표를 설정하기 좋은 시간입니다
    - 집중력을 방해하는 요소들을 미리 제거해보세요
    """,
}

# RAG 시스템 설정
def setup_rag_system():
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for emotion, info in emotion_info.items():
            # 임시 파일 생성 및 데이터 로드
            temp_file = os.path.join(temp_dir, f"{emotion}.txt")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(info)
            docs.extend(TextLoader(temp_file, encoding="utf-8").load())

        # 문서 분할 및 임베딩
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return FAISS.from_documents(
            splitter.split_documents(docs),
            HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
        )

vectorstore = setup_rag_system()

# Gemini 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

# Gemini 모델 초기화 함수
def setup_gemini_model():
    try:
        logger.info("Gemini 모델 초기화 중...")

        # 생성 설정
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # 안전 설정
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        # Gemini Pro 모델 사용
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return model
    except Exception as e:
        logger.error(f"Gemini 모델 초기화 오류: {str(e)}")
        return None

# Gemini 모델 초기화
gemini_model = setup_gemini_model()

# 관련 컨텍스트 검색 함수
def retrieve_relevant_context(query):
    try:
        # 기존 vectorstore를 사용하여 관련 문서 검색
        documents = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in documents])
        return context
    except Exception as e:
        logger.error(f"컨텍스트 검색 오류: {str(e)}")
        return ""

# Gemini 응답 생성 함수
def generate_gemini_response(query, context):
    try:
        if not gemini_model:
            return "Gemini 모델이 초기화되지 않았습니다."

        prompt = f"""
        다음은 사용자의 학습 상태와 질문입니다:
        
        {context}
        
        사용자 질문: {query}
        
        위 내용을 바탕으로 학습자에게 도움이 되는 구체적인 조언을 제공해주세요.
        학습 효율성을 높이기 위한 실용적인 팁을 포함해주세요.
        
        중요: 단순히 답만 제시하지 말고, 친절하고 자세한 설명을 추가해주세요.
        예를 들어, 수학 문제의 경우 단순히 "답은 X입니다"가 아니라 
        "이 문제의 답은 X입니다. 이렇게 계산할 수 있어요..." 와 같이 친절하게 설명하세요.
        
        친구처럼 대화하듯이 답변해주세요.
        """

        # generate_content를 사용하여 응답 생성
        response = gemini_model.generate_content(prompt)

        # 응답 텍스트 반환
        if hasattr(response, "text"):
            return response.text
        else:
            return "응답을 처리할 수 없습니다."

    except Exception as e:
        logger.error(f"Gemini 응답 생성 오류: {str(e)}")
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

# 이미지에서 감정 감지
def detect_emotion_from_image(image_data):
    try:
        # 이미지 로드 및 기본 검증
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 64 or img.shape[1] < 64:
            return {"error": "이미지를 처리할 수 없습니다"}

        # 얼굴 감지 및 랜드마크 추출
        faces = face_analyzer.get(img)
        if not faces:
            return {"error": "얼굴이 감지되지 않았습니다"}

        face = faces[0]
        landmarks = None

        # 랜드마크 추출 시도
        for attr, desc in [
            ("kps", "KPS"),
            ("landmark_2d", "2D"),
            ("landmark_3d", "3D"),
        ]:
            if hasattr(face, attr) and getattr(face, attr) is not None:
                curr_landmarks = getattr(face, attr)
                if len(curr_landmarks) >= 68:
                    landmarks = (
                        curr_landmarks[:, :2]
                        if attr == "landmark_3d"
                        else curr_landmarks
                    )
                    logger.info(f"{desc} 랜드마크 사용")
                    break

        if landmarks is None:
            return {"error": "얼굴 특징점을 추출할 수 없습니다"}

        try:
            # 좌표 정규화 및 특징점 계산
            landmarks = landmarks.astype(np.float32)
            landmarks /= np.array([img.shape[1], img.shape[0]])

            # 거리 계산
            eye_distance = np.linalg.norm(
                np.mean(landmarks[36:42], axis=0) - np.mean(landmarks[42:48], axis=0)
            )
            mouth_height = np.linalg.norm(
                np.mean(landmarks[51:54], axis=0) - np.mean(landmarks[57:60], axis=0)
            )

            if np.isnan(eye_distance) or np.isnan(mouth_height):
                return {"error": "특징점 거리 계산에 실패했습니다"}

            # 감정 상태 분류
            if mouth_height > 0.15:
                emotion_idx = 2  # interested
            elif eye_distance > 0.12:
                emotion_idx = 1  # focused
            elif mouth_height < 0.06:
                emotion_idx = 4  # tired
            else:
                emotion_idx = 0  # neutral

            dominant_emotion = EMOTION_MAPPING[emotion_idx]

            return {
                "dominant_emotion": dominant_emotion,
                "all_emotions": {
                    emotion: 0.1
                    for emotion in EMOTION_MAPPING.values()
                    if emotion != dominant_emotion
                }
                | {dominant_emotion: 0.6},
                "debug_info": {
                    "eye_distance": float(eye_distance),
                    "mouth_height": float(mouth_height),
                    "image_size": img.shape,
                    "landmarks_count": len(landmarks),
                    "landmark_type": next(
                        attr
                        for attr in ["kps", "landmark_2d", "landmark_3d"]
                        if hasattr(face, attr) and getattr(face, attr) is not None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"특징점 처리 오류: {str(e)}")
            return {"error": f"특징점 처리 중 오류 발생: {str(e)}"}

    except Exception as e:
        logger.error(f"감정 감지 오류: {str(e)}")
        return {"error": str(e)}

# 비디오에서 감정 감지
def detect_emotion_from_video(video_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_data)
            video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)
        emotions_count = {emotion: 0 for emotion in EMOTION_MAPPING.values()}
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or processed_frames >= 100:  # 최대 100프레임 처리
                break

            if processed_frames % 5 == 0:  # 5프레임마다 처리
                faces = face_analyzer.get(frame)
                if faces and faces[0].landmark_2d is not None:
                    landmarks = faces[0].landmark_2d
                    # 특징점 기반 감정 분류
                    eye_dist = np.mean(np.abs(landmarks[36:42] - landmarks[42:48]))
                    mouth_h = np.mean(np.abs(landmarks[51:54] - landmarks[57:60]))

                    emotion = (
                        "interested"
                        if mouth_h > 0.2
                        else (
                            "focused"
                            if eye_dist > 0.15
                            else "tired" if mouth_h < 0.1 else "neutral"
                        )
                    )

                    emotions_count[emotion] += 1
                    processed_frames += 1

        cap.release()
        os.unlink(video_path)

        if processed_frames == 0:
            return {"error": "영상에서 얼굴이 감지되지 않았습니다"}

        dominant_emotion = max(emotions_count, key=emotions_count.get)
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_ratios": {
                k: v / processed_frames for k, v in emotions_count.items() if v > 0
            },
            "frames_processed": processed_frames,
        }

    except Exception as e:
        logger.error(f"비디오 처리 오류: {str(e)}")
        return {"error": str(e)}

# 응답 생성 함수
def generate_response(emotion_data, query=None):
    try:
        if "error" in emotion_data:
            return {"response": f"오류가 발생했습니다: {emotion_data['error']}"}

        # 기본 감정 상태 정보
        base_response = emotion_info[emotion_data.get("dominant_emotion", "neutral")]

        # 사용자 질문이 있는 경우 Gemini로 추가 응답 생성
        if query and gemini_model:
            try:
                # 현재 감정 상태와 함께 질문 구성
                context = f"현재 감지된 감정 상태: {emotion_data.get('dominant_emotion', 'neutral')}\n"
                context += retrieve_relevant_context(query)

                custom_response = generate_gemini_response(query, context)

                # 기본 응답과 Gemini 응답 결합
                return {
                    "response": base_response + "\n\n추가 답변:\n" + custom_response,
                    "chat_response": True,
                }

            except Exception as chat_error:
                logger.error(f"Gemini 응답 생성 오류: {chat_error}")
                return {"response": base_response}

        return {"response": base_response}

    except Exception as e:
        logger.error(f"응답 생성 오류: {str(e)}")
        return {"response": f"응답 생성 중 오류가 발생했습니다: {str(e)}"}

# 대화 기록 관리를 위한 간단한 인메모리 스토리지
conversation_store = {}

# 서버 시작 시 DB 초기화
init_db()

#####################################################
# 라우트 정의
#####################################################

# 메인 페이지
@app.route('/')
def index():
    # 세션에서 사용자 정보 확인
    username = session.get('username')
    is_logged_in = 'user_id' in session
    theme = session.get('theme', 'light')  # 기본값은 라이트 모드
    
    return render_template('index.html', 
                          is_logged_in=is_logged_in, 
                          username=username,
                          theme=theme)

# 인증 관련 라우트 - 로그인, 회원가입 등을 /auth 접두사로 통일
@app.route('/auth/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))  # 이미 로그인된 사용자는 메인으로
    theme = session.get('theme', 'light')
    return render_template('auth/login.html', theme=theme)

@app.route('/auth/signup')
def signup_page():
    if 'user_id' in session:
        return redirect(url_for('index'))  # 이미 로그인된 사용자는 메인으로
    theme = session.get('theme', 'light')
    return render_template('auth/signup.html', theme=theme)

@app.route('/auth/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/auth/forgot-password')
def forgot_password_page():
    theme = session.get('theme', 'light')
    return render_template('auth/forgot-password.html', theme=theme)

# 테마 변경 라우트
@app.route("/auth/theme")
def toggle_theme():
    # 현재 테마 확인
    current_theme = session.get('theme', 'light')
    
    # 테마 전환
    session['theme'] = 'dark' if current_theme == 'light' else 'light'
    
    # 리디렉션 (이전 페이지로 돌아가기)
    referrer = request.referrer
    if referrer and referrer.startswith(request.host_url):
        return redirect(referrer)
    return redirect(url_for('index'))

# API 엔드포인트 - /api 접두사로 통일
@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.json
    
    username = data.get('username', '').strip()
    password = data.get('password', '')
    remember = data.get('remember', False)
    
    if not username or not password:
        return jsonify({'success': False, 'message': '아이디와 비밀번호를 모두 입력해주세요.'})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 사용자 조회
        cursor.execute("SELECT id, username, password, user_type FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'success': False, 'message': '아이디 또는 비밀번호가 일치하지 않습니다.'})
        
        # 비밀번호 검증
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            # 세션 설정
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_type'] = user['user_type']
            
            # 로그인 상태 유지 (remember me)
            if remember:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)
            
            # 마지막 로그인 시간 업데이트
            cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user['id'],))
            conn.commit()
            
            return jsonify({'success': True, 'redirect': '/'})
        else:
            return jsonify({'success': False, 'message': '아이디 또는 비밀번호가 일치하지 않습니다.'})
    
    except mysql.connector.Error as err:
        logger.error(f"데이터베이스 오류: {err}")
        return jsonify({'success': False, 'message': '서버 오류가 발생했습니다.'})
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/api/signup', methods=['POST'])
def signup_api():
    data = request.json
    
    # 필수 필드 검사
    required_fields = ['username', 'email', 'password', 'user_type']
    for field in required_fields:
        if field not in data or not data[field].strip():
            return jsonify({'success': False, 'message': f'{field} 필드는 필수입니다.', 'field': field})
    
    username = data['username'].strip()
    email = data['email'].strip()
    password = data['password']
    user_type = data['user_type']
    
    # 유효성 검사
    if len(username) < 4:
        return jsonify({'success': False, 'message': '아이디는 4자 이상이어야 합니다.', 'field': 'username'})
    
    if len(password) < 8:
        return jsonify({'success': False, 'message': '비밀번호는 8자 이상이어야 합니다.', 'field': 'password'})
    
    # 비밀번호 해싱
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 사용자 이름 중복 검사
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': '이미 사용 중인 아이디입니다.', 'field': 'username'})
        
        # 이메일 중복 검사
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': '이미 사용 중인 이메일입니다.', 'field': 'email'})
        
        # 사용자 등록
        cursor.execute(
            "INSERT INTO users (username, email, password, user_type) VALUES (%s, %s, %s, %s)",
            (username, email, hashed_password, user_type)
        )
        conn.commit()
        
        return jsonify({'success': True, 'message': '회원가입이 완료되었습니다.'})
    
    except mysql.connector.Error as err:
        logger.error(f"데이터베이스 오류: {err}")
        return jsonify({'success': False, 'message': '서버 오류가 발생했습니다.'})
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/api/check_username')
def check_username_api():
    username = request.args.get('username', '').strip()
    
    if not username or len(username) < 4:
        return jsonify({'available': False})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        return jsonify({'available': result is None})
    
    except mysql.connector.Error as err:
        logger.error(f"데이터베이스 오류: {err}")
        return jsonify({'available': False})
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/api/check_email')
def check_email_api():
    email = request.args.get('email', '').strip()
    
    if not email:
        return jsonify({'available': False})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        
        return jsonify({'available': result is None})
    
    except mysql.connector.Error as err:
        logger.error(f"데이터베이스 오류: {err}")
        return jsonify({'available': False})
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 감정 감지 API
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    if "file" not in request.files:
        return jsonify({"error": "파일이 제공되지 않았습니다"})

    file = request.files["file"]
    file_type = request.form.get("type", "image")
    query = request.form.get("query", "")

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다"})

    try:
        file_data = file.read()

        if file_type == "image":
            emotion_data = detect_emotion_from_image(file_data)
        else:
            emotion_data = detect_emotion_from_video(file_data)

        response_data = generate_response(emotion_data, query)
        response_data.update(emotion_data)

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"API 오류: {str(e)}")
        return jsonify({"error": str(e)})

# 채팅 API
@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "유효하지 않은 요청 데이터"}), 400

        messages = data.get("messages", [])
        session_id = data.get("session_id", "default_session")
        temperature = data.get("temperature", 0.3)
        max_tokens = data.get("max_tokens", 1000)

        if session_id not in conversation_store:
            conversation_store[session_id] = []

        try:
            # Gemini API 호출
            if not gemini_model:
                return jsonify({"error": "Gemini 모델이 초기화되지 않았습니다"}), 500

            # 메시지 구성 - 시스템 메시지와 사용자 대화 분리
            system_message = None
            chat_messages = []
            user_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_message = msg["content"]
                    chat_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_messages.append({"role": "assistant", "content": msg["content"]})

            if not user_message:
                return jsonify({"error": "사용자 메시지가 없습니다"}), 400

            # 모델에 적용할 커스텀 생성 구성 설정
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": max_tokens,
            }

            # 모델 설정 업데이트
            gemini_model.generation_config = generation_config

            # 최종 프롬프트 구성 - 시스템 메시지를 최우선으로 적용
            prompt = f"{system_message}\n\n{user_message}" if system_message else user_message

            # 이전 대화 컨텍스트 구성 (시스템 메시지 제외)
            history = []
            for msg in conversation_store[session_id]:
                if msg["role"] != "system":
                    content = [{"text": msg["content"]}]
                    history.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": content
                    })

            # 응답 생성
            if not history:
                # 대화 기록이 없으면 단일 프롬프트로 요청
                response = gemini_model.generate_content(prompt)
            else:
                # 대화 기록이 있으면 채팅 세션 사용
                chat = gemini_model.start_chat(history=history)
                response = chat.send_message(prompt)

            if not hasattr(response, 'text') or not response.text:
                logger.error("Gemini API 응답이 비어 있습니다")
                return jsonify({"error": "AI 응답을 생성할 수 없습니다"}), 500

            assistant_message = response.text.strip()

            # 대화 기록 업데이트
            conversation_store[session_id].append({"role": "user", "content": user_message})
            conversation_store[session_id].append({"role": "assistant", "content": assistant_message})

            # 대화 기록 제한 (최대 20개)
            if len(conversation_store[session_id]) > 20:
                conversation_store[session_id] = conversation_store[session_id][-20:]

            return jsonify({"message": assistant_message})

        except Exception as chat_error:
            logger.error(f"Gemini API 오류: {str(chat_error)}", exc_info=True)
            return jsonify({
                "error": "AI 응답 생성 중 오류가 발생했습니다",
                "details": str(chat_error)
            }), 500

    except Exception as e:
        logger.error(f"채팅 API 오류: {str(e)}", exc_info=True)
        return jsonify({"error": "서버 오류가 발생했습니다", "details": str(e)}), 500

# URL 패턴 리디렉션 (이전 URL과의 호환성 유지)
@app.route('/login')
def login_redirect():
    return redirect(url_for('login_page'))

@app.route('/signup')
def signup_redirect():
    return redirect(url_for('signup_page'))

@app.route('/logout')
def logout_redirect():
    return redirect(url_for('logout'))

@app.route('/theme')
def theme_redirect():
    return redirect(url_for('toggle_theme'))

if __name__ == "__main__":
    app.run(debug=True)