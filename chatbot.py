from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import mysql.connector
import os
import bcrypt
import secrets
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from flask_cors import CORS
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load environment variables - Fix the path
# Get the directory where chatbot.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to .env file in StartPage folder
env_path = os.path.join(current_dir, 'StartPage', '.env')
load_dotenv(env_path)



# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('focusmate.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# MySQL connection configuration
def get_db_connection():
    """Create and return a database connection using environment variables"""
    try:
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),  # Get from environment variable
            'database': os.getenv('DB_NAME', 'focusmate')
        }
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        return None

# Database initialization function
def init_db():
    """Initialize the database with required tables if they don't exist"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '')
        )
        cursor = conn.cursor()
        
        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS focusmate")
        cursor.execute("USE focusmate")
        
        # Create users table
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
        
        # Create study sessions table
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
        
        # Create emotion analyses table
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
        logger.info("Database initialization completed successfully")
    except mysql.connector.Error as err:
        logger.error(f"Database initialization error: {err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# Emotion mapping dictionary
EMOTION_MAPPING = {
    0: "neutral",   # Neutral expression
    1: "sleepy",   # Sleepy
    2: "tired",     # Tired/fatigued
}

# Emotion information and study suggestions
# emotion_info = {
#     "sleepy": """
#     졸린 상태가 감지되었습니다.  
#     집중력과 인지 능력이 일시적으로 저하될 수 있습니다. 학습 전에 휴식이 필요할 수 있습니다.

#     학습 제안:
#     - 20~30분 정도의 파워냅을 취해보세요
#     - 밝은 조명 아래에서 학습 환경을 유지해보세요
#     - 세수를 하거나 찬물로 얼굴을 씻어보세요
#     - 카페인이 함유된 음료를 적당히 섭취해보세요 (과도한 섭취는 피하세요)
#     - 집중이 어려울 경우, 반복 학습보다는 간단한 정리나 복습 위주로 진행해보세요
#     """,
#     "tired": """
#     지친 상태가 감지되었습니다.
#     피로가 쌓여 학습 효율이 저하될 수 있는 상태입니다. 잠시 휴식이 필요할 수 있습니다.
    
#     학습 제안:
#     - 15-20분 정도의 짧은 낮잠을 고려해보세요
#     - 가벼운 운동이나 산책으로 기분 전환을 해보세요
#     - 물을 마시고 간단한 간식으로 에너지를 보충해보세요
#     - 학습 내용을 작은 단위로 나누어 하나씩 진행해보세요
#     - 필요하다면 10분간 가벼운 스트레칭을 해보세요
#     """,
#     "neutral": """
#     중립적인 상태가 감지되었습니다.
#     안정적이고 균형 잡힌 감정 상태입니다. 차분히 학습을 시작하기 좋은 상태입니다.
    
#     학습 제안:
#     - 우선순위가 높은 과제부터 차근차근 시작해보세요
#     - 학습 계획을 세우고 목표를 설정하기 좋은 시간입니다
#     - 집중력을 방해하는 요소들을 미리 제거해보세요
#     """,
# }

# RAG (Retrieval Augmented Generation) system setup
def setup_rag_system():
    """Create a FAISS vector store from emotion information"""
    try:
        docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for emotion, info in emotion_info.items():
                # Create temporary files and load data
                temp_file = os.path.join(temp_dir, f"{emotion}.txt")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(info)
                docs.extend(TextLoader(temp_file, encoding="utf-8").load())

            # Split documents and create embeddings
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            return FAISS.from_documents(
                splitter.split_documents(docs),
                HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
            )
    except Exception as e:
        logger.error(f"RAG system setup error: {e}")
        return None

# Initialize the vector store
vectorstore = setup_rag_system()

# Initialize Gemini API
def setup_gemini_model():
    """Initialize and return the Gemini generative model"""
    try:
        # Get API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set in environment variables")
            return None
            
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Generation configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Safety settings
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

        # Create and return the model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        logger.info("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Gemini model initialization error: {e}")
        return None

# Initialize Gemini model
gemini_model = setup_gemini_model()

# Retrieve relevant context for a query
def retrieve_relevant_context(query):
    """Search the vector store for context relevant to the query"""
    try:
        if not vectorstore:
            logger.error("Vector store not initialized")
            return ""
            
        # Search for similar documents
        documents = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in documents])
        return context
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return ""

# Generate a response using Gemini
def generate_gemini_response(query, context, emotion=None):
    """Generate a response using the Gemini model"""
    try:
        if not gemini_model:
            return "Gemini 모델이 초기화되지 않았습니다."

        # Create a detailed prompt
        emotion_context = f"사용자의 감지된 감정 상태: {emotion}\n" if emotion else ""
        
        prompt = f"""
        다음은 사용자의 학습 상태와 질문입니다:

        감정 상태: {emotion if emotion else "neutral"}

        {context}

        사용자 질문: {query}

        아래 조건에 따라 답변을 작성하세요:
        - 무조권 짧게 친구같이 대답하세요.
        - 꼭 존대말 쓰세요. 존대말 필수. 
        - 긍정적이고 격려하는 어조로 작성하세요.
        - 감정 상태가 'sleepy'이면, 졸음을 극복하고 집중을 회복할 수 있는 실질적이고 따뜻한 조언을 포함하세요.
        - 감정 상태가 'tired'이면, 피로를 해소하고 에너지를 회복할 수 있는 실질적이고 친근한 조언을 포함하세요.
        - 감정 상태가 'neutral' 또는 기본값이면, 집중력 유지와 학습 동기 부여에 도움이 되는 조언을 포함하세요.
        - 질문에 대한 구체적이고 실행 가능한 학습 팁을 제공하세요.
        """

        # Generate content
        response = gemini_model.generate_content(prompt)

        # Return response text
        if hasattr(response, "text"):
            return response.text
        else:
            return "응답을 처리할 수 없습니다."

    except Exception as e:
        logger.error(f"Gemini response generation error: {e}")
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

# Detect emotion from image
def detect_emotion_from_image(image_data, face_analyzer):
    """Detect emotion from an image using facial features"""
    try:
        # Decode image
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 64 or img.shape[1] < 64:
            return {"error": "이미지를 처리할 수 없습니다"}

        # Detect faces
        faces = face_analyzer.get(img)
        if not faces:
            return {"error": "얼굴이 감지되지 않았습니다"}

        face = faces[0]
        landmarks = None

        # Try to extract landmarks
        landmark_attributes = [
            ("kps", "KPS"),
            ("landmark_2d", "2D"),
            ("landmark_3d", "3D"),
        ]
        
        for attr, desc in landmark_attributes:
            if hasattr(face, attr) and getattr(face, attr) is not None:
                curr_landmarks = getattr(face, attr)
                if len(curr_landmarks) >= 68:
                    landmarks = (
                        curr_landmarks[:, :2]
                        if attr == "landmark_3d"
                        else curr_landmarks
                    )
                    logger.info(f"{desc} landmarks being used")
                    break

        if landmarks is None:
            return {"error": "얼굴 특징점을 추출할 수 없습니다"}

        # Process landmarks
        landmarks = landmarks.astype(np.float32)
        landmarks /= np.array([img.shape[1], img.shape[0]])

        # Calculate distances
        eye_distance = np.linalg.norm(
            np.mean(landmarks[36:42], axis=0) - np.mean(landmarks[42:48], axis=0)
        )
        mouth_height = np.linalg.norm(
            np.mean(landmarks[51:54], axis=0) - np.mean(landmarks[57:60], axis=0)
        )

        if np.isnan(eye_distance) or np.isnan(mouth_height):
            return {"error": "특징점 거리 계산에 실패했습니다"}

        # Classify emotion based on facial features
        if mouth_height > 0.15:
            emotion_idx = 2  # interested
        elif eye_distance > 0.12:
            emotion_idx = 1  # focused
        elif mouth_height < 0.06:
            emotion_idx = 4  # tired
        else:
            emotion_idx = 0  # neutral

        # Get emotion label and create confidence scores
        dominant_emotion = EMOTION_MAPPING[emotion_idx]
        
        # Generate result with confidence scores and debug info
        result = {
            "dominant_emotion": dominant_emotion,
            "all_emotions": {
                emotion: 0.1
                for emotion in EMOTION_MAPPING.values()
                if emotion != dominant_emotion
            } | {dominant_emotion: 0.6},
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
        
        return result

    except Exception as e:
        logger.error(f"Emotion detection error: {str(e)}")
        return {"error": str(e)}


# Generate a response based on emotion data and query
def generate_response(emotion_data, query=None):
    """Generate a response based on detected emotion and optional query"""
    try:
        # Check for errors
        if "error" in emotion_data:
            return {"response": f"오류가 발생했습니다: {emotion_data['error']}"}

        # Get base response from emotion info
        dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
        #base_response = emotion_info.get(dominant_emotion, emotion_info["neutral"])
        base_response = ""

        # If there's a query, generate a custom response with Gemini
        if query and gemini_model:
            def markdown_to_html(text):
                # Replace **text** with <b>text</b>
                return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            try:
                # Build context
                context = f"현재 감지된 감정 상태: {dominant_emotion}\n"
                context += retrieve_relevant_context(query)

                # Generate custom response
                custom_response = generate_gemini_response(query, context, dominant_emotion)
                custom_response = markdown_to_html(custom_response)

                # Combine responses
                return {
                    "response": base_response + custom_response,
                    "chat_response": True,
                    "emotion": dominant_emotion
                }
            except Exception as chat_error:
                logger.error(f"Chat response generation error: {chat_error}")
                return {"response": base_response, "emotion": dominant_emotion}

        # Return base response if no query
        return {"response": base_response, "emotion": dominant_emotion}

    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        return {"response": f"응답 생성 중 오류가 발생했습니다: {str(e)}"}

# In-memory conversation store
conversation_store = {}

# Initialize DB on startup
if __name__ == "__main__":
    init_db()

# Ensure these variables and functions are available for import
__all__ = [
    'generate_response',
    'retrieve_relevant_context', 
    'conversation_store',
    'gemini_model',
    'emotion_info',
    'setup_gemini_model',
    'detect_emotion_from_image'
]