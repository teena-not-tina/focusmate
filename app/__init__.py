# Flask 앱 초기화 및 설정

from flask import Flask, render_template, request, jsonify

def create_app():
    app = Flask(__name__, 
                static_folder="../static", 
                template_folder="../templates")
    
    # 라우트 정의
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/detect_emotion', methods=['POST'])
    def detect_emotion():
        # 얼굴 감정 감지 API 로직
        ...
    
    # 기타 라우트...
    
    return app
