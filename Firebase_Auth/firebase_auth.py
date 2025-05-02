import firebase_admin
from firebase_admin import credentials, auth
import requests

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

FIREBASE_API_KEY = "AIzaSyAE92bSgXTgVrZnof9DeKWYflmio7TKqbE"  # 반드시 교체 필요

def signup_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return True, "✅ 회원가입 성공!"
    except auth.EmailAlreadyExistsError:
        return False, "⚠️ 이미 존재하는 이메일입니다."
    except Exception as e:
        return False, f"❌ 회원가입 실패: {e}"

def login_user(email, password):
    try:
        if FIREBASE_API_KEY == "YOUR_FIREBASE_API_KEY":
            return False, "❗ Firebase API 키가 설정되지 않았습니다."

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)
        data = response.json()

        if response.status_code == 200 and "idToken" in data:
            return True, "✅ 로그인 성공"
        else:
            error_msg = data.get("error", {}).get("message", "❌ 로그인 실패")
            return False, error_msg
    except Exception as e:
        return False, f"❌ 오류 발생: {e}"
