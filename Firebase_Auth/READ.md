.gitignore   --> firebase_key.json 변경후 사용
    발급 받아 사용 (개인용)

    
#####################################
1. Firebase에 회원가입하기
    - https://console.firebase.google.com 접속

    - Google 계정으로 로그인

2. 새 프로젝트 만들기
    - "프로젝트 만들기" 클릭

    - 프로젝트 이름은 자유롭게 (MyApp, FocusMateApp 등)

    - Google Analytics는 “사용 안 함” 선택하고 "만들기"

3. 이메일/비밀번호 로그인 기능 켜기
    - 프로젝트 왼쪽 메뉴에서 Authentication 클릭

    - 상단 탭에서 “로그인 방법” 클릭

    - “이메일/비밀번호” 항목 오른쪽 스위치 클릭 → “사용 설정” 후 저장

4. Firebase 연결용 🔑 비밀키 만들기
    -   왼쪽 메뉴에서 “⚙️ 프로젝트 설정” 클릭

    -   상단 탭 중에 “서비스 계정” 클릭

    - "새 비공개 키 생성" 버튼 클릭

    - 팝업이 뜨면 "확인" → .json 파일이 다운로드됨

    - 이 파일의 이름을 firebase_key.json 으로 바꾸고

    - 이 파일을 main.py가 있는 폴더에 넣기

    예시:firebase_key.json