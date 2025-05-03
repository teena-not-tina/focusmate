/**
 * FocusMate - AI 학습 어시스턴트
 * 주요 기능:
 * - 웹캠 접근 및 비디오 스트림 표시
 * - 학습 모니터링 및 타이머
 * - 사진 캡처 및 감정 분석
 * - AI 학습 어시스턴트와의 대화 기능
 */

// 전역 변수 정의
let videoStream = null;
let isRecording = false;
let recordedChunks = [];
let mediaRecorder = null;
let sessionId = generateSessionId();
let chatMessages = [];
let timerInterval = null;
let timerSeconds = 0;
let dropdown = null;

// DOM이 완전히 로드된 후 실행
document.addEventListener('DOMContentLoaded', function () {
    // 인증 페이지에서는 특정 기능 비활성화
    if (document.body.classList.contains('auth-page')) {
        // 인증 페이지에서는 헤더 관련 함수 실행하지 않음
        return;
    }
    
    // 기존 코드는 그대로 유지
    // 헤더 초기화
    initializeHeader();

    // 화면 크기에 따라 레이아웃 설정
    setupResponsiveLayout();

    // 상태 컨테이너 업데이트
    updateStatusContainer();

    // 화면 크기 변경 시 레이아웃 재설정
    window.addEventListener('resize', function () {
        setupResponsiveLayout();
        initializeHeader();
    });

    // 요소 참조
    const videoContainer = document.querySelector('.video-container');
    const controlBtns = document.querySelectorAll('.control-btn');
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.send-btn');
    const tabItems = document.querySelectorAll('.tab-item');

    // 헤더 드롭다운 생성
    createHeaderDropdown();

    // 비디오 요소 생성 (없는 경우)
    if (videoContainer && !videoContainer.querySelector('video')) {
        const videoElement = document.createElement('video');
        videoElement.setAttribute('autoplay', true);
        videoElement.setAttribute('muted', true);
        videoElement.classList.add('webcam-feed');
        videoContainer.appendChild(videoElement);
    }

    // 초기 상태 설정
    updateStatusIndicator('비활성화', false);

    // 시작 시 AI 메시지 표시
    displayChatMessage('안녕하세요! FocusMate AI 학습 어시스턴트입니다.<br>학습에 관한 질문이나도움이 필요한 내용이 있으면 언제든지 물어보세요.', 'assistant');

    // 타이머 초기화
    initializeTimer();

    // 탭 전환 기능
    initializeTabs(tabItems);

    // 컨트롤 버튼 이벤트 설정
    if (controlBtns && controlBtns.length > 0) {
        controlBtns.forEach((btn, index) => {
            btn.addEventListener('click', () => {
                const actions = [
                    toggleWebcam,           // 카메라 켜기/끄기
                    captureImage,           // 사진 찍기
                    toggleTimer,            // 타이머 설정
                    downloadRecording,      // 녹화 다운로드
                    stopRecording           // 종료
                ];

                if (index < actions.length) {
                    actions[index]();
                }
            });
        });
    }

    // 메시지 전송 기능
    initializeChatFunctions(chatInput, sendBtn);

    // 메뉴 토글 이벤트 초기화
    initializeMenuToggle();

    // 서버에서 전달받은 로그인 상태를 활용
    // data-* 속성을 통해 서버에서 전달한 값을 가져옵니다
    const bodyElement = document.body;
    const isLoggedIn = bodyElement.dataset.loggedIn === 'true';
    const username = bodyElement.dataset.username || '';
    
    updateDropdownMenu(isLoggedIn, username);

    // 테마 설정 불러오기
    loadThemePreference();
    
    // 테마 메뉴 아이템 업데이트
    setTimeout(updateThemeMenuItem, 100);
}, { once: true });

/**
 * 헤더 드롭다운 메뉴 생성
 */
function createHeaderDropdown() {
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.className = 'header-dropdown';
        dropdown.id = 'headerDropdown';
        document.body.appendChild(dropdown);
    }
}

/**
 * 로그인 상태에 따라 드롭다운 메뉴 업데이트
 */
function updateDropdownMenu(isLoggedIn, username) {
    const headerDropdown = document.getElementById('headerDropdown');
    if (!headerDropdown) return;
    
    // 드롭다운 초기화
    headerDropdown.innerHTML = '';
    
    if (isLoggedIn) {
        // 로그인된 상태일 때
        const userInfo = document.createElement('div');
        userInfo.className = 'dropdown-user-info';
        userInfo.innerHTML = `<span class="material-icons">account_circle</span> ${username || '사용자'}`;
        headerDropdown.appendChild(userInfo);
        
        const logoutItem = document.createElement('a');
        logoutItem.href = 'logout';
        logoutItem.className = 'dropdown-item';
        logoutItem.innerHTML = '<span class="material-icons">logout</span> 로그아웃';
        headerDropdown.appendChild(logoutItem);
        
        const divider = document.createElement('div');
        divider.className = 'dropdown-divider';
        headerDropdown.appendChild(divider);
    } else {
        // 로그인되지 않은 상태일 때
        const loginItem = document.createElement('a');
        loginItem.href = '/auth/login'; // 로그인 URL 수정
        loginItem.className = 'dropdown-item';
        loginItem.innerHTML = '<span class="material-icons">login</span> 로그인';
        headerDropdown.appendChild(loginItem);
        
        const signupItem = document.createElement('a');
        signupItem.href = '/auth/signup'; // 회원가입 URL 수정
        signupItem.className = 'dropdown-item';
        signupItem.innerHTML = '<span class="material-icons">person_add</span> 회원가입';
        headerDropdown.appendChild(signupItem);
        
        const divider = document.createElement('div');
        divider.className = 'dropdown-divider';
        headerDropdown.appendChild(divider);
    }
    
    // 현재 테마 확인
    const isDarkTheme = document.body.classList.contains('dark-theme');
    
    // 테마 전환 메뉴 항목
    const themeItem = document.createElement('a');
    themeItem.href = '/auth/theme'; // 테마 URL 수정
    themeItem.className = 'dropdown-item';
    themeItem.innerHTML = isDarkTheme 
        ? '<span class="material-icons">light_mode</span> 라이트 모드' 
        : '<span class="material-icons">dark_mode</span> 다크 모드';
    headerDropdown.appendChild(themeItem);
    
    // 설정 메뉴 항목
    const settingItem = document.createElement('a');
    settingItem.href = 'setting';
    settingItem.className = 'dropdown-item';
    settingItem.innerHTML = '<span class="material-icons">settings</span> 설정';
    headerDropdown.appendChild(settingItem);
}

/**
 * 테마 전환 함수
 */
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    // data-theme 속성 변경
    html.setAttribute('data-theme', newTheme);
    
    // body 클래스도 함께 변경
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${newTheme}-theme`);
    
    // 로컬 스토리지에 저장
    localStorage.setItem('theme', newTheme);
    
    // 메뉴 아이템 업데이트
    updateThemeMenuItem();
}

/**
 * 테마 메뉴 아이템 업데이트
 */
function updateThemeMenuItem() {
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const themeItems = document.querySelectorAll('a[href="theme"]');
    
    themeItems.forEach(item => {
        item.innerHTML = isDarkTheme 
            ? '<span class="material-icons">light_mode</span> 라이트 모드' 
            : '<span class="material-icons">dark_mode</span> 다크 모드';
    });
}

/**
 * 로컬 스토리지에서 테마 불러오기
 */
function loadThemePreference() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    // data-theme 속성 설정
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // body 클래스 설정
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${savedTheme}-theme`);
    
    console.log(`${savedTheme} 테마 로드됨`);
}

/**
 * 헤더 및 대시보드 메뉴 초기화
 */
function initializeHeader() {
    // 메인 컨테이너 상단 여백 조정
    const mainContainer = document.querySelector('.main-container');
    if (mainContainer) {
        mainContainer.style.marginTop = '70px';
    }
}

/**
 * 메뉴 토글 버튼 이벤트 - 수정
 */
function initializeMenuToggle() {
    const toggleBtn = document.querySelector('.menu-toggle-button');
    const headerDropdown = document.getElementById('headerDropdown');

    if (!toggleBtn || !headerDropdown) return;

    toggleBtn.addEventListener('click', function (e) {
        headerDropdown.classList.toggle('active');
        e.stopPropagation();
    });

    // 바디 클릭 시 드롭다운 닫기
    document.body.addEventListener('click', function (e) {
        if (!headerDropdown.contains(e.target) && !toggleBtn.contains(e.target)) {
            headerDropdown.classList.remove('active');
        }
    });

    // 이전 클릭 이벤트 핸들러 제거를 위해 이벤트 위임 방식으로 수정
    // document에 이미 click 이벤트가 있을 수 있으므로 여기서는 headerDropdown에만 등록
    headerDropdown.addEventListener('click', function(e) {
        const themeLink = e.target.closest('a[href="theme"]');
        if (themeLink) {
            e.preventDefault(); // 기본 동작 방지
            e.stopPropagation(); // 이벤트 버블링 방지
            
            // 클라이언트 측에서 테마 즉시 전환
            toggleTheme();
            
            // 서버에도 테마 변경 요청 (백그라운드에서 처리)
            fetch('/theme', { method: 'GET' })
                .then(response => console.log('테마 저장 완료'))
                .catch(err => console.error('테마 저장 오류:', err));
            
            // 드롭다운 닫기
            headerDropdown.classList.remove('active');
        }
    });
}

/**
 * 반응형 레이아웃 설정
 */
function setupResponsiveLayout() {
    const isWideScreen = window.innerWidth >= 1024;

    if (isWideScreen) {
        setupFullscreenLayout();
    } else {
        setupWindowLayout();
    }
}

/**
 * 전체 화면 모드 레이아웃 설정
 */
function setupFullscreenLayout() {
    // 콘텐츠 영역을 그리드로 설정
    const content = document.querySelector('.content');
    if (!content) return;

    // 기본 스타일 설정
    content.style.display = 'grid';
    content.style.gridTemplateColumns = '1fr 380px';
    content.style.gap = '20px';
    content.style.padding = '20px';
    content.style.height = 'calc(100vh - 70px)'; // 헤더 높이 제외

    // 비디오와 컨트롤을 포함할 왼쪽 섹션
    let videoSection = document.querySelector('.video-section');
    if (!videoSection) {
        videoSection = document.createElement('div');
        videoSection.className = 'video-section';

        // 비디오 컨테이너와 컨트롤을 비디오 섹션으로 이동
        const videoContainer = document.querySelector('.video-container');
        const statusContainer = document.querySelector('.status-container');
        const videoControls = document.querySelector('.video-controls');

        if (statusContainer) videoSection.appendChild(statusContainer);
        if (videoContainer) videoSection.appendChild(videoContainer);
        if (videoControls) videoSection.appendChild(videoControls);

        content.prepend(videoSection);
    }

    // 챗봇 섹션 (오른쪽)
    let assistantSection = document.querySelector('.assistant-section');
    if (!assistantSection) {
        assistantSection = document.createElement('div');
        assistantSection.className = 'assistant-section';

        // 챗봇 관련 요소들을 어시스턴트 섹션으로 이동
        const tabNavigation = document.querySelector('.tab-navigation');
        const tabContentWrapper = document.querySelector('.tab-content-wrapper');

        if (tabNavigation) assistantSection.appendChild(tabNavigation);
        if (tabContentWrapper) assistantSection.appendChild(tabContentWrapper);

        content.appendChild(assistantSection);
    }

    // 강제로 높이 맞추기
    // 메인 콘텐츠 영역의 높이에서 약간의 여백을 뺀 값
    const availableHeight = content.clientHeight - 40; // 상하 여백 40px 고려

    // 비디오와 어시스턴트 섹션에 동일한 높이 적용
    videoSection.style.height = `${availableHeight}px`;
    assistantSection.style.height = `${availableHeight}px`;

    // 비디오 컨테이너 설정
    const videoContainer = videoSection.querySelector('.video-container');
    if (videoContainer) {
        // 비디오 컨테이너는 상태 컨테이너와 컨트롤 버튼 높이를 제외한 나머지 공간 차지
        const statusContainer = videoSection.querySelector('.status-container');
        const videoControls = videoSection.querySelector('.video-controls');

        let statusHeight = statusContainer ? statusContainer.offsetHeight : 0;
        let controlsHeight = videoControls ? videoControls.offsetHeight : 0;

        // 마진 고려 (상태 컨테이너와 비디오 컨트롤 사이 마진)
        const marginHeight = 20; // 상하 마진 합계 추정치

        // 비디오 컨테이너 높이 계산
        const videoHeight = availableHeight - statusHeight - controlsHeight - marginHeight;

        // 기존 패딩 방식 대신 고정 높이 설정
        videoContainer.style.height = `${videoHeight}px`;
        videoContainer.style.paddingBottom = '0';
        videoContainer.style.position = 'relative'; // 추가: 위치 상대적으로 설정
        
        // 비디오 요소 스타일 설정
        const videoElement = videoContainer.querySelector('video');
        if (videoElement) {
            videoElement.style.position = 'absolute';
            videoElement.style.width = '100%';
            videoElement.style.height = '100%';
            videoElement.style.objectFit = 'cover';
            videoElement.style.left = '0';
            videoElement.style.top = '0';
        }
    }
}

/**
 * 창 모드 레이아웃 설정
 */
function setupWindowLayout() {
    // 콘텐츠 영역을 기본 레이아웃으로 되돌림
    const content = document.querySelector('.content');
    if (!content) return;

    content.style.display = 'block';
    content.style.gridTemplateColumns = '';
    content.style.gap = '';
    content.style.padding = '10px';
    content.style.height = '';

    // 비디오 섹션이 있다면 요소들을 다시 콘텐츠로 이동
    const videoSection = document.querySelector('.video-section');
    if (videoSection) {
        const parent = videoSection.parentNode;
        while (videoSection.firstChild) {
            parent.insertBefore(videoSection.firstChild, videoSection);
        }
        parent.removeChild(videoSection);
    }

    // 어시스턴트 섹션이 있다면 요소들을 다시 콘텐츠로 이동
    const assistantSection = document.querySelector('.assistant-section');
    if (assistantSection) {
        const parent = assistantSection.parentNode;
        while (assistantSection.firstChild) {
            parent.insertBefore(assistantSection.firstChild, assistantSection);
        }
        parent.removeChild(assistantSection);
    }

    // 비디오 컨테이너 크기 조정
    const videoContainer = document.querySelector('.video-container');
    if (videoContainer) {
        videoContainer.style.height = 'auto';
        videoContainer.style.paddingBottom = '56.25%'; // 16:9 비율
    }
}

/**
 * 상태 표시기 업데이트
 */
function updateStatusIndicator(status, active) {
    // 모든 상태 표시기 점과 텍스트 가져오기
    const statusDots = document.querySelectorAll('.status-dot');
    const statusTexts = document.querySelectorAll('.status-text');

    // 모든 상태 표시기 업데이트
    statusDots.forEach(dot => {
        dot.style.backgroundColor = active ? '#ef4444' : '#6b7280';
    });

    statusTexts.forEach(text => {
        text.textContent = `학습 모니터링 ${status}`;
        text.style.color = active ? '#ef4444' : '#6b7280';
    });

    // 타이머 제어
    if (active) {
        startTimer();
    } else {
        stopTimer();
    }
}

/**
 * 상태 컨테이너 업데이트 함수
 */
function updateStatusContainer() {
    const statusContainer = document.querySelector('.status-container');
    if (!statusContainer) return;

    // 기존 내용 지우고 새로 구성
    statusContainer.innerHTML = `
        <div class="status-indicator">
            <div class="status-dot"></div>
            <div class="status-text">학습 모니터링 비활성화</div>
        </div>
        <div class="timer-display">00:00:00</div>
    `;

    // 타이머 참조 업데이트
    const timerEl = statusContainer.querySelector('.timer-display');
    if (timerEl && timerInterval) {
        // 타이머가 활성화된 상태면 표시 업데이트
        const hrs = Math.floor(timerSeconds / 3600);
        const mins = Math.floor((timerSeconds % 3600) / 60);
        const secs = timerSeconds % 60;
        timerEl.textContent = `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
}

/**
 * 타이머 시작
 */
function startTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
    }

    const timerEl = document.querySelector('.timer-display');
    if (!timerEl) return;

    timerInterval = setInterval(() => {
        timerSeconds++;
        const hrs = Math.floor(timerSeconds / 3600);
        const mins = Math.floor((timerSeconds % 3600) / 60);
        const secs = timerSeconds % 60;

        timerEl.textContent = `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }, 1000);
}

/**
 * 타이머 중지
 */
function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

/**
 * 타이머 초기화
 */
function initializeTimer() {
    const timerEl = document.querySelector('.timer-display');
    if (timerEl) {
        timerEl.textContent = '00:00:00';
        timerSeconds = 0;
    }
}

/**
 * 탭 전환 기능 초기화
 */
function initializeTabs(tabItems) {
    if (!tabItems || tabItems.length === 0) return;

    tabItems.forEach(item => {
        item.addEventListener('click', () => {
            // 모든 탭 비활성화
            tabItems.forEach(tab => tab.classList.remove('active'));
            item.classList.add('active');

            // 모든 컨텐츠 숨김
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // 해당 컨텐츠 표시
            const tabName = item.getAttribute('data-tab');
            document.getElementById(tabName + 'Tab')?.classList.add('active');
        });
    });
}

/**
 * 웹캠 토글 함수
 */
function toggleWebcam() {
    const videoElement = document.querySelector('.webcam-feed');
    const webcamBtn = document.querySelector('.control-btn:first-child .material-icons');

    if (!videoElement || !webcamBtn) return;

    if (videoStream) {
        // 웹캠 중지
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        videoElement.srcObject = null;
        webcamBtn.textContent = 'videocam';
        updateStatusIndicator('비활성화', false);
    } else {
        // 웹캠 컨테이너 보이기 확인
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.style.display = 'block';
            videoContainer.style.position = 'relative';
            videoContainer.style.overflow = 'hidden';
        }
        
        // 웹캠 시작
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(stream => {
                videoStream = stream;
                videoElement.srcObject = stream;
                videoElement.style.display = 'block';
                videoElement.style.width = '100%';
                videoElement.style.height = '100%';
                videoElement.style.objectFit = 'cover';
                webcamBtn.textContent = 'videocam_off';
                updateStatusIndicator('활성화', true);

                // 30초마다 자동으로 사진 캡처하여 감정 분석
                startPeriodicEmotionDetection();
            })
            .catch(err => {
                console.error('웹캠 접근 오류:', err);
                alert('웹캠에 접근할 수 없습니다. 권한을 확인해주세요.');
            });
    }
}

/**
 * 주기적 감정 감지 시작
 */
function startPeriodicEmotionDetection() {
    if (!videoStream) return;

    // 30초마다 감정 분석
    const detectionInterval = setInterval(() => {
        if (!videoStream) {
            clearInterval(detectionInterval);
            return;
        }
        captureAndAnalyze();
    }, 30000);

    // 첫 번째 감정 분석 즉시 실행
    captureAndAnalyze();
}

/**
 * 이미지 캡처 함수
 */
function captureImage() {
    if (!videoStream) {
        alert('먼저 카메라를 활성화해주세요.');
        return;
    }

    captureAndAnalyze(true); // true = 결과 표시
}

/**
 * 이미지 캡처 및 분석 함수
 */
function captureAndAnalyze(showResult = false) {
    const videoElement = document.querySelector('.webcam-feed');
    if (!videoElement) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);

    canvas.toBlob(blob => {
        // 폼 데이터 생성
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        formData.append('type', 'image');

        // 서버로 데이터 전송
        fetch('/detect_emotion', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('감정 분석 오류:', data.error);
                    if (showResult) alert(`감정 분석 중 오류가 발생했습니다: ${data.error}`);
                    return;
                }

                // 감정 상태에 따른 UI 업데이트
                updateStudyStateUI(data);

                if (showResult) {
                    // 사용자에게 현재 감정 상태 알림
                    let emotionMessage = `현재 감지된 상태: ${translateEmotion(data.dominant_emotion)}`;
                    displayChatMessage(emotionMessage, 'system');

                    // AI 학습 어시스턴트의 감정 상태 기반 피드백
                    displayChatMessage(data.response, 'assistant');
                }
            })
            .catch(err => {
                console.error('서버 통신 오류:', err);
                if (showResult) alert('서버와 통신 중 오류가 발생했습니다.');
            });
    }, 'image/jpeg', 0.9);
}

/**
 * 감정 한글 변환
 */
function translateEmotion(emotion) {
    const emotionMap = {
        'focused': '집중',
        'interested': '흥미',
        'anxious': '불안',
        'tired': '피로',
        'neutral': '중립'
    };

    return emotionMap[emotion] || emotion;
}

/**
 * 학습 상태 UI 업데이트
 */
function updateStudyStateUI(data) {
    // 감정 상태에 따른 UI 색상/스타일 변경
    const emotion = data.dominant_emotion;
    let color;

    switch (emotion) {
        case 'focused':
            color = '#10b981'; // 집중 - 녹색
            break;
        case 'interested':
            color = '#3b82f6'; // 흥미 - 파란색
            break;
        case 'anxious':
            color = '#f59e0b'; // 불안 - 주황색
            break;
        case 'tired':
            color = '#ef4444'; // 피로 - 빨간색
            break;
        default:
            color = '#6b7280'; // 중립 - 회색
    }

    // 상태 표시기 색상 변경
    const statusDot = document.querySelector('.status-dot');
    if (statusDot) {
        statusDot.style.backgroundColor = color;
    }
}

/**
 * 타이머 토글 기능
 */
function toggleTimer() {
    alert('타이머 기능을 설정합니다.');
}

/**
 * 녹화 기능 - 다운로드
 */
function downloadRecording() {
    if (!isRecording && recordedChunks.length === 0) {
        alert('다운로드할 녹화 영상이 없습니다.');
        return;
    }

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `focusmate_recording_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.webm`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }, 100);
}

/**
 * 녹화 종료
 */
function stopRecording() {
    if (!videoStream) {
        alert('활성화된 웹캠이 없습니다.');
        return;
    }

    if (isRecording && mediaRecorder) {
        mediaRecorder.stop();
        isRecording = false;
    } else {
        alert('학습 모니터링을 종료합니다.');
        // 웹캠 종료
        toggleWebcam();
    }
}

/**
 * 채팅 기능 초기화
 */
function initializeChatFunctions(chatInput, sendBtn) {
    if (!chatInput || !sendBtn) return;

    function sendMessage() {
        const message = chatInput.value.trim();
        if (message) {
            // 사용자 메시지 표시
            displayChatMessage(message, 'user');

            // API 요청
            sendChatToApi(message);

            // 입력창 초기화
            chatInput.value = '';
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

/**
 * 수학 기호 변환 함수 - 수식 표현식에만 적용
 */
function convertMathSymbols(text) {
    // 정규식을 사용하여 수식 표현식만 식별
    // 숫자와 기호 사이에 있는 연산자를 감지하여 변환
    
    // 숫자 + / + 숫자 패턴 변환
    text = text.replace(/(\d+)\s*\/\s*(\d+)/g, '$1 ÷ $2');
    
    // 숫자 + * + 숫자 패턴 변환
    text = text.replace(/(\d+)\s*\*\s*(\d+)/g, '$1 × $2');
    
    // 괄호 안의 수식 변환 (예: (3*4))
    text = text.replace(/\((\d+)\s*\/\s*(\d+)\)/g, '($1 ÷ $2)');
    text = text.replace(/\((\d+)\s*\*\s*(\d+)\)/g, '($1 × $2)');
    
    // 변수와 함께 사용되는 경우 (예: x*y, a/b)
    text = text.replace(/([a-zA-Z])\s*\/\s*([a-zA-Z])/g, '$1 ÷ $2');
    text = text.replace(/([a-zA-Z])\s*\*\s*([a-zA-Z])/g, '$1 × $2');
    text = text.replace(/(\d+)\s*\/\s*([a-zA-Z])/g, '$1 ÷ $2');
    text = text.replace(/([a-zA-Z])\s*\/\s*(\d+)/g, '$1 ÷ $2');
    text = text.replace(/(\d+)\s*\*\s*([a-zA-Z])/g, '$1 × $2');
    text = text.replace(/([a-zA-Z])\s*\*\s*(\d+)/g, '$1 × $2');
    
    return text;
}

/**
 * 채팅 메시지 표시
 */
function displayChatMessage(message, sender) {
    // 메시지 저장 전에 수학 기호 변환 (AI 어시스턴트의 경우에만)
    let processedMessage = message;
    
    if (sender === 'assistant') {
        processedMessage = convertMathSymbols(message);
    }
    
    // 메시지 저장
    chatMessages.push({ sender, content: processedMessage });

    // 컨테이너에 추가
    const assistantContainer = document.querySelector('.assistant-container');
    if (!assistantContainer) return;

    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    let messageContent = '';
    if (sender === 'user') {
        messageContent = `<div class="message-bubble user">
                            <div class="message-text">${processedMessage}</div>
                          </div>`;
    } else if (sender === 'assistant') {
        messageContent = `<div class="message-bubble assistant">
                            <div class="message-text">${processedMessage}</div>
                          </div>`;
    } else {
        messageContent = `<div class="message-bubble system">
                            <div class="message-text">${processedMessage}</div>
                          </div>`;
    }

    messageElement.innerHTML = messageContent;

    // 메시지 추가
    assistantContainer.appendChild(messageElement);

    // 스크롤 자동 이동
    assistantContainer.scrollTop = assistantContainer.scrollHeight;
}

/**
 * API에 채팅 전송
 */
function sendChatToApi(message) {
    // 타이핑 인디케이터를 FocusMate AI 응답창에 표시
    const typingId = Date.now();
    displayChatMessage(`<span id="typing-${typingId}">메시지를 생성하는 중입니다<span class="typing-dots">...</span></span>`, 'assistant');
    startTypingAnimation(`typing-${typingId}`);

    // 이전 메시지 기록 가져오기 (HTML 태그 제거 및 형식 조정)
    const recentMessages = chatMessages
        .filter(msg => msg.sender !== 'system') // 시스템 메시지 제외
        .slice(-10)
        .map(msg => {
            const plainContent = msg.content.replace(/<[^>]*>/g, ''); // HTML 태그 제거
            return {
                role: msg.sender === 'user' ? 'user' : 'assistant',
                content: plainContent
            };
        });

    // 시스템 프롬프트 - 명확하고 상세하게
    const systemPrompt = {
        role: 'system',
        content: 
        `당신은 학생들의 학습을 돕는 FocusMate AI 어시스턴트입니다. 다음 원칙을 철저히 따르세요:

            1. 항상 정확하게 응답하세요.
            2. 질문의 복잡성에 비례하여 답변 길이를 조정하세요:
            - 간단한 인사나 일상 대화는 매우 짧은 문장으로 간결하게 응답
            - 학습 관련 질문은 충분히 자세하게 설명
            - "안녕", "고마워" 같은 인사에는 매우 짧게 답변

            3. 존댓말을 사용하되, 친절하고 공감적인 어조를 유지하세요.
            4. 중요 개념이나 핵심 용어는 **볼드체**로 강조하세요.
            5. 단계적 설명이 필요한 경우 번호나 글머리 기호를 사용하여 구조화하세요.
            6. 불필요한 인사말이나 결론 문구("추가 질문 있으신가요?" 등)는 생략하세요.
            7. 학습 동기 부여와 효율성 향상을 위한 실질적인 조언을 제공하세요.
            8. 질문과 직접 관련된 내용만 답변하고, 관련 없는 정보는 제공하지 마세요.

            이 지침을 반드시 준수하여 학습자에게 최적화된 도움을 제공하세요.`
    };

    // API 요청 객체 구성
    const requestData = {
        messages: [
            systemPrompt,
            ...recentMessages,
            { role: 'user', content: message }
        ],
        session_id: sessionId,
        temperature: 0.3,  // 매우 낮은 온도로 설정 (일관성 극대화)
        max_tokens: 1500    // 최대 토큰 수 늘림
    };

    // API 요청
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Request-With': 'XMLHttpRequest'
        },
        body: JSON.stringify(requestData)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // 타이핑 중 메시지 제거
            const typingMessage = document.querySelector(`#typing-${typingId}`);
            if (typingMessage) {
                const messageElement = typingMessage.closest('.message');
                if (messageElement) messageElement.remove();
            }

            if (data.error) {
                displayChatMessage(`오류가 발생했습니다: ${data.error}`, 'system');
            } else {
                // 응답 표시 (마크다운 변환 적용)
                const formattedResponse = markdownToHtml(data.message || "응답이 없습니다.");
                displayChatMessage(formattedResponse, 'assistant');
            }
        })
        .catch(err => {
            // 타이핑 중 메시지 제거
            const typingMessage = document.querySelector(`#typing-${typingId}`);
            if (typingMessage) {
                const messageElement = typingMessage.closest('.message');
                if (messageElement) messageElement.remove();
            }

            console.error('API 오류:', err);
            displayChatMessage('서버와 통신 중 오류가 발생했습니다.', 'system');
        });
}

/**
 * 타이핑 애니메이션 추가
 */
function startTypingAnimation(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const dotsElement = element.querySelector('.typing-dots');
    if (!dotsElement) return;

    let count = 0;
    const animationInterval = setInterval(() => {
        count = (count + 1) % 4;
        let dots = '';
        for (let i = 0; i < count; i++) {
            dots += '.';
        }
        dotsElement.textContent = dots;
    }, 400);

    // ID 저장해서 나중에 제거할 수 있게
    element.dataset.intervalId = animationInterval;
}

/**
 * 마크다운 형식을 HTML로 변환
 */
function markdownToHtml(markdown) {
    if (!markdown) return '';

    // HTML 특수 문자 이스케이프
    let html = markdown
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // 코드 블록 (```) 처리 - 언어 지원 추가
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (match, lang, code) {
        return `<pre class="code-block${lang ? ' language-' + lang : ''}"><code>${code.trim()}</code></pre>`;
    });

    // 볼드 처리 (**텍스트** 또는 __텍스트__)
    html = html.replace(/\*\*(.*?)\*\*|__(.*?)__/g, function (match, g1, g2) {
        return `<strong>${g1 || g2}</strong>`;
    });

    // 이탤릭체 (*텍스트* 또는 _텍스트_)
    html = html.replace(/\*([^\*]+)\*|_([^_]+)_/g, function (match, g1, g2) {
        return `<em>${g1 || g2}</em>`;
    });

    // 인라인 코드 (`코드`)
    html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // 헤더 처리 (### 제목)
    html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

    // 순서 없는 목록 처리
    html = html.replace(/^\s*[\*\-]\s+(.*?)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>\n)+/gs, function (match) {
        return '<ul>' + match + '</ul>';
    });

    // 순서 있는 목록 처리
    html = html.replace(/^\s*(\d+)\.\s+(.*?)$/gm, '<li>$2</li>');
    html = html.replace(/(<li>.*?<\/li>\n)+/gs, function (match) {
        return '<ol>' + match + '</ol>';
    });

    // 인용구 처리
    html = html.replace(/^>\s+(.*?)$/gm, '<blockquote>$1</blockquote>');

    // 줄바꿈 처리 (단락 처리 개선)
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // 최종 단락 래핑
    if (!html.startsWith('<pre') && !html.startsWith('<ul') &&
        !html.startsWith('<ol') && !html.startsWith('<h')) {
        html = '<p>' + html + '</p>';
    }

    return html;
}

/**
 * 고유 세션 ID 생성
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// 탭 전환 시 이전 이벤트 핸들러 제거 및 새 이벤트 핸들러 등록
document.addEventListener('DOMContentLoaded', function () {
    // 기존 코드는 그대로 실행...

    // 탭 전환 확실히 하기
    const tabItems = document.querySelectorAll('.tab-item');

    // 이전에 등록된 이벤트 리스너 제거
    tabItems.forEach(item => {
        const newItem = item.cloneNode(true);
        item.parentNode.replaceChild(newItem, item);
    });

    // 새 이벤트 리스너 등록
    document.querySelectorAll('.tab-item').forEach(item => {
        item.addEventListener('click', function () {
            // 모든 탭 비활성화
            document.querySelectorAll('.tab-item').forEach(tab => {
                tab.classList.remove('active');
            });

            // 클릭한 탭 활성화
            this.classList.add('active');

            // 모든 탭 컨텐츠 비활성화
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });

            // 선택한 탭 컨텐츠만 활성화
            const tabId = this.getAttribute('data-tab') + 'Tab';
            const selectedContent = document.getElementById(tabId);
            if (selectedContent) {
                selectedContent.classList.add('active');
                selectedContent.style.display = 'flex';
                selectedContent.style.flexDirection = 'column';
                selectedContent.style.height = '100%';
            }
        });
    });

    // 창 크기가 변경될 때 레이아웃 조정
    window.addEventListener('resize', function () {
        if (window.innerWidth >= 1024) {
            setupFullscreenLayout();
        }
    });
});

// 창 크기가 변경될 때 레이아웃 조정
window.addEventListener('resize', function () {
    if (window.innerWidth >= 1024) {
        setupFullscreenLayout();
    }
});

/**
 * auth.js 파일에서 로그인 및 회원가입 URL 수정
 */
function handleSignup(e) {
    // ...
    setTimeout(() => {
        window.location.href = '/auth/login'; // 변경된 URL 경로
    }, 1500);
    // ...
}