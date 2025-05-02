/**
 * FocusMate - AI 학습 조교
 * 주요 기능:
 * - 웹캠 접근 및 비디오 스트림 표시
 * - 학습 모니터링 및 타이머
 * - 사진 캡처 및 감정 분석
 * - AI 조교와의 대화 기능
 */

// 전역 변수 정의
let videoStream = null;
let isRecording = false;
let recordedChunks = [];
let mediaRecorder = null;
let sessionId = generateSessionId();
let chatMessages = [];

// 상태 관련 변수
let monitoringActive = false;

/**
 * 상태 표시기를 타이머 왼쪽으로 이동시키는 함수
 */
function moveStatusIndicator() {
    // 상태 표시기와 타이머 요소 가져오기
    const statusIndicator = document.querySelector('.status-indicator');
    const monitorTitle = document.querySelector('.monitor-title');
    
    if (statusIndicator && monitorTitle) {
        // 타이머 요소
        const timerEl = monitorTitle.querySelector('.monitor-timer');
        
        // 현재 내용 비우기
        monitorTitle.innerHTML = '';
        
        // 상태 표시기 스타일 조정
        statusIndicator.style.marginRight = '15px';
        
        // 요소들 추가
        monitorTitle.appendChild(statusIndicator.cloneNode(true));
        if (timerEl) monitorTitle.appendChild(timerEl);
    }
}

/**
 * 상태 표시기 업데이트
 */
function updateStatusIndicator(status, active) {
    // 모든 상태 표시기 점과 텍스트 가져오기 (타이머 옆으로 이동한 것 포함)
    const statusDots = document.querySelectorAll('.status-dot');
    const statusTexts = document.querySelectorAll('.status-indicator span:nth-child(2)');
    
    // 모든 상태 표시기 업데이트
    statusDots.forEach(dot => {
        dot.style.backgroundColor = active ? '#ef4444' : '#6b7280';
    });
    
    statusTexts.forEach(text => {
        text.innerHTML = `<strong>학습 모니터링</strong> <strong>${status}</strong>`;
        text.style.color = active ? '#ef4444' : '#6b7280';
    });
    
    // 타이머 제어
    if (active) {
        startTimer();
    } else {
        stopTimer();
    }
}

// 타이머 옆 학습 모니터링 제목 볼드 처리 함수
function makeMonitorTitleBold() {
    const monitorTitle = document.querySelector('.monitor-title');
    if (monitorTitle) {
        // 현재 내용 가져오기
        const content = monitorTitle.innerHTML;
        
        // "학습 모니터링" 텍스트를 찾아서 볼드 처리
        const boldContent = content.replace(
            /학습\s*모니터링/g, 
            '<strong>학습 모니터링</strong>'
        );
        
        // 변경된 내용 적용
        monitorTitle.innerHTML = boldContent;
    }
}

// DOM이 완전히 로드된 후 실행
document.addEventListener('DOMContentLoaded', function() {
    // 요소 참조
    const videoContainer = document.querySelector('.video-container');
    const controlBtns = document.querySelectorAll('.control-btn');
    const analysisBtn = document.querySelector('.analysis-btn');
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.send-btn');
    const tabItems = document.querySelectorAll('.tab-item');
    const assistantInfo = document.querySelector('.assistant-info');
    
    // 상태 표시기를 타이머 왼쪽으로 이동
    moveStatusIndicator();
    
    // 비디오 요소 생성
    const videoElement = document.createElement('video');
    videoElement.setAttribute('autoplay', true);
    videoElement.setAttribute('muted', true);
    videoElement.classList.add('webcam-feed');
    videoContainer.appendChild(videoElement);
    
    // 초기 상태 설정 - 비활성화로 설정
    updateStatusIndicator('비활성화', false);

    // 시작 시 AI 메시지 표시
    displayChatMessage('안녕하세요! FocusMate AI 학습 조교입니다. 학습에 관한 질문이나 도움이 필요한 내용이 있으면 언제든지 물어보세요.', 'assistant');

    // 타이머 기능
    initializeTimer();

    // 탭 전환 기능
    initializeTabs(tabItems);

    // 컨트롤 버튼 이벤트 설정
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

    // 학습 상태 분석 버튼
    analysisBtn.addEventListener('click', analyzeStudyState);

    // 메시지 전송 기능
    initializeChatFunctions(chatInput, sendBtn);
});

/**
 * 웹캠 상태에 따른 UI 업데이트 및 타이머 제어 기능
 * - 웹캠 비활성화: 회색 원, "학습 모니터링 비활성화" 텍스트, 타이머 중지
 * - 웹캠 활성화: 빨간 원, "학습 모니터링 활성화" 텍스트, 타이머 동작
 */

// 타이머 관련 변수
let timerInterval = null;
let timerSeconds = 0;

/**
 * 타이머 시작
 */
function startTimer() {
    // 이미 실행 중인 타이머가 있으면 중지
    if (timerInterval) {
        clearInterval(timerInterval);
    }
    
    const timerEl = document.querySelector('.monitor-timer');
    
    // 1초마다 타이머 업데이트
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
    
    // 타이머 초기화하지 않음 (시간 유지)
}

/**
 * 타이머 초기화
 */
function initializeTimer() {
    // 초기에는 타이머를 시작하지 않음
    // 웹캠이 활성화될 때만 시작됨
    
    // 타이머 표시 요소 초기화
    const timerEl = document.querySelector('.monitor-timer');
    timerEl.textContent = '00:00:00';
    timerSeconds = 0;
}

/**
 * 웹캠 토글 함수 - 기존 함수 사용
 * - toggleWebcam() 함수는 그대로 사용하며, 내부에서 updateStatusIndicator() 호출
 */

/**
 * 탭 전환 기능 초기화
 */
function initializeTabs(tabItems) {
    tabItems.forEach(item => {
        item.addEventListener('click', () => {
            tabItems.forEach(tab => tab.classList.remove('active'));
            item.classList.add('active');
            
            // 탭 컨텐츠 전환 구현 가능
            console.log(`${item.textContent} 탭으로 전환`);
        });
    });
}

/**
 * 웹캠 토글 함수
 */
function toggleWebcam() {
    const videoElement = document.querySelector('.webcam-feed');
    const webcamBtn = document.querySelector('.control-btn:nth-child(1) .material-icons');
    
    if (videoStream) {
        // 웹캠 중지
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        videoElement.srcObject = null;
        webcamBtn.textContent = 'videocam';
        updateStatusIndicator('비활성화', false);
    } else {
        // 웹캠 시작
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(stream => {
                videoStream = stream;
                videoElement.srcObject = stream;
                webcamBtn.textContent = 'videocam_off';
                updateStatusIndicator('활성화', true);
                
                // 5초마다 자동으로 사진 캡처하여 감정 분석
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
    }, 30000); // 30초
    
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
                
                // AI 조교의 감정 상태 기반 피드백
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
    // 감정 상태에 따른 UI 색상/스타일 변경 가능
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
    document.querySelector('.status-dot').style.backgroundColor = color;
}

/**
 * 타이머 토글 기능
 */
function toggleTimer() {
    alert('타이머 기능을 설정합니다.');
    // 타이머 설정 모달 또는 추가 기능 구현 가능
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
 * 학습 상태 분석
 */
function analyzeStudyState() {
    if (!videoStream) {
        alert('먼저 카메라를 활성화해주세요.');
        return;
    }
    
    // 분석 중 표시
    const originalBtnText = document.querySelector('.analysis-btn span:nth-child(2)').textContent;
    document.querySelector('.analysis-btn span:nth-child(2)').textContent = '분석 중...';
    document.querySelector('.analysis-btn').disabled = true;
    
    // 현재 상태 분석 요청
    captureAndAnalyze(true);
    
    // 버튼 상태 복원
    setTimeout(() => {
        document.querySelector('.analysis-btn span:nth-child(2)').textContent = originalBtnText;
        document.querySelector('.analysis-btn').disabled = false;
    }, 2000);
}

/**
 * 채팅 기능 초기화
 */
function initializeChatFunctions(chatInput, sendBtn) {
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
 * 채팅 메시지 표시
 */
function displayChatMessage(message, sender) {
    // 메시지 저장
    chatMessages.push({ sender, content: message });
    
    // 컨테이너에 추가
    const assistantContainer = document.querySelector('.assistant-container');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    
    let messageContent = '';
    if (sender === 'user') {
        messageContent = `<div class="message-bubble user">
                            <div class="message-sender">나</div>
                            <div class="message-text">${message}</div>
                          </div>`;
    } else if (sender === 'assistant') {
        messageContent = `<div class="message-bubble assistant">
                            <div class="message-sender">FocusMate AI</div>
                            <div class="message-text">${message}</div>
                          </div>`;
    } else {
        messageContent = `<div class="message-bubble system">
                            <div class="message-text">${message}</div>
                          </div>`;
    }
    
    messageElement.innerHTML = messageContent;
    
    // 기존 입력창 위에 메시지 삽입
    const chatInputContainer = document.querySelector('.chat-input-container');
    assistantContainer.insertBefore(messageElement, chatInputContainer);
    
    // 스크롤 자동 이동
    messageElement.scrollIntoView({ behavior: 'smooth' });
}

/**
 * API에 채팅 전송
 */
function sendChatToApi(message) {
    // 타이핑 중 표시
    displayChatMessage('메시지를 생성하는 중입니다...', 'system');
    
    // API 요청 객체
    const requestData = {
        messages: [
            { role: 'user', content: message }
        ],
        session_id: sessionId
    };
    
    // API 요청
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // 타이핑 중 메시지 제거 (마지막 시스템 메시지 제거)
        const systemMessage = document.querySelector('.message.system:last-child');
        if (systemMessage) systemMessage.remove();
        
        if (data.error) {
            displayChatMessage(`오류가 발생했습니다: ${data.error}`, 'system');
        } else {
            // 응답 표시
            displayChatMessage(data.message, 'assistant');
        }
    })
    .catch(err => {
        // 타이핑 중 메시지 제거
        const systemMessage = document.querySelector('.message.system:last-child');
        if (systemMessage) systemMessage.remove();
        
        console.error('API 오류:', err);
        displayChatMessage('서버와 통신 중 오류가 발생했습니다.', 'system');
    });
}

/**
 * 고유 세션 ID 생성
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// CSS 추가
function addChatStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .message {
            margin-bottom: 16px;
        }
        
        .message-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            position: relative;
        }
        
        .message-sender {
            font-size: 0.8rem;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .user .message-bubble {
            background-color: var(--primary);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .assistant .message-bubble {
            background-color: var(--secondary);
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        
        .system .message-bubble {
            background-color: #f3f4f6;
            font-size: 0.9rem;
            text-align: center;
            margin: 8px auto;
            max-width: 90%;
        }
        
        .webcam-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    `;
    document.head.appendChild(styleElement);
}

// 스타일 추가
addChatStyles();