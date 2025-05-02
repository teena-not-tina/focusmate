// 카메라 및 감정 감지 관련 기능

let videoElement;
let captureInterval;
let detectionActive = false;

function initializeEmotionDetection() {
    videoElement = document.getElementById('video-feed');
    const startButton = document.getElementById('start-detection');
    const stopButton = document.getElementById('stop-detection');
    const fileInput = document.getElementById('file-upload');
    
    if (!videoElement || !startButton || !stopButton) return;
    
    // 카메라 시작 버튼
    startButton.addEventListener('click', () => {
        startCamera()
            .then(() => {
                startEmotionDetection();
                startButton.style.display = 'none';
                stopButton.style.display = 'inline-block';
            })
            .catch(error => {
                showError(`카메라를 시작할 수 없습니다: ${error.message}`);
                console.error('카메라 시작 오류:', error);
            });
    });
    
    // 카메라 정지 버튼
    stopButton.addEventListener('click', () => {
        stopEmotionDetection();
        stopCamera();
        startButton.style.display = 'inline-block';
        stopButton.style.display = 'none';
    });
    
    // 파일 업로드
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
}

// 카메라 스트림 시작
async function startCamera() {
    try {
        const constraints = {
            video: { facingMode: 'user' },
            audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        return new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });
    } catch (error) {
        console.error('카메라 접근 오류:', error);
        throw error;
    }
}

// 카메라 스트림 정지
function stopCamera() {
    if (!videoElement.srcObject) return;
    
    const tracks = videoElement.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    videoElement.srcObject = null;
}

// 감정 감지 시작
function startEmotionDetection() {
    if (detectionActive) return;
    
    detectionActive = true;
    
    // 3초마다 감정 감지
    captureInterval = setInterval(() => {
        captureAndDetectEmotion();
    }, 3000);
    
    // 즉시 첫 감지 실행
    captureAndDetectEmotion();
}

// 감정 감지 정지
function stopEmotionDetection() {
    detectionActive = false;
    clearInterval(captureInterval);
}

// 이미지 캡처 및 감정 감지
async function captureAndDetectEmotion() {
    if (!videoElement.srcObject) return;
    
    try {
        // 로딩 표시
        document.getElementById('loading-overlay').style.display = 'flex';
        
        // 현재 비디오 프레임 캡처
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // 이미지를 Blob으로 변환
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
        
        // 서버에 요청 보내기
        await detectEmotionFromImage(blob);
        
    } catch (error) {
        console.error('감정 감지 오류:', error);
        showError('감정 감지 중 오류가 발생했습니다.');
    } finally {
        document.getElementById('loading-overlay').style.display = 'none';
    }
}

// 파일 업로드 처리
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        document.getElementById('loading-overlay').style.display = 'flex';
        
        // 이미지 파일 확인
        if (!file.type.match('image.*')) {
            throw new Error('이미지 파일만 지원됩니다.');
        }
        
        // 파일 크기 확인 (10MB 제한)
        if (file.size > 10 * 1024 * 1024) {
            throw new Error('파일 크기는 10MB 이하여야 합니다.');
        }
        
        // 서버에 전송
        await detectEmotionFromImage(file);
        
    } catch (error) {
        console.error('파일 처리 오류:', error);
        showError(`파일 처리 중 오류가 발생했습니다: ${error.message}`);
    } finally {
        document.getElementById('loading-overlay').style.display = 'none';
        // 파일 입력 필드 초기화
        event.target.value = '';
    }
}

// 이미지를 서버에 전송하여 감정 감지
async function detectEmotionFromImage(imageFile) {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('type', 'image');
        
        // 질문이 있는 경우 추가
        const questionElement = document.getElementById('question-input');
        if (questionElement && questionElement.value.trim()) {
            formData.append('query', questionElement.value.trim());
        }
        
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 결과 표시
        displayEmotionResults(data);
        
    } catch (error) {
        console.error('감정 감지 API 오류:', error);
        showError(`감정 감지 실패: ${error.message}`);
    }
}

// 감정 감지 결과 표시
function displayEmotionResults(data) {
    const resultElement = document.getElementById('emotion-result');
    if (!resultElement) return;
    
    // 감정 결과 표시
    const dominantEmotion = data.dominant_emotion || 'neutral';
    
    // 결과 HTML 구성
    let resultHTML = `
        <div class="emotion-card">
            <div class="emotion-label">
                <span>감정 상태:</span>
                <span class="emotion-badge">${getEmotionLabel(dominantEmotion)}</span>
            </div>
            <div class="emotion-message">${data.response || '감정 데이터를 불러올 수 없습니다.'}</div>
        </div>
    `;
    
    resultElement.innerHTML = resultHTML;
}

// 감정 라벨 번역
function getEmotionLabel(emotion) {
    const labels = {
        'focused': '집중',
        'interested': '흥미',
        'neutral': '중립',
        'anxious': '불안',
        'tired': '피로'
    };
    
    return labels[emotion] || emotion;
}