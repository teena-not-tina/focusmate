// 메인 스크립트 - 앱 초기화 및 전역 기능

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    
    // 감정 감지 기능 초기화
    if (document.getElementById('start-detection')) {
        initializeEmotionDetection();
    }
    
    // 모달 이벤트 초기화
    initializeModalEvents();
    
    console.log('FocusMate application initialized');
});

// 탭 전환 기능
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    if (!tabButtons.length) return;
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // 기존 활성 탭 비활성화
            document.querySelectorAll('.tab-button.active').forEach(activeBtn => {
                activeBtn.classList.remove('active');
            });
            
            document.querySelectorAll('.tab-content.active').forEach(activeContent => {
                activeContent.classList.remove('active');
            });
            
            // 새 탭 활성화
            const targetTab = button.dataset.tab;
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

// 모달 창 이벤트 초기화
function initializeModalEvents() {
    const modal = document.getElementById('confirm-modal');
    
    if (!modal) return;
    
    // 닫기 버튼
    const closeButtons = modal.querySelectorAll('.cancel-btn, .close-modal');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    });
    
    // 모달 외부 클릭 시 닫기
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// 오류 메시지 표시 함수
function showError(message, duration = 5000) {
    const errorElement = document.getElementById('error-message');
    
    if (!errorElement) return;
    
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // 지정된 시간 후 오류 메시지 숨기기
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, duration);
}

// 감정 감지 초기화 함수 - camera.js에서 구현
function initializeEmotionDetection() {
    // camera.js에서 구현됨
}