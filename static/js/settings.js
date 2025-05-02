// 설정 페이지 스크립트

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    
    // 설정 저장 버튼
    document.getElementById('save-settings').addEventListener('click', saveSettings);
    
    // 기본값으로 재설정 버튼
    document.getElementById('reset-settings').addEventListener('click', resetSettings);
    
    // 데이터 초기화 버튼
    document.getElementById('clear-data').addEventListener('click', () => {
        if (confirm('모든 대화 내용과 감정 기록이 삭제됩니다. 계속하시겠습니까?')) {
            clearAllData();
        }
    });
});

// 저장된 설정 불러오기
function loadSettings() {
    // 테마 설정
    const savedTheme = localStorage.getItem('theme-setting') || 'system';
    document.getElementById('theme-setting').value = savedTheme;
    
    // 언어 설정
    const savedLanguage = localStorage.getItem('language-setting') || 'ko';
    document.getElementById('language-setting').value = savedLanguage;
    
    // 민감도 설정
    const savedSensitivity = localStorage.getItem('sensitivity-setting') || 'medium';
    document.getElementById('sensitivity-setting').value = savedSensitivity;
    
    // 감지 간격 설정
    const savedInterval = localStorage.getItem('interval-setting') || '3';
    document.getElementById('interval-setting').value = savedInterval;
    
    // 조교 스타일 설정
    const savedStyle = localStorage.getItem('assistant-style') || 'friendly';
    document.getElementById('assistant-style').value = savedStyle;
    
    // 자동 추천 설정
    const autoRecommend = localStorage.getItem('auto-recommend') !== 'false';
    document.getElementById('auto-recommend').checked = autoRecommend;
    
    // 데이터 저장 설정
    const saveData = localStorage.getItem('save-data') !== 'false';
    document.getElementById('save-data').checked = saveData;
}

// 설정 저장
function saveSettings() {
    // 테마 설정
    const theme = document.getElementById('theme-setting').value;
    localStorage.setItem('theme-setting', theme);
    
    // 언어 설정
    const language = document.getElementById('language-setting').value;
    localStorage.setItem('language-setting', language);
    
    // 민감도 설정
    const sensitivity = document.getElementById('sensitivity-setting').value;
    localStorage.setItem('sensitivity-setting', sensitivity);
    
    // 감지 간격 설정
    const interval = document.getElementById('interval-setting').value;
    localStorage.setItem('interval-setting', interval);
    
    // 조교 스타일 설정
    const style = document.getElementById('assistant-style').value;
    localStorage.setItem('assistant-style', style);
    
    // 자동 추천 설정
    const autoRecommend = document.getElementById('auto-recommend').checked;
    localStorage.setItem('auto-recommend', autoRecommend);
    
    // 데이터 저장 설정
    const saveData = document.getElementById('save-data').checked;
    localStorage.setItem('save-data', saveData);
    
    // 설정 변경 알림
    showNotification('설정이 저장되었습니다', 'success');
    
    // 테마 설정 즉시 적용
    applyThemeSetting(theme);
}

// 테마 설정 적용
function applyThemeSetting(theme) {
    const body = document.body;
    
    // 모든 테마 클래스 제거
    body.classList.remove('dark-mode', 'light-mode');
    
    // 선택된 테마에 따라 클래스 추가
    if (theme === 'dark') {
        body.classList.add('dark-mode');
    } else if (theme === 'light') {
        body.classList.add('light-mode');
    } else {
        // 시스템 기본 설정 따르기
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            body.classList.add('dark-mode');
        } else {
            body.classList.add('light-mode');
        }
    }
}

// 설정 초기화
function resetSettings() {
    if (confirm('모든 설정을 기본값으로 되돌리시겠습니까?')) {
        localStorage.removeItem('theme-setting');
        localStorage.removeItem('language-setting');
        localStorage.removeItem('sensitivity-setting');
        localStorage.removeItem('interval-setting');
        localStorage.removeItem('assistant-style');
        localStorage.removeItem('auto-recommend');
        localStorage.removeItem('save-data');
        
        loadSettings();
        showNotification('설정이 기본값으로 초기화되었습니다', 'info');
    }
}

// 모든 데이터 초기화
async function clearAllData() {
    try {
        // 서버에 데이터 초기화 요청
        const response = await fetch('/api/clear_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('모든 데이터가 삭제되었습니다', 'success');
        } else {
            showNotification('데이터 삭제 중 오류가 발생했습니다', 'error');
        }
    } catch (error) {
        console.error('데이터 초기화 오류:', error);
        showNotification('서버 연결 오류가 발생했습니다', 'error');
    }
}