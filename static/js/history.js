// 히스토리 페이지 스크립트

document.addEventListener('DOMContentLoaded', () => {
    // 데이터 로드
    loadHistoryData();
    
    // 필터 이벤트
    document.getElementById('filter-period').addEventListener('change', loadHistoryData);
    
    // 감정 필터 체크박스
    document.querySelectorAll('input[name="emotion"]').forEach(checkbox => {
        checkbox.addEventListener('change', loadHistoryData);
    });
    
    // 내보내기 버튼
    document.getElementById('export-data').addEventListener('click', exportData);
});

// 히스토리 데이터 로드
async function loadHistoryData() {
    try {
        // 로딩 표시
        document.getElementById('history-items').innerHTML = '<div class="loading-message">데이터 로드 중...</div>';
        
        // 필터 값 가져오기
        const period = document.getElementById('filter-period').value;
        const emotions = Array.from(document.querySelectorAll('input[name="emotion"]:checked'))
                            .map(cb => cb.value);
        
        // 서버에 요청
        const response = await fetch('/api/history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ period, emotions })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 데이터 표시
        displayHistoryData(data);
        
        // 통계 차트 업데이트
        updateStatistics(data);
        
    } catch (error) {
        console.error('히스토리 데이터 로드 오류:', error);
        document.getElementById('history-items').innerHTML = 
            `<div class="error-message">데이터를 로드할 수 없습니다: ${error.message}</div>`;
    }
}

// 히스토리 데이터 표시
function displayHistoryData(data) {
    const historyContainer = document.getElementById('history-items');
    const emptyState = document.getElementById('empty-history');
    
    // 데이터가 없는 경우
    if (!data.records || data.records.length === 0) {
        historyContainer.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }
    
    // 데이터가 있는 경우
    emptyState.style.display = 'none';
    
    // HTML 생성
    let html = '';
    
    // 날짜별 그룹화
    const groupedByDate = groupByDate(data.records);
    
    // 각 날짜별 기록 표시
    for (const [date, records] of Object.entries(groupedByDate)) {
        html += `
            <div class="history-date">
                <h4>${formatDate(date)}</h4>
                <div class="history-date-items">
        `;
        
        // 각 기록 표시
        records.forEach(record => {
            html += `
                <div class="history-item">
                    <div class="history-item-time">${formatTime(record.timestamp)}</div>
                    <div class="history-item-content">
                        <div class="emotion-badge ${record.emotion}">${getEmotionLabel(record.emotion)}</div>
                        <div class="history-item-message">${record.message || '감정 상태 기록'}</div>
                    </div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    historyContainer.innerHTML = html;
}

// 통계 업데이트
function updateStatistics(data) {
    // 기본 통계 업데이트
    document.getElementById('total-sessions').textContent = data.stats?.totalSessions || 0;
    document.getElementById('focus-time').textContent = data.stats?.focusTimeFormatted || '0분';
    document.getElementById('dominant-emotion').textContent = 
        getEmotionLabel(data.stats?.dominantEmotion || '-');
    
    // 차트 업데이트
    createEmotionChart(data.stats?.emotionCounts || {});
}

// 감정 차트 생성
function createEmotionChart(emotionCounts) {
    const ctx = document.getElementById('emotion-chart').getContext('2d');
    
    // 기존 차트 제거
    if (window.emotionChart) {
        window.emotionChart.destroy();
    }
    
    // 차트 데이터 준비
    const labels = Object.keys(emotionCounts).map(getEmotionLabel);
    const data = Object.values(emotionCounts);
    
    // 차트 색상
    const colors = {
        focused: '#4F46E5',
        interested: '#10B981',
        neutral: '#6B7280',
        anxious: '#F59E0B',
        tired: '#EF4444'
    };
    
    const backgroundColor = Object.keys(emotionCounts).map(emotion => colors[emotion] || '#6B7280');
    
    // 차트 생성
    window.emotionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColor,
                borderColor: 'rgba(255, 255, 255, 0.6)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 12,
                        padding: 15
                    }
                }
            },
            cutout: '70%'
        }
    });
}

// 데이터 내보내기
async function exportData() {
    try {
        const response = await fetch('/api/export_data');
        const blob = await response.blob();
        
        // 파일 다운로드
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `focusmate_data_${formatDateFileName(new Date())}.json`;
        document.body.appendChild(a);
        a.click();
        
        window.URL.revokeObjectURL(url);
        a.remove();
        
        showNotification('데이터 내보내기가 완료되었습니다', 'success');
    } catch (error) {
        console.error('데이터 내보내기 오류:', error);
        showNotification('데이터 내보내기 중 오류가 발생했습니다', 'error');
    }
}

// 유틸리티 함수들
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('ko-KR', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric', 
        weekday: 'long' 
    });
}

function formatTime(timestampString) {
    const date = new Date(timestampString);
    return date.toLocaleTimeString('ko-KR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

function formatDateFileName(date) {
    return `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}`;
}

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

function groupByDate(records) {
    const groups = {};
    
    records.forEach(record => {
        const date = new Date(record.timestamp).toISOString().split('T')[0];
        if (!groups[date]) {
            groups[date] = [];
        }
        groups[date].push(record);
    });
    
    return groups;
}