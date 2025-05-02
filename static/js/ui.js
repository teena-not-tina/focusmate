// 메뉴 드롭다운 기능과 테마 전환 기능

// 메뉴 드롭다운 기능
document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.getElementById('menu-toggle');
    const dropdownContent = document.querySelector('.dropdown-content');
    const themeToggle = document.getElementById('theme-toggle');
    const settingsLink = document.getElementById('settings-link');
    const helpLink = document.getElementById('help-link');
    
    // 메뉴 토글
    if (menuToggle && dropdownContent) {
        menuToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            dropdownContent.classList.toggle('show');
        });
        
        // 문서 클릭 시 메뉴 닫기
        document.addEventListener('click', function(e) {
            if (dropdownContent.classList.contains('show') && 
                !dropdownContent.contains(e.target) && 
                e.target !== menuToggle) {
                dropdownContent.classList.remove('show');
            }
        });
    }
    
    // 테마 토글
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            toggleTheme();
            dropdownContent.classList.remove('show'); // 선택 후 메뉴 닫기
        });
    }
    
    // 설정 링크
    if (settingsLink) {
        settingsLink.addEventListener('click', function() {
            window.location.href = '/settings';
        });
    }
    
    // 도움말 링크
    if (helpLink) {
        helpLink.addEventListener('click', function() {
            window.location.href = '/help';
        });
    }
    
    // 페이지 로드 시 저장된 테마 적용
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
        
        // 테마에 맞는 아이콘 및 텍스트 설정
        const themeIcon = document.querySelector('#theme-toggle .material-icon');
        const themeText = document.querySelector('#theme-toggle .menu-text');
        
        if (themeIcon && themeText) {
            if (savedTheme === 'dark') {
                themeIcon.textContent = 'light_mode';
                themeText.textContent = '라이트모드';
            } else {
                themeIcon.textContent = 'dark_mode';
                themeText.textContent = '다크모드';
            }
        }
    }
});

// 테마 토글 함수
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    body.setAttribute('data-theme', newTheme);
    
    // 아이콘 변경
    const themeIcon = document.querySelector('#theme-toggle .material-icon');
    const themeText = document.querySelector('#theme-toggle .menu-text');
    
    if (themeIcon && themeText) {
        if (newTheme === 'dark') {
            themeIcon.textContent = 'light_mode';
            themeText.textContent = '라이트모드';
        } else {
            themeIcon.textContent = 'dark_mode';
            themeText.textContent = '다크모드';
        }
    }
    
    // 로컬 스토리지에 테마 저장
    localStorage.setItem('theme', newTheme);
}