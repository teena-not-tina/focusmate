/**
 * 공통 인증 관련 자바스크립트
 * - 로그인 및 회원가입 기능 통합
 */
document.addEventListener('DOMContentLoaded', function () {
    // 로그인 폼 처리
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }

    // 회원가입 폼 처리
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', handleSignup);

        // 실시간 유효성 검사 이벤트
        if (document.getElementById('username')) {
            document.getElementById('username').addEventListener('input', debounce(checkUsername, 500));
        }

        if (document.getElementById('email')) {
            document.getElementById('email').addEventListener('input', debounce(checkEmail, 500));
        }

        if (document.getElementById('password')) {
            document.getElementById('password').addEventListener('input', checkPasswordStrength);
        }

        if (document.getElementById('confirm_password')) {
            document.getElementById('confirm_password').addEventListener('input', checkPasswordMatch);
        }
    }

    // 테마 전환 버튼 처리
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
});

/**
 * 로그인 처리
 */
function handleLogin(e) {
    e.preventDefault();

    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const remember = document.getElementById('remember')?.checked || false;

    if (!username || !password) {
        showMessage('login-message', '아이디와 비밀번호를 입력해주세요.', 'error');
        return;
    }

    // API 요청
    fetch('/api/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username,
            password: password,
            remember: remember
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = data.redirect || '/';
            } else {
                showMessage('login-message', data.message || '로그인에 실패했습니다.', 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('login-message', '서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'error');
        });
}

/**
 * 회원가입 처리
 */
function handleSignup(e) {
    e.preventDefault();

    if (!validateSignupForm()) {
        return;
    }

    const formData = {
        username: document.getElementById('username').value.trim(),
        email: document.getElementById('email').value.trim(),
        password: document.getElementById('password').value,
        user_type: document.getElementById('user_type').value
    };

    // API 요청
    fetch('/api/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('signup-message', '회원가입이 성공적으로 완료되었습니다!', 'success');
                setTimeout(() => {
                    window.location.href = '/login';
                }, 1500);
            } else {
                if (data.field) {
                    showFieldError(data.field, data.message);
                } else {
                    showMessage('signup-message', data.message || '회원가입에 실패했습니다.', 'error');
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('signup-message', '서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'error');
        });
}

/**
 * 실시간 아이디 중복 체크
 */
function checkUsername() {
    const username = document.getElementById('username').value.trim();
    if (username.length < 4) {
        showFieldError('username', '아이디는 4자 이상이어야 합니다.');
        return;
    }

    fetch(`/api/check_username?username=${encodeURIComponent(username)}`)
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                showFieldSuccess('username', '사용 가능한 아이디입니다.');
            } else {
                showFieldError('username', '이미 사용 중인 아이디입니다.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

/**
 * 실시간 이메일 중복 체크
 */
function checkEmail() {
    const email = document.getElementById('email').value.trim();
    if (!validateEmail(email)) {
        showFieldError('email', '유효한 이메일 주소를 입력해주세요.');
        return;
    }

    fetch(`/api/check_email?email=${encodeURIComponent(email)}`)
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                showFieldSuccess('email', '사용 가능한 이메일입니다.');
            } else {
                showFieldError('email', '이미 사용 중인 이메일입니다.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

/**
 * 비밀번호 강도 체크
 */
function checkPasswordStrength() {
    const password = document.getElementById('password').value;

    if (password.length < 8) {
        showFieldError('password', '비밀번호는 8자 이상이어야 합니다.');
        return;
    }

    let strength = 0;
    if (/[A-Z]/.test(password)) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;

    switch (strength) {
        case 1:
            showFieldError('password', '비밀번호가 너무 약합니다. 대소문자, 숫자, 특수문자를 포함해주세요.');
            break;
        case 2:
            showFieldWarning('password', '비밀번호가 약합니다. 더 다양한 문자를 사용해주세요.');
            break;
        case 3:
            showFieldSuccess('password', '괜찮은 비밀번호입니다.');
            break;
        case 4:
            showFieldSuccess('password', '매우 안전한 비밀번호입니다!');
            break;
    }

    // 비밀번호 확인 필드가 있고 값이 있으면 일치 여부 다시 확인
    const confirmPassword = document.getElementById('confirm_password');
    if (confirmPassword && confirmPassword.value) {
        checkPasswordMatch();
    }
}

/**
 * 비밀번호 확인 일치 체크
 */
function checkPasswordMatch() {
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm_password').value;

    if (confirmPassword === '') {
        return;
    }

    if (password === confirmPassword) {
        showFieldSuccess('confirm_password', '비밀번호가 일치합니다.');
    } else {
        showFieldError('confirm_password', '비밀번호가 일치하지 않습니다.');
    }
}

/**
 * 회원가입 폼 전체 유효성 검사
 */
function validateSignupForm() {
    let isValid = true;

    // 아이디 검사
    const username = document.getElementById('username').value.trim();
    if (username.length < 4) {
        showFieldError('username', '아이디는 4자 이상이어야 합니다.');
        isValid = false;
    }

    // 이메일 검사
    const email = document.getElementById('email').value.trim();
    if (!validateEmail(email)) {
        showFieldError('email', '유효한 이메일 주소를 입력해주세요.');
        isValid = false;
    }

    // 비밀번호 검사
    const password = document.getElementById('password').value;
    if (password.length < 8) {
        showFieldError('password', '비밀번호는 8자 이상이어야 합니다.');
        isValid = false;
    }

    // 비밀번호 확인
    const confirmPassword = document.getElementById('confirm_password').value;
    if (password !== confirmPassword) {
        showFieldError('confirm_password', '비밀번호가 일치하지 않습니다.');
        isValid = false;
    }

    return isValid;
}

/**
 * 테마 전환 함수
 */
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    // data-theme 속성 변경
    html.setAttribute('data-theme', newTheme);

    // body 클래스도 함께 변경
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${newTheme}-theme`);

    // 로컬 스토리지에 저장
    localStorage.setItem('theme', newTheme);

    // 테마 토글 버튼 아이콘 업데이트
    updateThemeToggleIcon(newTheme);
}

/**
 * 테마 토글 버튼 아이콘 업데이트
 */
function updateThemeToggleIcon(theme) {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        if (theme === 'dark') {
            themeToggle.innerHTML = '<span class="material-icons">light_mode</span>';
            themeToggle.setAttribute('title', '라이트 모드로 전환');
        } else {
            themeToggle.innerHTML = '<span class="material-icons">dark_mode</span>';
            themeToggle.setAttribute('title', '다크 모드로 전환');
        }
    }

    // 드롭다운 메뉴의 테마 항목도 업데이트
    const themeItem = document.querySelector('.dropdown-item[href="theme"]');
    if (themeItem) {
        if (theme === 'dark') {
            themeItem.innerHTML = '<span class="material-icons">light_mode</span> 라이트 모드';
        } else {
            themeItem.innerHTML = '<span class="material-icons">dark_mode</span> 다크 모드';
        }
    }
}

/**
 * 유틸리티 함수: 메시지 표시
 */
function showMessage(elementId, message, type = 'info') {
    const messageElement = document.getElementById(elementId);
    if (!messageElement) {
        console.warn(`Message element #${elementId} not found`);
        return;
    }

    messageElement.textContent = message;
    messageElement.className = `message ${type}-message`;
    messageElement.style.display = 'block';

    // 성공 메시지는 일정 시간 후 사라지게 함
    if (type === 'success') {
        setTimeout(() => {
            messageElement.style.display = 'none';
        }, 5000);
    }
}

/**
 * 유틸리티 함수: 필드 오류 표시
 */
function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    const validationElement = document.getElementById(`${fieldId}-validation`);

    if (field) {
        field.classList.add('error');
        field.classList.remove('success', 'warning');
    }

    if (validationElement) {
        validationElement.textContent = message;
        validationElement.className = 'validation-message error-message';
    }
}

/**
 * 유틸리티 함수: 필드 경고 표시
 */
function showFieldWarning(fieldId, message) {
    const field = document.getElementById(fieldId);
    const validationElement = document.getElementById(`${fieldId}-validation`);

    if (field) {
        field.classList.add('warning');
        field.classList.remove('error', 'success');
    }

    if (validationElement) {
        validationElement.textContent = message;
        validationElement.className = 'validation-message warning-message';
    }
}

/**
 * 유틸리티 함수: 필드 성공 표시
 */
function showFieldSuccess(fieldId, message) {
    const field = document.getElementById(fieldId);
    const validationElement = document.getElementById(`${fieldId}-validation`);

    if (field) {
        field.classList.add('success');
        field.classList.remove('error', 'warning');
    }

    if (validationElement) {
        validationElement.textContent = message;
        validationElement.className = 'validation-message success-message';
    }
}

/**
 * 유틸리티 함수: 이메일 유효성 검사
 */
function validateEmail(email) {
    const re = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return re.test(email);
}

/**
 * 유틸리티 함수: 디바운스
 */
function debounce(func, wait) {
    let timeout;
    return function () {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

/**
 * 로드 시 테마 적용
 */
function loadThemePreference() {
    const savedTheme = localStorage.getItem('theme') || 'light';

    // data-theme 속성 설정
    document.documentElement.setAttribute('data-theme', savedTheme);

    // body 클래스 설정
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${savedTheme}-theme`);

    // 테마 토글 버튼 업데이트
    updateThemeToggleIcon(savedTheme);
}

// 페이지 로드 시 테마 적용
loadThemePreference();