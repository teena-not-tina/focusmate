// 채팅 기능 스크립트

document.addEventListener('DOMContentLoaded', () => {
    initializeChatFeatures();
});

// 채팅 기능 초기화
function initializeChatFeatures() {
    const sendButton = document.getElementById('send-btn');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.querySelector('.chat-messages');
    
    if (!sendButton || !messageInput || !chatMessages) return;
    
    // 전송 버튼 이벤트
    sendButton.addEventListener('click', () => {
        sendMessage();
    });
    
    // 입력창 엔터키 이벤트
    messageInput.addEventListener('keydown', (e) => {
        // Shift+Enter는 줄바꿈
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 첫 메시지 표시
    addAssistantMessage(
        "안녕하세요! FocusMate AI 학습 조교입니다. 학습에 관한 질문이나 도움이 필요한 내용이 있으면 언제든지 물어보세요."
    );
}

// 메시지 전송
async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.querySelector('.chat-messages');
    
    if (!messageInput || !chatMessages) return;
    
    // 입력값 가져오기
    const messageText = messageInput.value.trim();
    if (!messageText) return;
    
    // 사용자 메시지 표시
    addUserMessage(messageText);
    
    // 입력창 초기화
    messageInput.value = '';
    
    // 스크롤 최하단 이동
    scrollToBottom();
    
    // 타이핑 표시 추가
    const typingIndicator = addTypingIndicator();
    
    try {
        // 서버에 메시지 전송
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: [{
                    role: 'user',
                    content: messageText
                }],
                session_id: 'default_session'
            })
        });
        
        // 타이핑 표시 제거
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 응답 표시
        addAssistantMessage(data.message || '죄송합니다. 응답을 생성할 수 없습니다.');
        
    } catch (error) {
        // 오류 처리
        console.error('채팅 오류:', error);
        
        // 타이핑 표시 제거
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        addAssistantMessage(
            '죄송합니다. 메시지 처리 중 오류가 발생했습니다. 다시 시도해주세요.'
        );
    }
    
    // 스크롤 최하단 이동
    scrollToBottom();
}

// 사용자 메시지 추가
function addUserMessage(text) {
    const chatMessages = document.querySelector('.chat-messages');
    if (!chatMessages) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    messageElement.textContent = text;
    
    chatMessages.appendChild(messageElement);
}

// AI 조교 메시지 추가
function addAssistantMessage(text) {
    const chatMessages = document.querySelector('.chat-messages');
    if (!chatMessages) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message assistant-message';
    messageElement.textContent = text;
    
    chatMessages.appendChild(messageElement);
    
    // 스크롤 최하단 이동
    scrollToBottom();
}

// 타이핑 표시 추가
function addTypingIndicator() {
    const chatMessages = document.querySelector('.chat-messages');
    if (!chatMessages) return null;
    
    const typingElement = document.createElement('div');
    typingElement.className = 'message assistant-message typing-indicator';
    typingElement.innerHTML = '<span></span><span></span><span></span>';
    
    chatMessages.appendChild(typingElement);
    
    // 스크롤 최하단 이동
    scrollToBottom();
    
    return typingElement;
}

// 스크롤 최하단 이동
function scrollToBottom() {
    const chatMessages = document.querySelector('.chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}