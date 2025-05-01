#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import webbrowser
import urllib.parse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys

class MusicPlayer:
    def __init__(self, gui):
        self.gui = gui
        self.browser_opened = False
        self.driver = None
        self.music_playing = False
        self.current_search_term = None
    
    def reset_music_state(self):
        """음악 재생 상태 초기화"""
        self.current_search_term = None
        self.music_playing = False
    
    def play_music(self, search_term):
        """
        검색 및 자동 재생 (음악 재생 후 얼굴 감지 중지)
        
        Args:
            search_term (str): 검색어
            
        Returns:
            bool: 성공 여부
        """
        # 검색어 매핑 (코드에서 사용된 키워드를 실제 검색어로 변환)
        search_map = {
            "gamma": "감마파",
            "alpha": "알파파",
            "모짜르트": "모짜르트"
        }
        
        # 검색어 변환
        actual_search = search_map.get(search_term, search_term)
        
        try:
            self.gui.update_status(f"{actual_search} 검색 시작...")
            
            # 기존 브라우저 닫기
            self.close_browser()
            
            # 브라우저 초기화
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            self.driver = webdriver.Chrome(options=options)
            self.browser_opened = True
            
            # Pixabay 음악 페이지로 이동
            self.driver.get("https://pixabay.com/ko/music/")
            time.sleep(3)
            
            # 검색창 찾기 및 검색
            try:
                # 검색창 찾기
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div[1]/div[1]/div[2]/div[2]/div[2]/form/div[1]/input'))
                )
                
                # 검색창 클릭 및 내용 지우기
                search_input.click()
                search_input.clear()
                time.sleep(0.5)
                
                # 검색어 입력
                search_input.send_keys(actual_search)
                time.sleep(0.5)
                
                # 엔터키 입력으로 검색
                search_input.send_keys(Keys.RETURN)
                self.gui.update_status(f"{actual_search} 검색 완료, 결과 로딩 중...")
                
                # 결과 로딩 대기
                time.sleep(3)
                
                # 자동 재생 설정
                success = self.try_autoplay_music()
                
                if success:
                    # 성공적으로 재생 설정되면 얼굴 감지 중지 상태로 설정
                    self.music_playing = True
                    self.current_search_term = actual_search
                    self.gui.update_status(f"{actual_search} 음악 재생 중 - 얼굴 감지 일시 중지됨")
                
                return success
                
            except Exception as e:
                self.gui.update_status("검색창을 찾을 수 없습니다.", warning=True)
                
                # 직접 URL로 이동 시도
                try:
                    encoded_term = urllib.parse.quote(actual_search)
                    search_url = f"https://pixabay.com/ko/music/search/?search={encoded_term}"
                    
                    self.driver.get(search_url)
                    self.gui.update_status(f"{actual_search} 검색 페이지로 직접 이동합니다.")
                    time.sleep(3)
                    
                    # 자동 재생 설정
                    success = self.try_autoplay_music()
                    
                    if success:
                        # 성공적으로 재생 설정되면 얼굴 감지 중지 상태로 설정
                        self.music_playing = True
                        self.current_search_term = actual_search
                        self.gui.update_status(f"{actual_search} 음악 재생 중 - 얼굴 감지 일시 중지됨")
                    
                    return success
                    
                except Exception as direct_error:
                    self.gui.update_status("직접 URL로 이동 실패.", warning=True)
                    return False
            
        except Exception as e:
            self.gui.update_status("음악 검색 중 오류가 발생했습니다.", warning=True)
            
            # 일반 브라우저로 열기 시도
            try:
                encoded_term = urllib.parse.quote(actual_search)
                search_url = f"https://pixabay.com/ko/music/search/?search={encoded_term}"
                
                webbrowser.open(search_url)
                self.gui.update_status("기본 브라우저에서 페이지가 열렸습니다. 수동으로 재생해주세요.")
            except:
                pass
                
            return False

    def try_autoplay_music(self):
        """개선된 자동 재생 함수 - 끊김 없는 연속 재생 보장"""
        try:
            # 페이지 로딩 대기
            time.sleep(3)
            
            # 재생 버튼 XPath
            play_button_xpath = '//*[@id="app"]/div[1]/div[2]/div[2]/div[2]/div/div[1]/div[1]/div[1]/button'
            alt_button_xpath = '//*[@id="app"]/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/button'
            
            # 자동 반복 재생 스크립트 (개선된 버전)
            repeat_script = """
                // 개선된 반복 재생 설정 (짧은 간격으로 재생)
                function setupImprovedRepeatPlay() {
                    // 버튼 XPath
                    const buttonXPaths = [
                        '//*[@id="app"]/div[1]/div[2]/div[2]/div[2]/div/div[1]/div[1]/div[1]/button',
                        '//*[@id="app"]/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/button'
                    ];
                    
                    // 플레이어 상태 변수
                    let isPlaying = false;
                    let playCount = 0;
                    
                    // 버튼 찾기 함수
                    function findButton() {
                        // XPath로 버튼 찾기
                        for (const xpath of buttonXPaths) {
                            try {
                                const button = document.evaluate(
                                    xpath, document, null, 
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null
                                ).singleNodeValue;
                                
                                if (button) {
                                    return button;
                                }
                            } catch (e) {}
                        }
                        
                        // CSS 선택자로 버튼 찾기
                        return document.querySelector(
                            '.search-results-item:first-child button, ' + 
                            '.music-item:first-child button, ' +
                            'button.play-button, ' +
                            '[aria-label*="play"]'
                        );
                    }
                    
                    // 오디오 요소 찾기 함수
                    function findAudio() {
                        const audios = document.querySelectorAll('audio');
                        for (const audio of audios) {
                            return audio; // 첫 번째 오디오 요소 반환
                        }
                        return null;
                    }
                    
                    // 더 신뢰성 있는 버튼 클릭 함수
                    function clickButtonReliably(button) {
                        if (!button) return false;
                        
                        try {
                            // 버튼으로 스크롤
                            button.scrollIntoView({block: 'center'});
                            
                            // 다양한 방법으로 클릭 시도
                            setTimeout(() => {
                                try {
                                    // 1. 일반 클릭
                                    button.click();
                                } catch (e1) {
                                    try {
                                        // 2. 이벤트 디스패치
                                        const event = new MouseEvent('click', {
                                            bubbles: true,
                                            cancelable: true,
                                            view: window
                                        });
                                        button.dispatchEvent(event);
                                    } catch (e2) {}
                                }
                                
                                // 재생 카운트 증가
                                playCount++;
                                console.log(`음원 재생 ${playCount}회 시도`);
                                
                                // 상태 표시
                                const statusElement = document.getElementById('playback-status');
                                if (statusElement) {
                                    statusElement.textContent = `재생 중: ${playCount}회 클릭`;
                                }
                            }, 500);
                            
                            return true;
                        } catch (e) {
                            return false;
                        }
                    }
                    
                    // 첫 번째 버튼 찾기
                    const initialButton = findButton();
                    if (!initialButton) {
                        return {
                            success: false,
                            message: "재생 버튼을 찾을 수 없음"
                        };
                    }
                    
                    // 상태 표시 UI 생성
                    const statusElement = document.createElement('div');
                    statusElement.id = 'playback-status';
                    statusElement.textContent = '재생 준비 중...';
                    statusElement.style.position = 'fixed';
                    statusElement.style.top = '10px';
                    statusElement.style.right = '10px';
                    statusElement.style.background = 'rgba(0, 128, 0, 0.8)';
                    statusElement.style.color = 'white';
                    statusElement.style.padding = '8px 12px';
                    statusElement.style.borderRadius = '4px';
                    statusElement.style.fontWeight = 'bold';
                    statusElement.style.zIndex = '10000';
                    document.body.appendChild(statusElement);
                    
                    // 초기 클릭
                    clickButtonReliably(initialButton);
                    
                    // 짧은 간격(3초)으로 버튼 클릭 - 음원이 끝나기 전에 미리 다시 클릭하여 끊김 방지
                    const clickInterval = setInterval(() => {
                        const button = findButton();
                        if (button) {
                            clickButtonReliably(button);
                        }
                    }, 3000);
                    
                    return {
                        success: true,
                        message: "반복 재생 설정 완료"
                    };
                }
                
                return setupImprovedRepeatPlay();
            """
            
            # 스크립트 실행
            result = self.driver.execute_script(repeat_script)
            
            if result and result.get('success'):
                self.gui.update_status("자동 재생 설정 완료. 3초마다 버튼이 자동으로 클릭됩니다.")
                return True
            else:
                # 직접 버튼 찾아서 클릭
                try:
                    button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, play_button_xpath))
                    )
                except:
                    try:
                        button = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, alt_button_xpath))
                        )
                    except:
                        self.gui.update_status("재생 버튼을 찾을 수 없습니다.", warning=True)
                        return False
                
                # 버튼 클릭
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", button)
                
                self.gui.update_status("초기 재생 버튼 클릭 완료")
                
                # 간단한 반복 클릭 설정 (간격 3초로 단축)
                self.driver.execute_script("""
                    // 3초마다 같은 버튼 클릭
                    setInterval(() => {
                        const buttons = [
                            document.evaluate('//*[@id="app"]/div[1]/div[2]/div[2]/div[2]/div/div[1]/div[1]/div[1]/button', 
                                              document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue,
                            document.evaluate('//*[@id="app"]/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/button', 
                                              document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue
                        ];
                        
                        // 발견된 첫 번째 버튼 클릭
                        for (const btn of buttons) {
                            if (btn) {
                                btn.scrollIntoView({block: 'center'});
                                setTimeout(() => btn.click(), 300);
                                break;
                            }
                        }
                    }, 3000);  // 3초마다 클릭 (더 짧게 설정)
                """)
                
                self.gui.update_status("연속 재생 설정 완료")
                return True
            
        except Exception as e:
            self.gui.update_status("재생 설정 중 오류가 발생했습니다.", warning=True)
            return False

    def close_browser(self):
        """브라우저 닫기 시도"""
        if not self.browser_opened:
            return
        
        try:
            # Selenium 웹드라이버 종료
            if self.driver is not None:
                try:
                    self.driver.quit()
                except:
                    pass
            
            # 추가적으로 Chrome 프로세스 종료 시도 (Windows의 경우)
            if os.name == 'nt':
                try:
                    os.system('taskkill /f /im chromedriver.exe')
                    os.system('taskkill /f /im chrome.exe')
                except:
                    pass
                
            self.browser_opened = False
            self.driver = None
            self.music_playing = False
            self.gui.update_status("브라우저가 종료되었습니다.")
        except Exception as e:
            pass