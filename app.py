이전 CSS 코드가 사이드바를 여는 메뉴 버튼(Hamburger Menu)까지 숨겨버려서 사이드바가 안 보여.
CSS를 수정해서 **"사이드바 버튼은 보이게 하고"**, 나머지 불필요한 요소(왕관, 프로필, 푸터)만 숨겨줘.

`app.py`의 `hide_st_style` 부분을 아래 코드로 교체해줘:

```python
# --- HIDE STREAMLIT BRANDING ---
hide_st_style = """
<style>
    /* 1. 상단 헤더(Header)는 보이게 하되, 배경색 투명하게 처리 (사이드바 버튼 살리기 위함) */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* 2. 햄버거 메뉴(사이드바 버튼)는 무조건 보이게 설정 */
    [data-testid="collapsedControl"] {
        display: block !important;
    }

    /* 3. 우측 상단/하단 'Manage App(왕관)', 'Deploy' 버튼 숨기기 */
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }

    /* 4. 프로필 사진(Running Man) 및 상태 위젯 숨기기 */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* 5. 하단 푸터(Made with Streamlit) 숨기기 */
    footer {
        display: none !important;
    }
    .stApp > footer {
        display: none !important;
    }

    /* 6. 모바일 하단 'Created by' 뱃지 숨기기 */
    .viewerBadge_container__1QS1n {
        display: none !important;
    }
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
