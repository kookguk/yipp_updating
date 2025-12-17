import streamlit as st
from io import BytesIO
from PIL import Image
import base64
import os
import time
import pandas as pd
import numpy as np

# ✅ Google GenAI SDK (v1.0 최신 버전)
from google import genai
from google.genai import types

# -----------------------------
# 0. 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="YIPP X KBO 선수 카드 업데이트",
    page_icon="logo.png",
    layout="centered"
)


# -----------------------------
# 1. Gemini Client 초기화
# -----------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error("❌ `.streamlit/secrets.toml` 파일에 `GEMINI_API_KEY`가 없습니다.")
    st.stop()
except Exception as e:
    st.error(f"❌ 클라이언트 연결 오류: {e}")
    st.stop()


# -----------------------------
# 상수 및 설정
# -----------------------------
# 로고 매칭을 위한 구단 리스트 (logos/구단명.png 파일과 일치해야 함)
KBO_TEAMS = [
    "SSG 랜더스", "롯데 자이언츠", "KIA 타이거즈", "삼성 라이온즈", "한화 이글스",
    "두산 베어스", "LG 트윈스", "KT 위즈", "NC 다이노스", "키움 히어로즈"
]

REFERENCE_IMAGE_PATH = "image.png"
LOGO_DIR = "logos"
CSV_FILE_PATH = "customer.csv"

# 테마 컬러 정의 (민트색)
THEME_COLOR = "#008F53"


# -----------------------------
# 세션 상태 초기화
# -----------------------------
def init_session_state():
    defaults = {
        "step": 1,
        "player_data": None,    # CSV에서 가져온 사용자 데이터 행
        "team": None,
        "player_name": "",      
        "account": "",
        "number": None,
        "position": None,
        "card_image_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# -----------------------------
# 유틸리티 함수
# -----------------------------
def load_reference_bytes():
    try:
        with open(REFERENCE_IMAGE_PATH, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_and_resize_logo(team_name, size=(80, 80)):
    # CSV에 저장된 팀 이름과 로고 파일명이 일치한다고 가정
    path = os.path.join(LOGO_DIR, f"{team_name}.png")
    if os.path.exists(path):
        try:
            img = Image.open(path)
            img.thumbnail(size)
            return img
        except Exception:
            return None
    return None

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def determine_position(row):
    """
    CSV 데이터를 기반으로 포지션을 결정하는 로직
    가장 높은 점수를 기록한 스탯을 기준으로 포지션 할당
    """
    # CSV 컬럼명과 매핑 (실제 CSV 컬럼명에 따라 수정 필요할 수 있음)
    try:
        stats = {
            "초공격형 레전드 슬러거": float(row.get('거래금액', 0)),
            "공격형 슈퍼소닉 리드오프": float(row.get('거래빈도', 0)),
            "밸런스형 육각형 올라운더": float(row.get('분산투자', 0)),
            "수비형 철벽 유격수": float(row.get('안정성_점수', 0)), 
            "안정형 정밀 타격 머신": float(row.get('해외비중', 0))  
        }
        # 가장 높은 값을 가진 키(포지션명) 반환
        best_pos = max(stats, key=stats.get)
        return best_pos
    except:
        return "밸런스형 육각형 올라운더" # 기본값

def validate_user(name, account):
    """
    customer.csv 파일을 읽어 이름과 계좌번호가 일치하는지 확인
    """
    if not os.path.exists(CSV_FILE_PATH):
        st.error("❌ 고객 데이터 파일(customer.csv)을 찾을 수 없습니다.")
        return False, None
    
    try:
        # CSV 읽기 (인코딩 문제는 상황에 따라 utf-8, cp949 등 조정 필요)
        df = pd.read_csv(CSV_FILE_PATH, dtype={'계좌번호': str})
        
        # 공백 제거 등 전처리
        df['이름'] = df['이름'].astype(str).str.strip()
        df['계좌번호'] = df['계좌번호'].astype(str).str.strip().str.replace('-', '') # 하이픈 제거 비교
        
        input_account = account.replace('-', '').strip()
        input_name = name.strip()
        
        # 일치하는 행 찾기
        user_row = df[(df['이름'] == input_name) & (df['계좌번호'] == input_account)]
        
        if not user_row.empty:
            # Series 객체 반환
            return True, user_row.iloc[0]
        else:
            return False, None
            
    except Exception as e:
        st.error(f"데이터 확인 중 오류 발생: {e}")
        return False, None


# -----------------------------
# 🔥 Gemini 이미지 생성 함수 (업데이트용)
# -----------------------------
def generate_updated_card_gemini(team: str, position: str, number: str, name: str, stats_data) -> bytes:
    
    model_id = "gemini-3-pro-image-preview"
    
    # CSV 데이터에서 값 추출 (컬럼명 매핑 확인 필요)
    # 없는 경우 기본값 처리
    p_avg = stats_data.get('AVG(수익률)', '???')
    p_ops = stats_data.get('OPS(활동성)', '???')
    p_era = stats_data.get('ERA(안정성)', '???')
    
    radar_power = stats_data.get('거래금액', 50)
    radar_defense = stats_data.get('안정성_점수', 50)
    radar_contact = stats_data.get('분산투자', 50)
    radar_speed = stats_data.get('거래빈도', 50)
    radar_global = stats_data.get('해외비중', 50)
    
    # 프롬프트 구성
    prompt_text = f"""
    You are an expert UI/UX designer for sports trading cards.
    
    [Task]
    Generate an **UPDATED** baseball player card image **optimized for Instagram Story sharing (9:16)**.
    STRICTLY follow the visual style, layout, and composition of the attached reference image.
    
    [Content to Replace & Details]
    1. **General Layout**: Follow the reference image format precisely.
    2. **Player Number**: Place the player number "{number}" in the top-left position.
    3. **Player Images**:
        - Generate **two** distinct photos of a **professional** baseball player (generic face) wearing the **"{team}"** uniform (use team colors).
        - **Photo 1 (Front View)**: An energetic action shot reflecting the position "{position}".
        - **Photo 2 (Back View)**: **MUST show the player from the back.** On the back of the jersey, **clearly display the number "{number}" and the Korean name "{name}"**.
        - Composition: Blend these two images artistically (e.g., large foreground, background accent).
    4. **Player Name**:
        - Display the Korean name "{name}" prominently at the bottom.
        - Add their English name directly below.
    5. **Stats Section (Data Injection)**:
        - Title: "YIPP PRO" (Update from Rookie).
        - Position: "{position}".
        - **Radar Chart**:
            - Fill the radar chart polygon based on these values (0-100 scale):
                - Power (거래금액): {radar_power}
                - Defense (안정성): {radar_defense}
                - Contact (분산투자): {radar_contact}
                - Speed (거래빈도): {radar_speed}
                - Global (해외비중): {radar_global}
        - **Stats Values**:
            - AVG: {p_avg}
            - OPS: {p_ops}
            - ERA: {p_era}
    
    [Output Requirement]
    - Output ONLY the generated image.
    - Aspect Ratio: 9:16 (Vertical).
    - High quality, infographic style.
    - Ensure Korean text is legible.
    """

    parts = [types.Part.from_text(text=prompt_text)]
    ref_bytes = load_reference_bytes()
    
    if ref_bytes:
        parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
    else:
        st.warning(f"⚠️ {REFERENCE_IMAGE_PATH} 파일을 찾을 수 없어 텍스트로만 요청합니다.")

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"], 
        image_config=types.ImageConfig(image_size="1K")
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=generate_content_config,
        )

        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    raw_data = part.inline_data.data
                    try:
                        Image.open(BytesIO(raw_data)).verify()
                        return raw_data
                    except Exception:
                        pass
                    try:
                        decoded_data = base64.b64decode(raw_data)
                        Image.open(BytesIO(decoded_data)).verify()
                        return decoded_data
                    except Exception as e:
                        print(f"이미지 디코딩 실패: {e}")
                        continue

        raise Exception("모델 응답에서 유효한 이미지 데이터를 추출하지 못했습니다.")

    except Exception as e:
        st.error(f"❌ 이미지 생성 실패: {e}")
        fallback = Image.new('RGB', (540, 960), color=(50, 50, 80))
        buf = BytesIO()
        fallback.save(buf, format="PNG")
        return buf.getvalue()


# -----------------------------
# UI 단계별 함수
# -----------------------------

def step_login():
    st.header("① 내 선수 정보 입력")
    st.write("현재까지 투자 내역을 바탕으로 내 선수 카드를 업데이트합니다.")

    # CSS 적용 (민트색 버튼)
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: {THEME_COLOR} !important;
        border: none !important;
        color: white !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{
        background-color: {THEME_COLOR} !important;
        opacity: 0.9;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 1. 이름 입력
    name = st.text_input("선수 이름 입력", value=st.session_state["player_name"], placeholder="이름을 입력하세요")
    st.session_state["player_name"] = name

    # 2. 계좌번호 입력
    st.markdown("---")
    account = st.text_input("YIPP 계좌번호 입력 (12자리)", value=st.session_state["account"], max_chars=12, placeholder="숫자만 입력해주세요")
    st.session_state["account"] = account

    # 유효성 검사 (길이 및 숫자 여부)
    is_valid_name = len(name.strip()) > 0
    is_valid_length = len(account) == 12
    is_numeric = account.isdigit()

    if account and (not is_numeric or not is_valid_length):
         st.markdown(f":red[❌ YIPP 계좌번호는 12자리입니다.]")

    st.markdown("<br>", unsafe_allow_html=True)

    # 로그인/조회 버튼
    if st.button("내 카드 확인하기", type="primary", use_container_width=True, disabled=not(is_valid_name and is_valid_length and is_numeric)):
        
        # customer.csv 조회 로직
        is_registered, row_data = validate_user(name, account)
        
        if is_registered:
            # 데이터가 있으면 세션에 저장
            st.session_state["player_data"] = row_data
            
            # [수정] CSV의 '팀' 컬럼에서 구단 정보 가져오기 (랜덤 할당 로직 제거)
            fetched_team = row_data.get('팀', None)
            
            if fetched_team and str(fetched_team).lower() != 'nan' and str(fetched_team).strip() != "":
                # CSV에 있는 팀 이름을 그대로 사용
                st.session_state["team"] = str(fetched_team).strip()
            else:
                # 팀 정보가 없을 경우 기본값 할당 (예: SSG 랜더스)
                st.session_state["team"] = "SSG 랜더스"
            
            st.session_state["number"] = account[-2:] # 계좌번호 뒤 2자리
            
            # 스탯 기반 포지션 재산정
            new_position = determine_position(row_data)
            st.session_state["position"] = new_position
            
            st.success(f"환영합니다, {name} 선수! ({st.session_state['team']})\n업데이트된 투자 내역을 불러오는 중입니다...")
            time.sleep(1) 
            go_next_step()
            st.rerun()
        else:
            # 데이터가 없으면 에러 메시지
            st.error("등록되지 않은 선수입니다. YIPP 계좌 개설 후, 신인 선수 등록을 먼저 진행해주세요.")

def step_result():
    st.header("🏅 나의 선수 카드 (업데이트)")

    # 세션 데이터 가져오기
    data = st.session_state["player_data"]
    team = st.session_state["team"]
    num = st.session_state["number"]
    name = st.session_state["player_name"]
    pos = st.session_state["position"]

    # 버튼 스타일 복구
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: {THEME_COLOR} !important;
        color: white !important;
        border: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 텍스트 정보 표시
    st.subheader(f"{team} | No.{num} | {name} | {pos}")
    
    # 디버깅용: 실제 데이터 확인 (접을 수 있음)
    with st.expander("📊 내 상세 투자 내역 확인하기"):
        st.write(f"**AVG (수익률)**: {data.get('AVG(수익률)', '-')}")
        st.write(f"**OPS (활동성)**: {data.get('OPS(활동성)', '-')}")
        st.write(f"**ERA (안정성)**: {data.get('ERA(안정성)', '-')}")
        st.write(f"거래금액 {data.get('거래금액',0)}점 | 안정성 {data.get('안정성_점수',0)}점 | 분산투자 {data.get('분산투자',0)}점 | 거래빈도 {data.get('거래빈도',0)}점 | 해외비중 {data.get('해외비중',0)}점")

    status_container = st.empty()

    # 이미지 생성
    if st.session_state["card_image_bytes"] is None:
        status_container.info(f"🎨 {name}님의 투자 내역을 분석하여 선수 카드를 업데이트 중입니다...")
        
        # 실제 데이터와 CSV에서 가져온 팀 정보를 넘겨서 이미지 생성
        img_bytes = generate_updated_card_gemini(team, pos, num, name, data)
        st.session_state["card_image_bytes"] = img_bytes

    if st.session_state["card_image_bytes"]:
        status_container.info("🎊 업데이트 완료!")
        try:
            img = Image.open(BytesIO(st.session_state["card_image_bytes"]))
            st.image(img, use_container_width=True)
            
            st.download_button(
                label="📸 내 카드 공유하기",
                data=st.session_state["card_image_bytes"],
                file_name=f"yipp_pro_card_{num}.png",
                mime="image/png",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error("이미지를 표시할 수 없습니다.")
            st.error(e)

    col1, col2 = st.columns(2)
    col1.button("뒤로", on_click=go_prev_step, type="secondary", use_container_width=True)
    col2.button("처음으로", on_click=reset_all, type="secondary", use_container_width=True)


# -----------------------------
# 네비게이션
# -----------------------------
def go_next_step():
    st.session_state["step"] += 1

def go_prev_step():
    st.session_state["step"] = max(1, st.session_state["step"] - 1)

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session_state()


# -----------------------------
# 메인 실행 루프
# -----------------------------
def main():
    st.title("YIPP X KBO 내 선수 카드 업데이트")
    
    step = st.session_state["step"]
    
    if step == 1:
        step_login()
    elif step == 2:
        step_result()

if __name__ == "__main__":
    main()