import streamlit as st
import cv2
import time
import tempfile
import os
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

st.set_page_config(page_title="Marine Danger Alert System", layout="wide")

# =========================================================
# 1. 전역 상수 및 설정 (모델 경로 반영)
# =========================================================
MODEL_PATH = "best_openvino_model"

CLASS_RISK_WEIGHT = {
    "shark":     1.0, 
    "jellyfish": 0.85, 
    "stingray":  0.6,
    "puffin":    0.1, 
    "penguin":   0.1,  
    "fish":      0.15, 
    "starfish":  0.05,
}
DEFAULT_CLASS_WEIGHT = 0.2
HISTORY_LEN = 15
MIN_POINTS_FOR_SLOPE = 5
CLASS_WEIGHT_RATIO = 0.45
GROWTH_WEIGHT_RATIO = 0.55
GROWTH_RATE_SATURATION = 0.8

LEVEL_THRESHOLDS = {"위험": 0.75, "경고": 0.5, "주의": 0.25}

@st.cache_resource
def load_openvino_model(model_dir_path):
    return YOLO(model_dir_path, task="detect")

def bbox_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def linear_regression_slope(times, values):
    n = len(times)
    if n < 2: 
        return 0.0
    t_mean, v_mean = sum(times) / n, sum(values) / n
    numerator = sum((t - t_mean) * (v - v_mean) for t, v in zip(times, values))
    denominator = sum((t - t_mean) ** 2 for t in times)
    return numerator / denominator if denominator != 0 else 0.0

def normalized_growth_rate(times, values):
    slope = linear_regression_slope(times, values)
    base_area = max(values[0], 1e-6)
    return slope / base_area

def growth_rate_to_risk(relative_rate):
    if relative_rate <= 0: 
        return 0.0
    return min(1.0, relative_rate / GROWTH_RATE_SATURATION)

def classify_level(score):
    if score >= LEVEL_THRESHOLDS["위험"]:   
        return "위험", (255, 0, 0)      
    elif score >= LEVEL_THRESHOLDS["경고"]: 
        return "경고", (255, 128, 0)    
    elif score >= LEVEL_THRESHOLDS["주의"]: 
        return "주의", (255, 255, 0)    
    else:                                 
        return "안전", (0, 255, 0)

# =========================================================
# 2. UI 제어 콜백 함수 (상태 변경 및 이력 초기화)
# =========================================================
def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing

def skip_backward():
    fps = st.session_state.get('video_fps', 30)
    st.session_state.current_frame = max(0, st.session_state.current_frame - int(fps * 5))
    st.session_state.history.clear()

def skip_forward():
    fps = st.session_state.get('video_fps', 30)
    tot = st.session_state.get('total_frames', 0)
    st.session_state.current_frame = min(tot - 1, st.session_state.current_frame + int(fps * 5))
    st.session_state.history.clear()

def go_first():
    st.session_state.current_frame = 0
    st.session_state.is_playing = False
    st.session_state.history.clear()

def slider_changed():
    st.session_state.current_frame = st.session_state.slider_val
    st.session_state.history.clear()

# =========================================================
# 3. 데이터 프로세싱 파이프라인
# =========================================================
def process_frame(frame, model, names, conf_thres):
    now = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(frame_rgb, persist=True, conf=conf_thres, verbose=False)
    
    current_danger_level = "안전"
    detected_count = 0

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes
        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy().astype(int)
        id_list = boxes.id.cpu().numpy().astype(int)
        detected_count = len(id_list)

        for xyxy, cls_id, track_id in zip(xyxy_list, cls_list, id_list):
            class_name = names.get(int(cls_id), str(cls_id))
            area = bbox_area(xyxy)
            st.session_state.history[track_id].append((now, area))

            pts = st.session_state.history[track_id]
            if len(pts) >= MIN_POINTS_FOR_SLOPE:
                times = [p[0] for p in pts]
                areas = [p[1] for p in pts]
                rel_rate = normalized_growth_rate(times, areas)
            else:
                rel_rate = 0.0

            growth_risk = growth_rate_to_risk(rel_rate)
            class_risk = CLASS_RISK_WEIGHT.get(class_name, DEFAULT_CLASS_WEIGHT)

            final_score = (CLASS_WEIGHT_RATIO * class_risk + GROWTH_WEIGHT_RATIO * growth_risk)
            level, color = classify_level(final_score)
            
            if final_score >= LEVEL_THRESHOLDS["주의"]:
                current_danger_level = level

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 3)
            
            label = f"ID{track_id} {class_name} [{level}] Rate:{rel_rate*100:.0f}%/s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_rgb, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            text_color = (0, 0, 0) if level == "주의" else (255, 255, 255)
            cv2.putText(frame_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
    return frame_rgb, current_danger_level, detected_count

# =========================================================
# 4. 코어 렌더링 루프
# =========================================================
if "current_frame" not in st.session_state: st.session_state.current_frame = 0
if "is_playing" not in st.session_state: st.session_state.is_playing = False
if "total_frames" not in st.session_state: st.session_state.total_frames = 0
if "video_fps" not in st.session_state: st.session_state.video_fps = 30
if "history" not in st.session_state: st.session_state.history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
if "last_video_source" not in st.session_state: st.session_state.last_video_source = None

def main():
    with st.sidebar:
        st.header("시스템 설정")
        conf_thres = st.slider("탐지 임계값 (Confidence)", 0.0, 1.0, 0.6, 0.05)
        st.markdown("---")
        
        # 비디오 소스 선택 라디오 버튼
        st.header("📹 비디오 소스 선택")
        video_source = st.radio(
            "테스트 영상을 선택하세요:",
            ("직접 업로드", "샘플 영상: 해파리", "샘플 영상: 상어", "샘플 영상: 다이버 뷰")
        )

    st.title("수중 위험 감지 시스템 (OpenVINO 가속)")
    
    video_path = None

    # 사용자의 소스 선택에 따른 경로 할당
    if video_source == "직접 업로드":
        uploaded_video = st.file_uploader("분석할 영상을 업로드하십시오.", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()
            video_path = tfile.name
    else:
        sample_map = {
            "샘플 영상: 해파리": "samples/sample_jellyfish.mp4",
            "샘플 영상: 상어": "samples/sample_shark.mp4",
            "샘플 영상: 다이버 뷰": "samples/sample_diver.mp4"
        }
        selected_sample = sample_map[video_source]
        if os.path.exists(selected_sample):
            video_path = selected_sample
            st.info(f"✅ 기본 샘플 영상이 로드되었습니다: {video_source}")
        else:
            st.error(f"샘플 영상 파일을 찾을 수 없습니다. (경로: {selected_sample})")

    # 영상 소스가 선택되었을 때만 분석 실행
    if video_path is not None:
        try:
            model = load_openvino_model(MODEL_PATH)
            names = model.names
        except Exception as e:
            st.error(f"모델 로드 실패. 내부 경로에 best_openvino_model 폴더가 존재하는지 확인하십시오. 에러 내용: {str(e)}")
            return

        # 비디오 소스가 변경되었을 경우 초기화
        if st.session_state.last_video_source != video_path:
            st.session_state.last_video_source = video_path
            st.session_state.current_frame = 0
            st.session_state.is_playing = False
            st.session_state.history.clear()
            
            cap_init = cv2.VideoCapture(video_path)
            st.session_state.total_frames = int(cap_init.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap_init.get(cv2.CAP_PROP_FPS)
            st.session_state.video_fps = fps if fps > 0 else 30
            cap_init.release()

        cap = cv2.VideoCapture(video_path)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stframe = st.empty()
            
            # 제어 컨트롤러 배정
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
            with ctrl_col1: st.button("5초 뒤로", on_click=skip_backward, use_container_width=True)
            with ctrl_col2: 
                play_label = "일시정지" if st.session_state.is_playing else "재생"
                st.button(play_label, on_click=toggle_play, use_container_width=True)
            with ctrl_col3: st.button("5초 앞으로", on_click=skip_forward, use_container_width=True)
            with ctrl_col4: st.button("처음으로", on_click=go_first, use_container_width=True)

            # 타임라인 제어 슬라이더
            st.slider(
                "타임라인", 
                min_value=0, 
                max_value=max(1, st.session_state.total_frames - 1), 
                value=st.session_state.current_frame,
                step=1,
                key="slider_val",
                on_change=slider_changed,
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("### 실시간 관제 상태")
            status_text = st.empty()

        # 핵심 연산 로직 영역
        if st.session_state.is_playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            
            # 외부 rerun 호출을 전면 배제하고 내부 루프로만 전진시켜 깜빡임을 방지합니다.
            while cap.isOpened() and st.session_state.is_playing:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.is_playing = False
                    st.session_state.current_frame = 0
                    break
                    
                st.session_state.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_rgb, lvl, cnt = process_frame(frame, model, names, conf_thres)
                
                # 동일 컴포넌트에 바이너리만 인플레이스로 주입하여 드라이브 렉을 최소화합니다.
                stframe.image(frame_rgb, use_container_width=True)
                status_text.markdown(f"""
                - **현재 위험도 정보:** {lvl}
                - **감지된 객체 수:** {cnt} 개
                - **재생 위치:** {st.session_state.current_frame} / {st.session_state.total_frames - 1} 프레임
                """)
                
                # CPU 루프 속도를 제어하여 정상 배속을 유지시킵니다.
                time.sleep(0.01)
        else:
            # 일시정지 상태이거나 조작이 가해졌을 때는 지정된 인덱스의 한 페이지만 렌더링하고 대기합니다.
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            if ret:
                frame_rgb, lvl, cnt = process_frame(frame, model, names, conf_thres)
                stframe.image(frame_rgb, use_container_width=True)
                status_text.markdown(f"""
                - **현재 위험도 정보:** {lvl}
                - **감지된 객체 수:** {cnt} 개
                - **재생 위치:** {st.session_state.current_frame} / {st.session_state.total_frames - 1} 프레임
                """)

        cap.release()

if __name__ == "__main__":
    main()