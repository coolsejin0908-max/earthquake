# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

# -------------------------------
# 0. 한글 폰트 설정 (Nanum)
# -------------------------------
def setup_korean_font():
    # 시스템에 NanumGothic이 설치되어 있다고 가정 (packages.txt에 fonts-nanum 추가)
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_font = [f for f in font_path if 'NanumGothic' in f or 'Nanum' in f]
    if nanum_font:
        plt.rcParams['font.family'] = fm.FontProperties(fname=nanum_font[0]).get_name()
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # fallback
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# -------------------------------
# 1. 데이터 수집 (USGS API)
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_earthquake_data(days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": 0,
        "orderby": "time"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        earthquakes = []
        for feature in data["features"]:
            coords = feature["geometry"]["coordinates"]
            props = feature["properties"]
            earthquakes.append({
                "위도": coords[1],
                "경도": coords[0],
                "깊이_km": coords[2],
                "규모": props["mag"],
                "장소": props["place"],
                "시간": datetime.fromtimestamp(props["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(earthquakes)
        df = df.dropna(subset=["규모", "위도", "경도"])
        return df
    except Exception as e:
        st.error(f"USGS 데이터 수집 실패: {e}")
        return pd.DataFrame()

# -------------------------------
# 2. 클러스터링 (위험도 그룹화)
# -------------------------------
def assign_risk_clusters(df):
    if df.empty:
        return df
    
    features = df[["위도", "경도", "규모"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)
    
    cluster_mag = df.groupby("cluster")["규모"].mean().sort_values()
    high_risk_cluster = cluster_mag.idxmax()
    low_risk_cluster = cluster_mag.idxmin()
    medium_risk_cluster = cluster_mag.index[~cluster_mag.index.isin([high_risk_cluster, low_risk_cluster])][0]
    
    risk_map = {
        high_risk_cluster: "높음",
        low_risk_cluster: "낮음",
        medium_risk_cluster: "중간"
    }
    color_map = {
        "높음": "red",
        "낮음": "blue",
        "중간": "green"
    }
    
    df["위험도"] = df["cluster"].map(risk_map)
    df["색상"] = df["위험도"].map(color_map)
    return df

# -------------------------------
# 3. Folium 지도 생성
# -------------------------------
def create_earthquake_map(df, user_lat=None, user_lon=None):
    if not df.empty and user_lat is not None and user_lon is not None:
        map_center = [user_lat, user_lon]
        zoom_start = 6
    elif not df.empty:
        map_center = [df["위도"].mean(), df["경도"].mean()]
        zoom_start = 2
    else:
        map_center = [0, 0]
        zoom_start = 2
    
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")
    
    sample_df = df if len(df) <= 5000 else df.sample(5000, random_state=42)
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=3,
            color=row["색상"],
            fill=True,
            fill_color=row["색상"],
            fill_opacity=0.7,
            popup=f"규모: {row['규모']}<br>위치: {row['장소']}<br>시간: {row['시간']}"
        ).add_to(m)
    
    if user_lat is not None and user_lon is not None:
        folium.Marker(
            location=[user_lat, user_lon],
            icon=folium.Icon(color="black", icon="star", prefix="fa"),
            popup="선택한 위치"
        ).add_to(m)
    
    return m

# -------------------------------
# 4. 선택한 위치 주변 위험도 분석
# -------------------------------
def analyze_location_risk(df, lat, lon, radius_deg=2):
    if df.empty:
        return "데이터 없음", 0, "데이터가 없습니다."
    
    nearby = df[
        (df["위도"] >= lat - radius_deg) & (df["위도"] <= lat + radius_deg) &
        (df["경도"] >= lon - radius_deg) & (df["경도"] <= lon + radius_deg)
    ]
    
    if nearby.empty:
        return "정보 부족", 0, f"반경 {radius_deg}° 내 지진 없음"
    
    risk_counts = nearby["위험도"].value_counts()
    main_risk = risk_counts.idxmax()
    count = len(nearby)
    details = f"반경 {radius_deg}° 내 지진 {count}개\n위험도 분포: {risk_counts.to_dict()}"
    return main_risk, count, details

# -------------------------------
# 5. 그래프 그리기 (한글 지원)
# -------------------------------
def plot_magnitude_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["규모"], bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("지진 규모 (Magnitude)")
    ax.set_ylabel("빈도수")
    ax.set_title("최근 30일 지진 규모 분포")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def plot_daily_counts(df):
    df["날짜"] = pd.to_datetime(df["시간"]).dt.date
    daily = df.groupby("날짜").size().reset_index(name="횟수")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["날짜"], daily["횟수"], marker='o', linestyle='-', color='coral')
    ax.set_xlabel("날짜")
    ax.set_ylabel("지진 발생 횟수")
    ax.set_title("일별 지진 발생 추이")
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# -------------------------------
# 6. Streamlit 앱 메인
# -------------------------------
st.set_page_config(page_title="지진 분석 대시보드", layout="wide")
st.title("🌍 실시간 지진 분석 대시보드")
st.markdown("USGS 지진 카탈로그(최근 30일) 데이터 기반 위험도 분석")

# 데이터 로드
with st.spinner("USGS에서 최근 30일 지진 데이터를 불러오는 중..."):
    df_raw = fetch_earthquake_data(days=30)

if df_raw.empty:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

df = assign_risk_clusters(df_raw)

# -------------------------------
# 사이드바: 통계 요약
# -------------------------------
st.sidebar.header("📊 통계 요약")
total_eq = len(df)
max_mag = df["규모"].max()
avg_mag = df["규모"].mean()
strong_eq = df[df["규모"] >= 5.0].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 지진 횟수", f"{total_eq:,}")
col2.metric("최대 규모", f"{max_mag:.1f}")
col3.metric("평균 규모", f"{avg_mag:.2f}")
col4.metric("강진 횟수 (M≥5)", strong_eq)

# -------------------------------
# 지도 표시
# -------------------------------
st.subheader("🗺️ 지진 분포 지도 (위험도별 색상: 빨강=높음, 초록=중간, 파랑=낮음)")

# 사용자 입력 및 위치 분석 (사이드바)
st.sidebar.subheader("📍 위치 위험도 분석")
user_lat = st.sidebar.number_input("위도", value=36.5, format="%.4f")
user_lon = st.sidebar.number_input("경도", value=127.5, format="%.4f")
radius_deg = st.sidebar.slider("분석 반경 (도)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# 분석 버튼
if st.sidebar.button("이 위치 분석하기"):
    risk, cnt, details = analyze_location_risk(df, user_lat, user_lon, radius_deg)
    st.sidebar.success(f"**위험도 판정: {risk}**")
    st.sidebar.info(f"주변 지진 {cnt}개 기반")
    with st.sidebar.expander("세부 정보"):
        st.text(details)

# 지도에 사용자 위치 표시 체크박스
show_user_marker = st.sidebar.checkbox("지도에 내 위치 표시", value=True)
if show_user_marker:
    map_obj = create_earthquake_map(df, user_lat, user_lon)
else:
    map_obj = create_earthquake_map(df, None, None)

folium_static(map_obj, width=1000, height=600)

# -------------------------------
# 그래프 섹션 (한글)
# -------------------------------
st.subheader("📈 지진 통계 그래프")
graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.pyplot(plot_magnitude_histogram(df))
with graph_col2:
    st.pyplot(plot_daily_counts(df))

# -------------------------------
# 데이터 테이블
# -------------------------------
st.subheader("📋 상세 지진 목록")
display_df = df[["시간", "위도", "경도", "규모", "장소", "위험도"]].copy()
display_df = display_df.sort_values("시간", ascending=False)

page_size = 20
total_pages = len(display_df) // page_size + 1
page_num = st.number_input("페이지", min_value=1, max_value=total_pages, value=1)
start_idx = (page_num - 1) * page_size
end_idx = start_idx + page_size
st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)

st.caption(f"총 {len(display_df)}개 지진 기록 (최근 30일, USGS 기준)")

if strong_eq > 0:
    st.warning(f"⚠️ 지난 30일 동안 규모 5.0 이상의 강진이 {strong_eq}회 발생했습니다.")
