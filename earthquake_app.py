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
import time

# -------------------------------
# 1. 데이터 수집 (USGS API)
# -------------------------------
@st.cache_data(ttl=3600)  # 1시간 캐시
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
        # 결측치 제거 (규모 없는 경우 제외)
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
    
    # 클러스터링에 사용할 피처
    features = df[["위도", "경도", "규모"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    # 3개 그룹으로 클러스터링
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)
    
    # 각 클러스터의 평균 규모 계산
    cluster_mag = df.groupby("cluster")["규모"].mean().sort_values()
    # 평균 규모가 가장 큰 클러스터 -> 위험 높음 (0)
    # 평균 규모가 가장 작은 클러스터 -> 위험 낮음 (1)
    # 나머지 -> 위험 중간 (2)
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
    # 지도 중심 (사용자 좌표 또는 전체 평균)
    if user_lat is not None and user_lon is not None:
        map_center = [user_lat, user_lon]
        zoom_start = 6
    else:
        map_center = [df["위도"].mean(), df["경도"].mean()] if not df.empty else [0, 0]
        zoom_start = 2
    
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")
    
    # 지진 데이터가 너무 많으면 성능을 위해 샘플링
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
    
    # 사용자 위치 마커 추가
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
    """지정된 반경(위도/경도 각도) 내 지진의 주요 클러스터로 위험도 결정"""
    if df.empty:
        return "데이터 없음", 0
    
    nearby = df[
        (df["위도"] >= lat - radius_deg) & (df["위도"] <= lat + radius_deg) &
        (df["경도"] >= lon - radius_deg) & (df["경도"] <= lon + radius_deg)
    ]
    
    if nearby.empty:
        return "정보 부족 (주변 2° 내 지진 없음)", 0
    
    # 주변 지진들의 위험도 분포
    risk_counts = nearby["위험도"].value_counts()
    main_risk = risk_counts.idxmax() if not risk_counts.empty else "정보 부족"
    count = len(nearby)
    
    # 상세 정보 표시용 문자열
    details = f"반경 {radius_deg}° 내 지진 {count}개 발견\n"
    details += f"위험도 분포: {risk_counts.to_dict()}"
    return main_risk, count, details

# -------------------------------
# 5. Streamlit 앱 메인
# -------------------------------
st.set_page_config(page_title="지진 분석 대시보드", layout="wide")
st.title("🌍 실시간 지진 분석 대시보드")
st.markdown("USGS 지진 카탈로그(최근 30일) 데이터를 기반으로 지진 위험도를 분석합니다.")

# 데이터 로드
with st.spinner("USGS에서 최근 30일 지진 데이터를 불러오는 중..."):
    df_raw = fetch_earthquake_data(days=30)

if df_raw.empty:
    st.stop()

df = assign_risk_clusters(df_raw)

# -------------------------------
# 사이드바: 통계 및 필터
# -------------------------------
st.sidebar.header("📊 통계 요약")
total_eq = len(df)
max_mag = df["규모"].max()
avg_mag = df["규모"].mean()
strong_eq = df[df["규모"] >= 5.0].shape[0]  # 규모 5 이상을 강진으로 정의

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 지진 횟수", f"{total_eq:,}")
col2.metric("최대 규모", f"{max_mag:.1f}")
col3.metric("평균 규모", f"{avg_mag:.2f}")
col4.metric("강진 횟수 (M≥5)", strong_eq)

# 지도 표시
st.subheader("🗺️ 지진 분포 지도 (클러스터별 색상)")

# 사용자 입력 위치
st.sidebar.subheader("📍 위치 위험도 분석")
user_lat = st.sidebar.number_input("위도", value=36.5, format="%.4f")
user_lon = st.sidebar.number_input("경도", value=127.5, format="%.4f")
radius_deg = st.sidebar.slider("분석 반경 (도)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

if st.sidebar.button("이 위치 분석하기"):
    risk, cnt, details = analyze_location_risk(df, user_lat, user_lon, radius_deg)
    st.sidebar.success(f"**위험도 판정: {risk}**")
    st.sidebar.info(f"주변 {cnt}개 지진 기반")
    with st.sidebar.expander("세부 정보"):
        st.text(details)

# 지도 생성 및 표시
map_obj = create_earthquake_map(df, user_lat if st.sidebar.button("지도에 위치 표시", key="map_btn") else None, 
                                 user_lon if st.sidebar.button("지도에 위치 표시", key="map_btn_lon") else None)
folium_static(map_obj, width=1000, height=600)

# 데이터 테이블
st.subheader("📋 상세 지진 목록")
# 필요한 컬럼만 선택 및 한글 순서 정리
display_df = df[["시간", "위도", "경도", "규모", "장소", "위험도"]].copy()
display_df = display_df.sort_values("시간", ascending=False)

# 페이지네이션 (Streamlit 기본 사용)
page_size = 20
total_pages = len(display_df) // page_size + 1
page_num = st.number_input("페이지", min_value=1, max_value=total_pages, value=1)
start_idx = (page_num - 1) * page_size
end_idx = start_idx + page_size
st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)

st.caption(f"총 {len(display_df)}개 지진 기록 (최근 30일, USGS 기준)")

# 추가 정보: 강진 경고
if strong_eq > 0:
    st.warning(f"⚠️ 지난 30일 동안 규모 5.0 이상의 강진이 {strong_eq}회 발생했습니다.")
