import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, 선형 회귀로 특정 연도의 "
    "기온을 추정합니다. 전체 기간과 최근 20년의 기온 변화 추세도 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일수 = 300


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing))}"
        )

    # 날짜 형식이 YYYYMMDD 또는 일반 날짜 문자열인 경우 모두 처리
    date_text = df["날짜"].astype("string").str.strip()
    yyyymmdd = date_text.str.fullmatch(r"\d{8}", na=False)

    parsed_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    parsed_dates.loc[yyyymmdd] = pd.to_datetime(
        date_text.loc[yyyymmdd], format="%Y%m%d", errors="coerce"
    )
    parsed_dates.loc[~yyyymmdd] = pd.to_datetime(
        date_text.loc[~yyyymmdd], errors="coerce"
    )

    df["날짜"] = parsed_dates
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 수업 기준인 2025년 이후 자료 제외
    df = df[df["연도"] <= 기준연도]
    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"기온 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("인터넷 연결, 데이터 URL, CSV의 열 이름을 확인해 주세요.")
    st.stop()


# --------------------------------------------------
# 연도별 평균기온과 관측일수 계산
# --------------------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("날짜", "nunique"),
    )
    .reset_index()
)

# 관측일수가 300일 미만인 연도 제외
yearly = yearly[yearly["관측일수"] >= 최소관측일수].copy()
yearly = yearly.sort_values("연도").reset_index(drop=True)

if len(yearly) < 2:
    st.error(
        "조건을 만족하는 연도가 2개 미만이라 회귀 분석을 할 수 없습니다. "
        "데이터와 관측일수 조건을 확인해 주세요."
    )
    st.stop()

