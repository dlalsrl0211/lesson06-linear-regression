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


# --------------------------------------------------
# 전체 기간 회귀 분석
# --------------------------------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = yearly["연도"].corr(yearly["연평균기온"])

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())
사용연도수 = len(yearly)
전체100년변화량 = 기울기 * 100


def predict_temperature(year):
    """전체 분석 기간의 회귀 직선으로 특정 연도의 기온을 추정합니다."""
    return 기울기 * year + 절편


# --------------------------------------------------
# 최근 20년 회귀 분석
# 기준: 분석 데이터에서 가장 최근 연도
# --------------------------------------------------
최근20년시작 = 끝연도 - 19
recent = yearly[
    (yearly["연도"] >= 최근20년시작) &
    (yearly["연도"] <= 끝연도)
].copy()

if len(recent) >= 2:
    최근기울기, 최근절편 = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    최근상관계수 = recent["연도"].corr(recent["연평균기온"])
    최근100년변화량 = 최근기울기 * 100
else:
    최근기울기 = None
    최근절편 = None
    최근상관계수 = None
    최근100년변화량 = None


# --------------------------------------------------
# 분석 기간 정보
# --------------------------------------------------
st.subheader("📊 회귀 분석에 사용한 데이터")

c1, c2, c3 = st.columns(3)
c1.metric("사용한 연도 수", f"{사용연도수:,}개")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")

st.caption(
    f"분석 조건: {기준연도}년까지의 자료 중 연간 관측일수가 "
    f"{최소관측일수}일 이상인 연도만 사용합니다."
)


# --------------------------------------------------
# 100년당 기온 변화량 비교
# --------------------------------------------------
st.divider()
st.subheader("🌡️ 100년당 기온 변화량 비교")

left, right = st.columns(2)

with left:
    st.markdown("#### 전체 분석 기간")
    st.metric(
        label=f"{시작연도}~{끝연도}년",
        value=f"{전체100년변화량:+.2f} °C",
        help="전체 분석 기간의 회귀 기울기를 100배 한 값입니다.",
    )
    st.caption(f"회귀 분석에 사용한 연도: {사용연도수}개")

with right:
    st.markdown("#### 최근 20년 범위")
    if 최근기울기 is not None:
        st.metric(
            label=f"{최근20년시작}~{끝연도}년",
            value=f"{최근100년변화량:+.2f} °C",
            help="최근 20년 범위의 회귀 기울기를 100배 한 값입니다.",
        )
        st.caption(f"해당 범위에서 조건을 통과한 연도: {len(recent)}개")
    else:
        st.warning("최근 20년 범위에 회귀 분석 가능한 연도가 부족합니다.")

if 최근기울기 is not None:
    차이 = 최근100년변화량 - 전체100년변화량
    st.write(
        f"**두 변화량의 차이:** 최근 20년의 100년당 변화량은 "
        f"전체 기간보다 {차이:+.2f}°C입니다."
    )
