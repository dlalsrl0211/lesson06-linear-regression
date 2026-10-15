
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형 회귀로 선택한 연도의 기온을 추정합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일수 = 300


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing))}"
        )

    # 날짜와 기온 자료 정리
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 연도 이후 자료 제외
    df = df[df["연도"] <= 기준연도]

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"기온 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("데이터 URL과 CSV의 열 이름을 확인해 주세요.")
    st.stop()


# -----------------------------
# 연도별 평균기온 및 관측일수 계산
# -----------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("날짜", "nunique")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 연도 제외
yearly = yearly[
    yearly["관측일수"] >= 최소관측일수
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

if len(yearly) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 2개 미만입니다. "
        "데이터와 관측일수 조건을 확인해 주세요."
    )
    st.stop()


# -----------------------------
# 선형 회귀 및 상관계수
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = yearly["연도"].corr(yearly["연평균기온"])

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())
사용연도수 = len(yearly)


def predict_temperature(year):
    """회귀 직선으로 특정 연도의 평균기온 추정."""
    return 기울기 * year + 절편


# -----------------------------
# 주요 분석 정보
# -----------------------------
st.subheader("📊 회귀 분석에 사용한 데이터")

c1, c2, c3 = st.columns(3)

c1.metric("사용한 연도 수", f"{사용연도수:,}개")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")

st.caption(
    f"분석 조건: {기준연도}년까지의 자료 중 "
    f"연간 관측일수 {최소관측일수}일 이상인 연도만 사용합니다."
)

# -----------------------------
# 연도 선택 및 예측
# -----------------------------
st.divider()
st.subheader("🌡️ 몇 년도의 기온이 궁금한가요?")

선택연도 = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

예상기온 = predict_temperature(선택연도)

left, right = st.columns([1, 2])

with left:
    st.markdown(f"### {선택연도}년 예상 연평균기온")
    st.markdown(
        f"<div style='font-size: 48px; font-weight: bold;'>"
        f"{예상기온:.2f} °C</div>",
        unsafe_allow_html=True
    )
    st.caption("선형 회귀 모델의 추정값")

with right:
    st.markdown("#### 회귀식")
    st.latex(
        rf"\hat{{y}} = {기울기:.5f}x "
        rf"{절편:+.2f}"
    )
    st.write(
        f"연도가 1년 증가할 때 회귀 모델의 추정 기온은 "
        f"약 {기울기:.4f}°C 변합니다."
    )


# -----------------------------
# 산점도 + 회귀 직선
# -----------------------------
st.divider()
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}°C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선: 1900~2100년
line_x = np.array([1900, 2100])
line_y = predict_temperature(line_x)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 슬라이더로 선택한 연도 표시
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name="선택한 연도의 예상 기온",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 상관계수와 해석
# -----------------------------
st.subheader("🔎 상관계수")

m1, m2 = st.columns(2)
m1.metric("피어슨 상관계수", f"{상관계수:.4f}")
m2.metric("회귀 기울기", f"{기울기:.4f} °C/년")

if 상관계수 > 0:
    st.write(
        "연도와 연평균기온 사이에 양의 상관관계가 있습니다. "
        "분석 자료에서는 시간이 지날수록 평균기온이 높아지는 경향을 보입니다."
    )
elif 상관계수 < 0:
    st.write(
        "연도와 연평균기온 사이에 음의 상관관계가 있습니다. "
        "분석 자료에서는 시간이 지날수록 평균기온이 낮아지는 경향을 보입니다."
    )
else:
    st.write("연도와 연평균기온 사이에 선형 상관관계가 거의 없습니다.")

st.caption(
    "주의: 상관관계와 회귀 직선은 과거 자료의 통계적 경향을 나타냅니다. "
    "미래 기온은 기후 변화, 자연 변동 등 여러 요인의 영향을 받으므로 "
    "이 값은 실제 기온을 보장하지 않습니다."
)

# -----------------------------
# 연도별 분석 데이터 확인
# -----------------------------
with st.expander("연도별 분석 데이터 보기"):
    보여줄자료 = yearly.copy()
    보여줄자료["연평균기온"] = 보여줄자료["연평균기온"].round(2)
    보여줄자료.columns = ["연도", "연평균기온(°C)", "관측일수"]
    st.dataframe(보여줄자료, use_container_width=True, hide_index=True)
