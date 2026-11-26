"""공연 음악 장르별 성장 양상 대시보드

질문: 2019년 이후 공연 음악 장르(대중음악·클래식·국악)는 얼마나 성장했고, 그 성장은 관객 수와 가격 중 무엇에서 나왔는가?
자료: 예술경영지원센터 공연예술통합전산망(KOPIS) 예매통계 장르별통계 (월 단위로 조회해 합침)
      국가데이터처(통계청) 소비자물가지수 총지수(2020=100) 월별 — 물가 비교·물가 반영 계산용
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="공연 음악 장르별 성장 양상", layout="wide")

# ---------------------------------------------------------------- 설정
DATA_PATH = Path(__file__).parent / "data" / "kopis_genre_monthly.csv"
CPI_PATH = Path(__file__).parent / "data" / "cpi_monthly.csv"
DATA_DATE = "2026-11-26"       # 자료를 내려받은 날(KOPIS 최종집계 시점)
LAST_MONTH = "2026-10"         # 마지막으로 끝난 달 (2026-11은 11/25까지의 부분 자료라 제외)
BASE_YEAR = 2019               # 비교 기준 연도 (물가 반영 금액은 이 해의 평균 물가 기준 가격)

MUSIC = ["대중음악", "서양음악(클래식)", "한국음악(국악)"]
COMPARE = ["뮤지컬", "연극", "합계"]
LABEL = {"서양음악(클래식)": "클래식", "한국음악(국악)": "국악", "합계": "전체 공연"}
# 색: 검증된 범주형 팔레트의 앞 5색을 장르에 고정해서 배정(선택이 바뀌어도 같은 장르는 같은 색). 합계는 무채색.
COLORS = {"대중음악": "#2a78d6", "서양음악(클래식)": "#eb6834", "한국음악(국악)": "#1baf7a",
          "뮤지컬": "#eda100", "연극": "#e87ba4", "합계": "#9aa0a8"}
CPI_COLOR = "#6b7380"
NAVY = "#14284b"
INK = "#3a4150"
GRID = "#e8ebf0"
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
# 참고 구간(국내 첫 확진 ~ 사회적 거리두기 전면 해제). 설명용 음영이며 분석 기준이 아니다.
COVID_SPAN = ("2020-01-20", "2022-04-18")
GROWTH_VIEW = "성장 (2019년 = 100)"
LEVEL_VIEW = "실제 값 (월별)"


def name(g: str) -> str:
    return LABEL.get(g, g)


def md(text: str) -> str:
    """마크다운 문장용: ~ 사이의 글자에 취소선이 생기지 않도록 ~를 이스케이프."""
    return text.replace("~", "\\~")


def josa(word: str, with_batchim: str, without: str) -> str:
    """받침 유무에 따라 조사를 고른다(예: 은/는, 이/가, 으로/로)."""
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return with_batchim


def qlabel(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{ts.quarter}"


# ---------------------------------------------------------------- 데이터/계산
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[df["월"] <= LAST_MONTH].copy()
    df["날짜"] = pd.to_datetime(df["월"] + "-01")
    df["순예매액"] = df["총예매액_천원"] / 1e5     # 천원 → 억 원
    df["순예매수"] = df["총예매수"] / 1e4          # 건(장) → 만 장
    # 물가 반영: 월별 소비자물가지수를 기준 연도 평균으로 나눈 값(물가)으로 금액을 나눠 '기준 연도 가격'으로 환산
    cpi = pd.read_csv(CPI_PATH, encoding="utf-8-sig")
    cpi["날짜"] = pd.to_datetime(cpi["월"] + "-01")
    base = cpi.loc[cpi["월"].str[:4] == str(BASE_YEAR), "소비자물가지수"].mean()
    cpi["물가"] = cpi["소비자물가지수"] / base
    df = df.merge(cpi[["월", "물가"]], on="월", how="left")
    df["실질예매액"] = df["순예매액"] / df["물가"]
    price = cpi[cpi["월"] <= LAST_MONTH].set_index("날짜")["물가"].sort_index()
    return df, price


def monthly_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="날짜", columns="장르", values=metric).sort_index()


def to_quarterly(w: pd.DataFrame) -> pd.DataFrame:
    """3개월이 모두 있는 분기만 합산. 인덱스는 분기 첫 달."""
    per = w.index.to_period("Q")
    full = w.groupby(per).size()
    full = full[full == 3].index
    q = w.groupby(per).sum().loc[full]
    q.index = q.index.to_timestamp()
    return q


def ratio(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """장당 평균 가격 = 금액 ÷ 수량. 금액이나 수량이 0 이하인 달(분기)은 계산하지 않는다."""
    return (a / c.where(c > 0)).where(a > 0)


def last12(w, end_ts: pd.Timestamp):
    return w[(w.index <= end_ts) & (w.index > end_ts - pd.DateOffset(months=12))]


def rolling_index(w: pd.DataFrame, denom: pd.DataFrame = None) -> pd.DataFrame:
    """각 달까지 '최근 12개월 합계'가 BASE_YEAR 연간 합계의 몇 %인지(2019-12 이후).
    월 하나가 아니라 12개월 합계끼리 비교하므로 계절성과 기준 달 하나의 영향이 없다.
    denom(수량)을 주면 금액 ÷ 수량, 즉 장당 평균 가격의 지수가 된다."""
    base = w[w.index.year == BASE_YEAR]
    start = pd.Timestamp(f"{BASE_YEAR}-12-01")
    if len(base) < 12:
        return pd.DataFrame(np.nan, index=w.index[w.index >= start], columns=w.columns)
    roll = w.rolling(12, min_periods=12).sum()
    if denom is None:
        b = base.sum()
        out = roll / b.where(b > 0) * 100
    else:
        bd = denom[denom.index.year == BASE_YEAR].sum()
        rd = denom.rolling(12, min_periods=12).sum()
        bp = base.sum() / bd.where(bd > 0)
        out = (roll / rd.where(rd > 0)) / bp.where(bp > 0) * 100
    return out[out.index >= start]


def rolling_price_level(price: pd.Series) -> pd.Series:
    """소비자물가의 최근 12개월 평균이 기준 연도 평균의 몇 %인가(2019-12 이후)."""
    out = price.rolling(12, min_periods=12).mean() * 100
    return out[out.index >= pd.Timestamp(f"{BASE_YEAR}-12-01")]


def decompose(wa_n: pd.DataFrame, wa_r: pd.DataFrame, wc: pd.DataFrame, end_ts: pd.Timestamp):
    """종료 월까지 최근 12개월을 BASE_YEAR 12개월과 비교한 배수(기준 연도 = 1)와 그 구성.
    명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인  (곱셈으로 정확히 성립)
    물가 요인 = 명목 예매액 배수 ÷ 물가 반영 예매액 배수 (금액 가중 물가 상승분)."""
    base_n = wa_n[wa_n.index.year == BASE_YEAR]
    if len(base_n) < 12 or end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01"):
        return None
    cur_n, cur_r, cur_c = last12(wa_n, end_ts), last12(wa_r, end_ts), last12(wc, end_ts)
    if len(cur_n) < 12:
        return None
    b_n, b_r = base_n.sum(), wa_r[wa_r.index.year == BASE_YEAR].sum()
    b_c = wc[wc.index.year == BASE_YEAR].sum()
    c_n, c_r, c_c = cur_n.sum(), cur_r.sum(), cur_c.sum()
    ok = ((b_n > 0) & (b_r > 0) & (b_c > 0) & (c_n > 0) & (c_r > 0) & (c_c > 0)).reindex(wa_n.columns)
    A, R, Q = c_n / b_n, c_r / b_r, c_c / b_c
    out = pd.DataFrame({"명목 예매액": A, "예매액(물가 반영)": R, "예매수": Q,
                        "장당 가격(명목)": A / Q, "장당 가격(물가 반영)": R / Q, "물가 요인": A / R})
    out.loc[~ok.to_numpy()] = np.nan
    return out


def contributions(row: pd.Series) -> dict:
    """명목 예매액 성장(로그)을 세 요인으로 나눈 몫(%). 성장하지 않았거나 계산할 수 없으면 빈 dict."""
    vals = row[["예매수", "장당 가격(물가 반영)", "물가 요인", "명목 예매액"]]
    if vals.isna().any() or (vals <= 0).any() or row["명목 예매액"] <= 1:
        return {}
    la = float(np.log(row["명목 예매액"]))
    return {"관객 수": float(np.log(row["예매수"])) / la * 100,
            "가격(물가 반영)": float(np.log(row["장당 가격(물가 반영)"])) / la * 100,
            "물가 상승": float(np.log(row["물가 요인"])) / la * 100}


def ols_line(x: np.ndarray, y: np.ndarray):
    """단순 선형회귀 y = a + b·x. (기울기 b, 기울기의 표준오차, 결정계수 R², 관측 수, 절편 a)를 돌려준다."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sxx = float(((x - x.mean()) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    se = float(np.sqrt(resid @ resid / (n - 2) / sxx)) if sxx > 0 else np.nan
    return float(beta[1]), se, (float(1 - resid @ resid / sst) if sst > 0 else np.nan), n, float(beta[0])


# 코로나 시기로 표시하는 달(설명용 음영 구간과 같은 기간의 달). 전월 대비 변화는 두 달이 필요해서 앞뒤 어느 쪽이든 이 달이면 코로나 시기로 본다.
COVID_MONTHS = pd.date_range("2020-02-01", "2022-04-01", freq="MS")


def monthly_change_frame(wa: pd.DataFrame, wc: pd.DataFrame, price: pd.Series, g: str) -> pd.DataFrame:
    """월별 전월 대비 변화(로그 변화 × 100, 대략 %): 소비자물가와 장르의 장당 평균 가격(명목)."""
    p = ratio(wa[[g]], wc[[g]])[g]
    fr = pd.DataFrame({"물가": np.log(price.reindex(p.index)).diff() * 100, "가격": np.log(p).diff() * 100})
    prev = fr.index - pd.DateOffset(months=1)
    fr["코로나"] = fr.index.isin(COVID_MONTHS) | pd.Index(prev).isin(COVID_MONTHS)
    return fr.dropna(subset=["물가", "가격"])


def mom_regression(fr: pd.DataFrame, include_covid: bool):
    d = fr if include_covid else fr[~fr["코로나"]]
    return ols_line(d["물가"].to_numpy(), d["가격"].to_numpy()) if len(d) >= 8 else None


def in_range(s: pd.DataFrame, unit: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if unit == "월별":
        return s[(s.index >= start_ts) & (s.index <= end_ts)]
    return s[(s.index >= start_ts) & (s.index + pd.DateOffset(months=2) <= end_ts)]


# ---------------------------------------------------------------- 차트
def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GRID),
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def add_covid_band(fig: go.Figure, x_min: pd.Timestamp, x_max: pd.Timestamp):
    a, b = pd.Timestamp(COVID_SPAN[0]), pd.Timestamp(COVID_SPAN[1])
    if b < x_min or a > x_max:
        return
    fig.add_shape(type="rect", xref="x", yref="paper", x0=max(a, x_min), x1=min(b, x_max), y0=0, y1=1,
                  fillcolor="#eef1f6", opacity=0.9, line_width=0, layer="below")
    fig.add_annotation(x=max(a, x_min), xref="x", y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                       text="참고: 국내 첫 확진(2020.01)~거리두기 전면 해제(2022.04)", showarrow=False,
                       font=dict(size=11, color="#6b7380"))


def make_line_figure(s: pd.DataFrame, genres: list, ytitle: str, ref100: bool, quarterly: bool,
                     cpi: pd.Series = None, hover_suffix: str = "") -> go.Figure:
    """장르별 선 그래프. cpi를 주면 소비자물가 선을 점선으로 함께 그린다."""
    fig = go.Figure()
    for g in genres:
        if g not in s.columns:
            continue
        labels = [qlabel(t) if quarterly else t.strftime("%Y-%m") for t in s.index]
        fig.add_trace(go.Scatter(
            x=s.index, y=s[g], mode="lines", name=name(g), customdata=labels,
            line=dict(color=COLORS[g], width=2.5, dash="dot" if g == "합계" else "solid"),
            hovertemplate="<b>" + name(g) + "</b><br>%{customdata}" + hover_suffix + "<br>%{y:,.1f}<extra></extra>"))
    if cpi is not None and len(cpi):
        fig.add_trace(go.Scatter(
            x=cpi.index, y=cpi.to_numpy(), mode="lines", name="소비자물가", customdata=[t.strftime("%Y-%m") for t in cpi.index],
            line=dict(color=CPI_COLOR, width=2, dash="dash"),
            hovertemplate="<b>소비자물가</b><br>%{customdata}" + hover_suffix + "<br>%{y:,.1f}<extra></extra>"))
    if ref100:
        fig.add_hline(y=100, line_dash="dot", line_color="#b4bac4", line_width=1)
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(tickformat="%Y", dtick="M12", hoverformat="%Y-%m")
    xs = [s.index.min(), s.index.max()] if len(s) else []
    if xs:
        add_covid_band(fig, xs[0], xs[1])
    return style(fig)


def make_share_figure(share: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    xs = [qlabel(t) for t in share.index]
    for g in MUSIC:
        fig.add_trace(go.Bar(
            x=xs, y=share[g], name=name(g), marker=dict(color=COLORS[g], line=dict(color="white", width=2)),
            hovertemplate="<b>" + name(g) + "</b><br>%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="음악 3장르 중 비중 (%)", range=[0, 100])
    fig.update_xaxes(type="category", tickangle=-45)
    return style(fig)


def make_mom_figure(fr: pd.DataFrame, g: str, include_covid: bool, fit) -> go.Figure:
    """월별 전월 대비 변화 산점도: x = 소비자물가, y = 장당 가격. 한 점이 한 달."""
    fig = go.Figure()
    base = fr[~fr["코로나"]]
    fig.add_trace(go.Scatter(
        x=base["물가"], y=base["가격"], mode="markers", name="코로나 시기 외",
        customdata=[t.strftime("%Y-%m") for t in base.index],
        marker=dict(color=COLORS[g], size=8, line=dict(color="white", width=1)),
        hovertemplate="%{customdata}<br>물가 %{x:.2f}%<br>가격 %{y:.1f}%<extra></extra>"))
    cv = fr[fr["코로나"]]
    if include_covid and len(cv):
        fig.add_trace(go.Scatter(
            x=cv["물가"], y=cv["가격"], mode="markers", name="코로나 시기",
            customdata=[t.strftime("%Y-%m") for t in cv.index],
            marker=dict(color="rgba(255,255,255,0)", size=8, line=dict(color="#6b7380", width=1.6)),
            hovertemplate="%{customdata}<br>물가 %{x:.2f}%<br>가격 %{y:.1f}%<extra></extra>"))
    used = fr if include_covid else base
    if fit is not None and len(used):
        x0, x1 = float(used["물가"].min()), float(used["물가"].max())
        fig.add_trace(go.Scatter(x=[x0, x1], y=[fit[4] + fit[0] * x0, fit[4] + fit[0] * x1], mode="lines", name="회귀선",
                                 line=dict(color=NAVY, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title="소비자물가 전월 대비 (%)")
    fig.update_yaxes(title="장당 평균 가격 전월 대비 (%)")
    return style(fig)


# ---------------------------------------------------------------- 문장
def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:,.0f}%"


def result_sentences(dec, cpi_end: float, share_now: float, share_base: float, end: str) -> list:
    out = []
    if dec is not None:
        m = dec.loc[[g for g in MUSIC if g in dec.index]]

        def line(col, label, extra=""):
            parts = [f"{name(g)} {fmt_pct(m.loc[g, col] * 100)}" for g in m.index if not pd.isna(m.loc[g, col])]
            if parts:
                out.append(f"{end} 기준 최근 12개월 {label}{josa(label, '은', '는')} 2019년 연간 대비 " + ", ".join(parts) + f"입니다{extra}.")
        line("명목 예매액", "예매액")
        line("예매수", "예매수")
        line("장당 가격(명목)", "장당 평균 가격")
        if not pd.isna(cpi_end):
            parts = [f"{name(g)} {fmt_pct(m.loc[g, '장당 가격(물가 반영)'] * 100)}" for g in m.index
                     if not pd.isna(m.loc[g, "장당 가격(물가 반영)"])]
            if parts:
                out.append(f"같은 기간 소비자물가는 2019년의 {cpi_end:.0f}%였고, 물가를 반영한 장당 가격은 " + ", ".join(parts) + "입니다.")
        top = m["명목 예매액"].dropna()
        if not top.empty:
            g = top.idxmax()
            c = contributions(m.loc[g])
            if c:
                out.append(f"예매액이 가장 크게 성장한 {name(g)}의 성장은 로그 기준으로 관객 수 {c['관객 수']:.0f}%, "
                           f"물가를 뺀 가격 상승 {c['가격(물가 반영)']:.0f}%, 물가 상승 {c['물가 상승']:.0f}%에서 나왔습니다.")
    if not (pd.isna(share_now) or pd.isna(share_base)):
        out.append(f"음악 3장르 중 대중음악의 비중(예매액 기준)은 2019년 {share_base:.0f}%에서 최근 12개월 {share_now:.0f}%입니다.")
    elif not pd.isna(share_now):
        out.append(f"음악 3장르 중 대중음악의 비중은 최근 12개월 예매액 기준 {share_now:.0f}%입니다.")
    return out


def limit_lines(neg_months: int, share: float) -> list:
    return [
        "**예매액은 예매에서 취소를 뺀 값이고 예매일자 기준이며, 공연 주최자의 수익이 아닙니다.** 티켓을 산 날짜로 집계되어, "
        "취소가 몰린 달에는 값이 작아지거나 음수가 됩니다. 코로나 시기의 하락에는 수요 감소뿐 아니라 대규모 취소가 섞여 있습니다."
        + (f" 선택한 기간에는 장르별 월 값이 음수인 경우가 {neg_months}건 있습니다." if neg_months else ""),
        "**2019년은 기준이 낮을 수 있습니다.** 공연법에 따른 자료 수집이 본격화된 해라서, 2019년 대비 성장에는 집계 범위 확대가 "
        "섞여 있을 수 있습니다. 이 자료만으로는 그 크기를 알 수 없습니다. 2019년 안에서도 상반기(1~5월)는 하반기보다 크게 낮아, "
        "달 하나를 2019년 같은 달과 비교하면 성장이 과장됩니다. 이 대시보드는 최근 12개월 합계를 2019년 연간과 비교합니다.",
        "**물가 반영은 소비자물가지수(총지수)로 나눈 값입니다.** 공연 티켓 고유의 물가와는 다를 수 있습니다. "
        "또 ‘물가를 뺀 가격 상승’은 좌석 등급, 할인, 공연장 규모, 어떤 공연이 많이 팔렸는지의 구성 변화도 포함하므로 가격 인상만을 뜻하지 않습니다.",
        "**규모가 한쪽으로 쏠려 있습니다.**" + (f" 최근 12개월 음악 3장르 중 대중음악이 {share:.0f}%입니다." if not pd.isna(share) else "")
        + " 규모가 작은 국악이나 클래식은 몇 건의 공연에도 지수가 크게 흔들리므로, 장르 사이의 지수를 같은 무게로 비교하면 안 됩니다.",
        "**전산 발권된 공연만 집계됩니다.** 무료 공연이나 현장 판매가 많은 장르는 실제보다 작게 보일 수 있습니다.",
        "**공연 티켓 시장의 자료입니다.** 음원·스트리밍 소비와는 다릅니다.",
    ]


# ================================================================ 화면
df, price_s = load_data()
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.banner {{ background: linear-gradient(135deg, {NAVY} 0%, #1f3d73 100%); border-radius: 10px;
          padding: 1.5rem 1.8rem 1.4rem; margin-bottom: 1rem; }}
.banner .eyebrow {{ color: #9fb8e0; font-size: .76rem; font-weight: 700; letter-spacing: .1em; margin-bottom: .35rem; }}
.banner .title {{ color: #ffffff; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.25; }}
.banner .lead {{ color: #d6e2f5; font-size: 1.02rem; margin-top: .45rem; }}
h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; font-weight: 700; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #ffffff; border-color: #dfe4ec; border-radius: 8px; }}
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-top: 3px solid {NAVY}; box-shadow: 0 1px 3px rgba(20,40,75,.08); }}
[data-testid="stMetricLabel"] p {{ color: #52606d; font-weight: 600; font-size: .86rem; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; font-size: 1.9rem; }}
button[role="tab"] p {{ font-weight: 600; font-size: .98rem; }}
button[role="tab"][aria-selected="true"] {{ color: {NAVY}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {NAVY}; height: 3px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid #dfe4ec; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; color: {NAVY}; border-bottom: 2px solid {NAVY}; padding-bottom: .4rem; }}
[data-testid="stExpander"] {{ border-color: #dfe4ec; border-radius: 8px; background: #fbfcfe; }}
</style>
<div class="banner">
<div class="eyebrow">예술경영지원센터 공연예술통합전산망(KOPIS) · 예매통계</div>
<div class="title">공연 음악 장르별 성장 양상 대시보드</div>
<div class="lead">2019년 이후 공연 음악 장르(대중음악·클래식·국악)는 얼마나 성장했고, 그 성장은 관객 수와 가격 중 무엇에서 나왔는가?</div>
</div>
""", unsafe_allow_html=True)
st.caption(md("왼쪽에서 장르와 기간을 바꾸면 차트가 함께 바뀝니다. 금액은 예매에서 취소를 뺀 예매액이며 "
              f"예매일자 기준입니다. 자료 조회일 {DATA_DATE}."))

months = [m for m in sorted(df["월"].unique())]
with st.sidebar:
    st.header("조건 선택")
    genres = st.multiselect("장르 (차트에 표시)", MUSIC + COMPARE, default=MUSIC, format_func=name,
                            help="핵심 지표와 결론은 항상 음악 3장르(대중음악·클래식·국악) 기준입니다. 뮤지컬·연극·전체 공연은 비교용입니다.")
    start = st.selectbox("시작 월", months, index=0)
    end = st.selectbox("종료 월", months, index=len(months) - 1)

if start >= end:
    st.error("시작 월이 종료 월보다 앞서야 합니다. 기간을 다시 선택해 주세요.")
    st.stop()

start_ts, end_ts = pd.Timestamp(start + "-01"), pd.Timestamp(end + "-01")
wa_n, wa_r, wc = monthly_wide(df, "순예매액"), monthly_wide(df, "실질예매액"), monthly_wide(df, "순예매수")
dec = decompose(wa_n, wa_r, wc, end_ts)
mrow = dec.loc[[g for g in MUSIC if g in dec.index]] if dec is not None else None


def share_of(frame: pd.DataFrame):
    """음악 3장르 중 대중음악의 몫(%, 예매액 기준). 12개월이 모두 있고 합계가 양수일 때만."""
    f = frame[MUSIC]
    return float(f["대중음악"].sum() / f.sum().sum() * 100) if len(f) == 12 and f.sum().sum() > 0 else np.nan


share_ttm = share_of(last12(wa_n, end_ts))
share_base = np.nan if end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01") else share_of(wa_n[wa_n.index.year == BASE_YEAR])


_l12 = last12(price_s, end_ts)
cpi_end = float(_l12.mean() * 100) if len(_l12) == 12 and end_ts > pd.Timestamp(f"{BASE_YEAR}-12-01") else np.nan


def window(s):
    """선택한 시작~종료 월로 자른다(월 단위 인덱스)."""
    return s[(s.index >= start_ts) & (s.index <= end_ts)]


# ---------------------------------------------------------------- 핵심 지표
def kpi(col, title: str, value: str, sub: str, help_text: str):
    with col:
        with st.container(border=True):
            st.metric(title, value, help=help_text)
            st.caption(md(sub))


st.caption(md(f"{start} ~ {end} · 핵심 지표는 종료 월까지 최근 12개월을 2019년 연간과 비교한 값입니다."))
k1, k2, k3, k4 = st.columns(4)
need = "종료 월을 2019-12 이후로 선택하세요"
for col, title, key, what in [
        (k1, "예매액 성장 1위", "명목 예매액", "예매액"),
        (k2, "관객 수(예매수) 성장 1위", "예매수", "예매수"),
        (k3, "장당 가격 성장 1위", "장당 가격(명목)", "장당 평균 가격")]:
    ser = mrow[key].dropna() if mrow is not None else pd.Series(dtype=float)
    if ser.empty:
        kpi(col, title, "-", need, f"음악 3장르 중, 최근 12개월 {what}이 2019년 연간의 몇 %인지 비교합니다.")
    else:
        top = ser.idxmax()
        kpi(col, title, name(top), f"최근 12개월이 2019년의 {ser[top] * 100:,.0f}%",
            f"음악 3장르 중, 종료 월까지 최근 12개월 {what}이 2019년 연간의 몇 %인지 가장 높은 장르입니다.")
kpi(k4, "대중음악 비중 (음악 3장르)", "-" if pd.isna(share_ttm) else f"{share_ttm:.0f}%",
    "최근 12개월 예매액 기준" + ("" if pd.isna(share_base) else f" (2019년 {share_base:.0f}%)"),
    "대중음악 + 클래식 + 국악 합계 중 대중음악의 몫입니다. 규모가 한쪽으로 쏠려 있어 장르 비교에 주의가 필요합니다.")

with st.expander("용어 풀이: 예매액 · 성장 지수 · 최근 12개월 · 물가 반영 · 성장의 구성"):
    st.markdown(md(
        "- **예매액(예매수):** 예매에서 취소를 뺀 티켓 금액(장수)입니다. 티켓을 산 날짜로 집계하며, 공연이 열린 날짜가 아닙니다. 공연 주최자의 수익과는 다릅니다.\n"
        f"- **성장 지수(2019년 = 100):** 그 달까지 **최근 12개월 합계**가 {BASE_YEAR}년 연간 합계의 몇 %인지입니다. 200이면 {BASE_YEAR}년의 두 배입니다. "
        "12개월을 합쳐 비교하므로 계절성과, 기준이 되는 달 하나가 유난히 작아서 생기는 왜곡이 없습니다.\n"
        f"- **물가 반영(실질):** 월별 소비자물가지수를 {BASE_YEAR}년 평균으로 나눈 값으로 금액을 나눠, {BASE_YEAR}년 물가 기준 가격으로 바꾼 값입니다. 물가가 오른 만큼은 성장으로 보지 않습니다.\n"
        "- **장당 평균 가격:** 예매액 ÷ 예매수입니다(만 원). 합계끼리 나눈 값이라 큰 공연의 가격이 더 크게 반영됩니다.\n"
        "- **성장의 구성:** 명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인으로 나눕니다. 세 요인의 몫은 로그 기준으로 합이 100%입니다. "
        "음수인 몫은 그 요인이 성장을 깎았다는 뜻입니다.\n"
        "- **비중:** 대중음악·클래식·국악 합계 중 각 장르의 몫입니다."))

tab_rev, tab_cnt, tab_price, tab_share = st.tabs(["1. 예매액", "2. 예매수", "3. 장당 가격과 물가", "장르 비중"])

growth_note = ("각 점은 그 달까지 최근 12개월 합계를 2019년 연간 합계와 비교한 값입니다(2019-12에 100에서 시작). "
               "음영은 설명용 참고 구간(국내 첫 확진~거리두기 전면 해제)이며 분석 기준이 아닙니다.")
level_note = "월별 값입니다. 계절성이 있고, 취소가 몰린 달에는 값이 작아지거나 음수가 될 수 있습니다. 규모가 작은 장르는 성장 보기에서 흐름을 비교하세요."


def growth_tab(w_all: pd.DataFrame, level_title: str, key: str, label: str):
    view = st.radio("보기", [GROWTH_VIEW, LEVEL_VIEW], horizontal=True, key=key)
    if not genres:
        st.info("왼쪽에서 장르를 하나 이상 선택해 주세요.")
    elif view == GROWTH_VIEW:
        g = window(rolling_index(w_all))
        if g.dropna(how="all").empty:
            st.warning("선택한 기간에 계산할 값이 없습니다. 종료 월을 2019-12 이후로, 기간을 넓혀 주세요.")
        else:
            st.plotly_chart(make_line_figure(g, genres, "2019년 연간 = 100 (최근 12개월 합계 기준)", ref100=True, quarterly=False,
                                             hover_suffix=" (최근 12개월)"))
            st.caption(md(growth_note + (" 금액은 물가를 반영하지 않은 명목값입니다." if label == "예매액" else "")))
    else:
        mv = window(w_all)
        if mv.empty:
            st.warning("선택한 기간에 해당하는 값이 없습니다. 기간을 넓혀 주세요.")
        else:
            st.plotly_chart(make_line_figure(mv, genres, level_title, ref100=False, quarterly=False))
            st.caption(md(level_note))


with tab_rev:
    st.markdown(md("**예매액**(티켓 판매 금액)은 얼마나 늘었는가?"))
    growth_tab(wa_n, "억 원", "rev_view", "예매액")

with tab_cnt:
    st.markdown(md("**예매수**(티켓 판매 장수)는 얼마나 늘었는가?"))
    growth_tab(wc, "만 장", "cnt_view", "예매수")

with tab_price:
    ALL = MUSIC + COMPARE
    last_ts = pd.Timestamp(LAST_MONTH + "-01")
    dec_all = decompose(wa_n, wa_r, wc, last_ts)
    st.markdown(md("**장당 평균 가격**(예매액 ÷ 예매수)은 물가보다 더 올랐는가? 소비자물가지수는 이 탭에서만 사용합니다. "
                   f"이 탭은 왼쪽에서 고른 장르·기간과 상관없이 **전체 장르, 전체 기간(2019~{LAST_MONTH.replace('-', '.')})**을 보여줍니다."))
    g = rolling_index(wa_n, denom=wc)
    st.plotly_chart(make_line_figure(g, ALL, "2019년 연간 = 100 (최근 12개월 기준)", ref100=True, quarterly=False,
                                     cpi=rolling_price_level(price_s), hover_suffix=" (최근 12개월)"))
    st.caption(md("장르 선이 점선(소비자물가)보다 위에 있으면 물가 상승분을 넘어 가격이 올랐다는 뜻입니다. 가격은 명목값이고, "
                  "소비자물가는 최근 12개월 평균 지수입니다. 공연마다 가격 차이가 매우 커서 가격의 실제 값은 보여주지 않고 "
                  "2019년 대비 변화만 비교합니다. " + growth_note))
    if dec_all is not None:
        cpi_last = float(last12(price_s, last_ts).mean() * 100)
        rows = [{"장르": name(gname),
                 "장당 가격·명목 (2019=100)": round(dec_all.loc[gname, "장당 가격(명목)"] * 100, 1),
                 "소비자물가 (2019=100)": round(cpi_last, 1),
                 "장당 가격·물가 반영 (2019=100)": round(dec_all.loc[gname, "장당 가격(물가 반영)"] * 100, 1)}
                for gname in ALL if gname in dec_all.index and not dec_all.loc[gname].isna().all()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(md(f"{LAST_MONTH} 기준 최근 12개월입니다. 물가 반영 가격은 월별 물가로 금액을 나눠 계산하므로, "
                      "명목 가격 지수를 소비자물가 지수로 나눈 값과 1~2%p 다를 수 있습니다."))

    st.markdown("##### 가격은 물가를 따라 움직였나: 월별 전월 대비 변화")
    st.markdown(md("한 점이 한 달입니다. 가로축은 그 달 소비자물가가 전월보다 몇 % 변했는지, 세로축은 그 달 장당 평균 가격이 전월보다 몇 % 변했는지입니다. "
                   "물가가 많이 오른 달에 가격도 많이 올랐다면 점들이 오른쪽 위로 늘어섭니다."))
    c1, c2 = st.columns([2, 3])
    with c1:
        gsel = st.selectbox("산점도로 볼 장르", ALL, format_func=name, key="mom_genre")
    with c2:
        inc_cv = st.checkbox("코로나 시기(2020.02~2022.04) 포함", value=True, key="mom_covid",
                             help="이 시기에는 판매가 거의 없고 취소가 몰려 가격이 의미 없이 크게 튑니다. 아래 표에는 포함·제외 결과를 나란히 보여줍니다.")
    fr = monthly_change_frame(wa_n, wc, price_s, gsel)
    fit_sel = mom_regression(fr, inc_cv)
    st.plotly_chart(make_mom_figure(fr, gsel, inc_cv, fit_sel))
    if fit_sel is not None:
        used = fr if inc_cv else fr[~fr["코로나"]]
        st.caption(md(f"{name(gsel)}: 기울기 {fit_sel[0]:.1f} ± {fit_sel[1]:.1f}, R² {fit_sel[2]:.3f}, {fit_sel[3]}개월. "
                      f"소비자물가의 전월 대비 변화는 표준편차 {used['물가'].std():.1f}%인데, 장당 가격은 {used['가격'].std():.0f}%로 훨씬 크게 흔들립니다."))
    rows = []
    for gname in ALL:
        frg = monthly_change_frame(wa_n, wc, price_s, gname)
        a_, b_ = mom_regression(frg, True), mom_regression(frg, False)
        rows.append({"장르": name(gname),
                     "전체: 기울기 (±표준오차)": "-" if a_ is None else f"{a_[0]:.1f} ± {a_[1]:.1f}",
                     "전체: R²": None if a_ is None else round(a_[2], 3),
                     "코로나 제외: 기울기 (±표준오차)": "-" if b_ is None else f"{b_[0]:.1f} ± {b_[1]:.1f}",
                     "코로나 제외: R²": None if b_ is None else round(b_[2], 3),
                     "개월 수 (전체 / 제외)": f"{'-' if a_ is None else a_[3]} / {'-' if b_ is None else b_[3]}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(md("**읽는 법:** 기울기는 ‘소비자물가가 전월보다 1% 오른 달에 장당 가격이 평균 몇 % 변했는가’이고, R²는 가격 변화 중 물가 변화로 설명되는 비율입니다. "
                  "R²가 0에 가까우면 월별 가격 변화는 물가와 관계가 없다는 뜻입니다. 가격은 그 달에 어떤 공연이 많이 팔렸는지에 따라 달라지고, "
                  "물가는 한 달에 0.3% 안팎으로만 변하기 때문입니다. 코로나 시기(전후 한 달 포함)는 포함해도 제외해도 결론이 같은지 확인하려고 나란히 두었습니다. "
                  "표준오차는 자기상관이 있으면 실제보다 작게 나옵니다. 인과가 아니라 ‘함께 움직인 정도’로만 해석하세요."))

    if dec_all is not None:
        rows = []
        for gname in ALL:
            if gname not in dec_all.index or dec_all.loc[gname].isna().all():
                continue
            r = dec_all.loc[gname]
            c = contributions(r)
            rows.append({"장르": name(gname),
                         "명목 예매액 (2019=100)": round(r["명목 예매액"] * 100, 1),
                         "예매수 (2019=100)": round(r["예매수"] * 100, 1),
                         "장당 가격·물가 반영 (2019=100)": round(r["장당 가격(물가 반영)"] * 100, 1),
                         "물가 요인 (2019=100)": round(r["물가 요인"] * 100, 1),
                         "성장 몫: 관객 수 (%)": None if not c else round(c["관객 수"], 1),
                         "성장 몫: 가격 (%)": None if not c else round(c["가격(물가 반영)"], 1),
                         "성장 몫: 물가 (%)": None if not c else round(c["물가 상승"], 1)})
        if rows:
            st.markdown("##### 예매액 성장은 무엇에서 왔나 (최근 12개월, 2019년 대비)")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(md("명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인입니다. 성장 몫은 명목 예매액이 늘어난 장르에서만 계산하며 "
                          "세 요인의 합이 100%입니다. 음수는 그 요인이 성장을 깎았다는 뜻입니다."))

with tab_share:
    st.markdown(md("**음악 3장르 중 각 장르의 비중**입니다(분기별, 명목 예매액 기준). 규모가 한쪽으로 쏠려 있는지, 그 구성이 어떻게 바뀌었는지 확인하는 용도입니다."))
    qa = to_quarterly(wa_n)[MUSIC]
    qa = in_range(qa, "분기별", start_ts, end_ts)
    qa = qa[(qa > 0).all(axis=1)]            # 취소가 몰려 음수가 된 분기는 비중을 계산하지 않음
    if qa.empty:
        st.warning("선택한 기간에 비중을 계산할 분기가 없습니다. 기간을 넓혀 주세요.")
    else:
        share = qa.div(qa.sum(axis=1), axis=0) * 100
        st.plotly_chart(make_share_figure(share))
        st.caption(md("예매액이 음수인 분기는 제외했습니다. 클래식과 국악이 차지하는 비중이 매우 작아 막대에서 잘 보이지 않을 수 있습니다."))

# ---------------------------------------------------------------- 결론 (탭 아래)
neg_view = int((in_range(wa_n, "월별", start_ts, end_ts)[genres] < 0).sum().sum()) if genres else 0
sents = result_sentences(dec, cpi_end, share_ttm, share_base, end)
with st.container(border=True):
    st.subheader("결론")
    for t in sents:
        st.markdown("- " + md(t))
    if not sents:
        st.markdown("- 선택한 조건에서는 문장으로 요약할 결과가 없습니다. 종료 월과 기간을 확인해 주세요.")
with st.expander("이 결론의 한계", expanded=True):
    for t in limit_lines(neg_view, share_ttm):
        st.markdown("- " + md(t))

with st.expander("출처와 집계 기준"):
    st.markdown(md(
        "- **출처:** 예술경영지원센터, 공연예술통합전산망(KOPIS) 예매통계 ‘장르별통계’. 월 단위로 조회해 합쳤고, "
        f"조회일은 {DATA_DATE}(최종집계 2026.11.26)입니다. 통계는 갱신되므로 조회일에 따라 값이 달라질 수 있습니다.\n"
        "- **물가:** 국가데이터처(통계청) 소비자물가조사, 소비자물가지수 총지수(2020=100) 월별. 연평균과 최근 월 값을 공식 통계와 대조했습니다.\n"
        f"- **기간:** 2019.01~{LAST_MONTH.replace('-', '.')} (2026.11은 11/25까지의 부분 자료라 제외). "
        "KOPIS의 본격적인 자료 수집은 2019년 공연법 시행 이후입니다.\n"
        "- **지표:** 예매액 = 총 티켓예매액(예매 − 취소, 예매일자 기준, 천원 단위를 억 원으로 환산). "
        "예매수 = 총 티켓예매수(예매수 − 취소수, 만 장). 장당 평균 가격 = 예매액 ÷ 예매수(만 원).\n"
        f"- **성장 지수:** 그 달까지 최근 12개월 합계 ÷ {BASE_YEAR}년 연간 합계 × 100. 장당 가격은 12개월 예매액 ÷ 12개월 예매수로 계산합니다. "
        "물가 반영 금액 = 월별 금액 ÷ (그 달 소비자물가지수 ÷ 2019년 평균 지수). 공연건수는 월별 합산 시 중복이 생겨 사용하지 않았습니다.\n"
        "- **장르 분류:** KOPIS는 2023년 1월 장르를 개편했습니다. 이 자료에는 2019년부터 대중음악이 집계되어 있으나, "
        "개편 전후의 정의 차이는 확인하지 못했습니다.\n"
        "- **음영 구간:** 국내 첫 확진(2020.01)과 사회적 거리두기 전면 해제(2022.04)는 설명용 참고일이며, 분석 기준이 아닙니다."))
