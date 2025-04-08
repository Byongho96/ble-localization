import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, gaussian_kde

def calculate_statistics(df: pd.DataFrame, column_name: str) -> pd.DataFrame:

    azimuth = df[column_name]

    sigma = azimuth.std(ddof=1)  # ddof=1 -> sample standard deviation

    # 왜도 (skewness)
    S = skew(azimuth)

    # 첨도 (kurtosis)
    # scipy.stats.kurtosis 의 기본 옵션은 Fisher’s definition(Excess Kurtosis)
    K = kurtosis(azimuth)  # Excess Kurtosis 반환
    # 만약 통계학 교과서에서 흔히 말하는 '일반' Kurtosis(“moment”)를 원한다면:
    # K_moment = kurtosis(azimuth, fisher=False)

    # 하이퍼 왜도 (5차 표준모멘트 활용 예시)
    # 5차 모멘트(평균 중심) = E[(X - μ)^5]
    # 5차 표준모멘트 = ( E[(X - μ)^5] ) / (σ^5 )
    mean_val = azimuth.mean()
    m5 = ((azimuth - mean_val) ** 5).mean()
    hyper_skew = m5 / (sigma ** 5)

    # 피크 확률 (Peak Probability)
    # 보통은 분포 자체의 "최고 확률 밀도" 정도로 해석 가능 -> 커널 밀도 추정(KDE) 예시
    kde = gaussian_kde(azimuth)
    xs = np.linspace(azimuth.min(), azimuth.max(), 1000)  # 구간 세분화
    pdf_vals = kde(xs)
    peak_prob = pdf_vals.max()  # 가장 큰 PDF 값

    df_statistics = pd.DataFrame({
        'X_Real': df['X_Real'].iloc[0],
        'Y_Real': df['Y_Real'].iloc[0],
        'Sigma': sigma,
        'S': S,
        'K': K,
        'Hyper_Skew': hyper_skew,
        'Peak_Prob': peak_prob
    }, index=[0])

    return df_statistics