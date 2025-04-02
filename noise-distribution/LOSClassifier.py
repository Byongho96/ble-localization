import numpy as np
from scipy.stats import rayleigh, rice
from collections import deque

class LOSClassifier:
    def __init__(self, window_size=100):
        """
        window_size: 분포 피팅에 사용할 데이터 개수(슬라이딩 윈도 크기)
        """
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)

    def add_data(self, rssi_dbm):
        """
        실시간(또는 반복)으로 새 dBm RSSI 값을 추가하는 메서드.
        """
        self.data_window.append(rssi_dbm)

    def classify(self):
        """
        현재 window에 모인 RSSI(dBm) 데이터를 선형 진폭으로 변환한 뒤
        Rayleigh vs Rician 분포 파라미터를 피팅하고,
        각각의 로그우도를 비교하여 LOS(NLOS) 를 판별한다.
        
        returns: str ("LOS (Rician)" 또는 "NLOS (Rayleigh)" 등)
        """
        # 충분한 데이터가 쌓이지 않으면 분류하지 않음
        if len(self.data_window) < self.window_size:
            return "Insufficient data"

        # dBm -> 선형 진폭으로 변환
        #   Amplitude = 10^(RSSI(dBm)/20)
        #   (전력으로 변환하고 싶다면: Power(mW) = 10^(RSSI(dBm)/10))
        data_dbm = np.array(self.data_window, dtype=float)
        data_amp = 10 ** (data_dbm / 20.0)  # 선형 진폭

        # Rayleigh 분포 파라미터 추정
        rayleigh_params = rayleigh.fit(data_amp, floc=0)
        loglike_rayleigh = np.sum(np.log(rayleigh.pdf(data_amp, *rayleigh_params) + 1e-12))

        # Rician(Rice) 분포 파라미터 추정
        rice_params = rice.fit(data_amp, floc=0)
        loglike_rice = np.sum(np.log(rice.pdf(data_amp, *rice_params) + 1e-12))

        # 로그우도(log-likelihood)가 더 큰 쪽을 선택
        if loglike_rice > loglike_rayleigh:
            return 0 # "LOS (Rician)"
        else:
            return 1 # "NLOS (Rayleigh)"
