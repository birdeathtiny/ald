# ==============================================================================
# [GRAND MASTER VERSION] ALD AI & Physics Hybrid Optimization Platform
# ==============================================================================
# 파일명: ald_grand_master.py
# 작성일: 2024-05-22
# 버전: v4.0 Ultimate
# ------------------------------------------------------------------------------
# [시스템 개요]
# 1. Physics Engine: Knudsen 확산, Mean Free Path, Gordon's Step Coverage 모델 탑재
# 2. AI Modeling: PyTorch (MLP, 1D-CNN), XGBoost, RandomForest, Gaussian Process
# 3. Optimization: Bayesian Optimization, Genetic Algorithm, SLSQP
# 4. Visualization: Radar Chart, Trend Sweep Line Plot, SHAP, Physics Comparison
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
import itertools
from typing import Dict, List, Tuple, Any, Optional

# --- 시각화 라이브러리 ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- AI 및 데이터 처리 라이브러리 ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# --- 최적화 라이브러리 ---
from scipy.optimize import minimize, differential_evolution

# --- 고급 라이브러리 로드 (예외 처리) ---
try:
    import xgboost as xgb
    import shap
    HAS_ADVANCED_LIB = True
except ImportError:
    HAS_ADVANCED_LIB = False
    warnings.warn("XGBoost 또는 SHAP 라이브러리가 없습니다. 일부 기능이 제한됩니다.")

# 경고 무시 및 스타일 설정
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_theme(style="whitegrid")


# ==============================================================================
# [MODULE 1] DOMAIN KNOWLEDGE BASE (물리 상수 데이터베이스)
# ==============================================================================
class ALDConstants:
    """ALD 공정 물리 상수 및 전구체 정보 관리 클래스"""
    N_A = 6.02214076e23      # 아보가드로 수 (mol^-1)
    k_B = 1.380649e-23       # 볼츠만 상수 (J/K)
    R_GAS = 8.314462618      # 기체 상수 (J/(mol*K))

    # 전구체 물성 DB
    PRECURSORS = {
        "TMA": {
            "name": "Trimethylaluminum",
            "formula": "Al(CH3)3",
            "mass_g_mol": 72.09,
            "diameter_m": 5.0e-10,
            "sticking_c": 0.05,
            "density": 0.752
        },
        "TDMAH": {
            "name": "Tetrakis(dimethylamido)hafnium",
            "formula": "Hf(NMe2)4",
            "mass_g_mol": 354.79,
            "diameter_m": 8.5e-10,
            "sticking_c": 0.01,
            "density": 1.30
        },
        "TEMAHf": {
            "name": "Tetrakis(ethylmethylamido)hafnium",
            "formula": "Hf(NEtMe)4",
            "mass_g_mol": 410.9,
            "diameter_m": 12.0e-10,
            "sticking_c": 0.02,
            "density": 1.10
        },
        "Zr(NEt2)4": {
            "name": "Tetrakis(diethylamido)zirconium",
            "formula": "Zr(NEt2)4",
            "mass_g_mol": 379.79,
            "diameter_m": 11.0e-10,
            "sticking_c": 0.08,
            "density": 1.05
        },
        "TiCl4": {
            "name": "Titanium Tetrachloride",
            "formula": "TiCl4",
            "mass_g_mol": 189.68,
            "diameter_m": 6.0e-10,
            "sticking_c": 0.10,
            "density": 1.73
        }
    }

    @classmethod
    def get_properties(cls, precursor_name: str) -> Dict:
        return cls.PRECURSORS.get(precursor_name, cls.PRECURSORS["TMA"])


# ==============================================================================
# [MODULE 2] PHYSICS ENGINE (물리 엔진)
# ==============================================================================
class PhysicsEngine:
    """
    기체 분자 운동론 및 확산 모델을 기반으로 한 물리 연산 엔진
    """
    
    @staticmethod
    def calculate_gas_kinetics(T_c: float, P_torr: float, precursor_name: str) -> Tuple[float, float, float]:
        """기체 분자의 기본 운동학적 파라미터 계산"""
        props = ALDConstants.get_properties(precursor_name)
        d_m = props["diameter_m"]
        mass_kg = props["mass_g_mol"] / 1000.0 / ALDConstants.N_A
        
        T_K = T_c + 273.15
        P_Pa = P_torr * 133.322
        
        # 1. 평균 자유 행로 (Mean Free Path)
        # lambda = kT / (sqrt(2) * pi * d^2 * P)
        lambda_m = (ALDConstants.k_B * T_K) / (np.sqrt(2) * np.pi * (d_m**2) * P_Pa)
        
        # 2. 평균 열 속도 (Mean Thermal Velocity)
        # v = sqrt(8kT / pi*m)
        v_avg = np.sqrt((8 * ALDConstants.k_B * T_K) / (np.pi * mass_kg))
        
        return lambda_m, v_avg, props["sticking_c"]

    @staticmethod
    def calculate_diffusion_coefficients(lambda_m: float, v_avg: float, CD_m: float) -> Tuple[float, float, float]:
        """확산 계수 (Bulk, Knudsen, Effective) 계산"""
        # Bulk Diffusion (D_bulk = 1/3 * lambda * v)
        D_bulk = (1.0/3.0) * lambda_m * v_avg
        
        # Knudsen Diffusion (D_Kn = 1/3 * CD * v) -> 작은 구멍에서의 확산
        D_knudsen = (1.0/3.0) * CD_m * v_avg
        
        # Bosanquet Formula (Effective Diffusivity)
        # 1/D_eff = 1/D_bulk + 1/D_Kn
        D_eff = 1.0 / ((1.0 / (D_bulk + 1e-20)) + (1.0 / (D_knudsen + 1e-20)))
        
        return D_bulk, D_knudsen, D_eff

    @staticmethod
    def calculate_step_coverage(T_c, P_torr, pulse_time, precursor, CD_nm, AR) -> Dict[str, float]:
        """
        Gordon's Model을 응용한 Step Coverage 및 포화도 계산 (메인 함수)
        """
        # 단위 변환
        CD_m = CD_nm * 1e-9
        L_depth_m = CD_m * AR
        
        # 1. 기체 운동학 파라미터 산출
        lambda_m, v_avg, sticking = PhysicsEngine.calculate_gas_kinetics(T_c, P_torr, precursor)
        
        # 2. Knudsen Number 계산 (유동 영역 판단)
        Kn = lambda_m / (CD_m + 1e-12)
        
        # 3. 확산 계수 산출
        _, _, D_eff = PhysicsEngine.calculate_diffusion_coefficients(lambda_m, v_avg, CD_m)
        
        # 4. 필요 포화 시간 (Saturation Time) 예측
        # t_sat = (L^2 / 2*D_eff) * factor (Sticking coeff 및 AR 고려)
        # Sticking probability가 높으면 입구 막힘 현상으로 더 오래 걸림
        t_diffusion = (L_depth_m**2) / (2 * D_eff + 1e-20)
        t_req = t_diffusion * (1 + AR * 0.5 + sticking * 10.0)

        # 5. Step Coverage 계산 (S-Curve 근사)
        # 펄스 시간이 요구 시간 대비 얼마나 충분한가? (Saturation Ratio)
        sat_ratio = pulse_time / (t_req + 1e-9)
        
        if sat_ratio >= 3.0:
            sc = 99.9
        else:
            # Sigmoid 형태의 SC 곡선 가정
            sc = 100.0 * (1.0 - np.exp(-sat_ratio * 1.5))
            
        return {
            "SC": sc,
            "Kn": Kn,
            "MFP": lambda_m,
            "T_req": t_req,
            "Regime": "Knudsen" if Kn > 1 else "Continuum"
        }


# ==============================================================================
# [MODULE 3] DEEP LEARNING ARCHITECTURES (PyTorch 모델)
# ==============================================================================
class ALD_MLP(nn.Module):
    """정형 데이터를 위한 Multi-Layer Perceptron"""
    def __init__(self, input_size, output_size):
        super(ALD_MLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(), # Swish Function
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.SiLU()
        )
        self.regressor = nn.Linear(32, output_size)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)

class ALD_CNN_1D(nn.Module):
    """1D Convolution을 이용한 Feature Extraction 모델"""
    def __init__(self, input_size, output_size):
        super(ALD_CNN_1D, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (Batch, 1, Features)
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Flatten size 계산 필요 (여기서는 AdaptivePool 사용으로 해결)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==============================================================================
# [MODULE 4] OPTIMIZATION CONTROLLER (최적화 컨트롤러)
# ==============================================================================
class UltimateALDOptimizer:
    """전체 시스템을 관장하는 최적화 컨트롤러"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.models = {}
        self.scaler_X = None
        self.scaler_Y = None
        self.feature_cols = []
        self.target_cols = []
        self.cat_cols = ['Precursor', 'Co-reactant', 'Purge Gas']
        
        # 시스템 초기화
        self._initialize_system()

    def _initialize_system(self):
        """데이터 로드 및 모델 학습 파이프라인 실행"""
        try:
            df = self._load_data()
            self._preprocess_data(df)
            self._train_ml_models()
            self._train_dl_models()
        except Exception as e:
            st.error(f"시스템 초기화 중 치명적 오류 발생: {str(e)}")

    def _load_data(self) -> pd.DataFrame:
        """CSV 파일 로드 또는 더미 데이터 생성"""
        if os.path.exists(self.file_path):
            try:
                return pd.read_csv(self.file_path, encoding='CP949')
            except:
                return pd.read_csv(self.file_path, encoding='utf-8')
        else:
            return self._generate_dummy_data()

    def _generate_dummy_data(self) -> pd.DataFrame:
        """파일이 없을 경우 테스트를 위한 더미 데이터 생성"""
        np.random.seed(42)
        N = 200
        data = {
            'Temperature (c)': np.random.uniform(150, 450, N),
            'Pressure (torr)': np.random.uniform(0.1, 5.0, N),
            'Precursor_Pulse Time (s)': np.random.uniform(0.1, 3.0, N),
            'Purge Time (s)': np.random.uniform(5, 60, N),
            'Precursor': np.random.choice(list(ALDConstants.PRECURSORS.keys()), N),
            'Co-reactant': np.random.choice(['H2O', 'O3'], N),
            'Purge Gas': np.random.choice(['N2', 'Ar'], N),
            # Target Variables (물리 공식에 노이즈 추가하여 생성)
            'Uniformity (%)': np.random.uniform(90, 99.9, N)
        }
        df = pd.DataFrame(data)
        # 상관관계가 있는 타겟 생성
        df['GPC (A/cycle)'] = (df['Temperature (c)'] / 300) * 0.8 + np.random.normal(0, 0.1, N)
        df['Thickness (nm)'] = df['GPC (A/cycle)'] * np.random.randint(20, 100, N)
        df['Step Coverage (sc, %)'] = 100 - (df['Pressure (torr)'] * 2) - (400 - df['Temperature (c)'])*0.05
        df['Surface Roughness (RMS, nm)'] = (df['Temperature (c)'] - 250)**2 / 10000 + 0.2
        return df

    def _preprocess_data(self, df: pd.DataFrame):
        """데이터 전처리: 인코딩, 스케일링, 분할"""
        df.replace('-', np.nan, inplace=True)
        
        # 타겟 컬럼 정의
        potential_targets = ['Thickness (nm)', 'Surface Roughness (RMS, nm)', 'GPC (A/cycle)', 'Step Coverage (sc, %)', 'Uniformity (%)']
        self.target_cols = [c for c in potential_targets if c in df.columns]
        
        # 숫자형 변환
        for c in df.columns:
            if c not in self.cat_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=[c for c in self.cat_cols if c in df.columns])
        
        # Feature Selection
        drop_cols = self.target_cols + ['순서', 'Aspect Ratio (AR)', 'Leackage', 'Dielectric']
        self.feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
        
        # Array 변환 & Imputation
        X = df_encoded[self.feature_cols].values
        Y = df_encoded[self.target_cols].values
        
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        Y = imputer.fit_transform(Y)
        
        # Train/Test Split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Scaling
        self.scaler_X = StandardScaler().fit(self.X_train)
        self.scaler_Y = MinMaxScaler().fit(self.Y_train)
        
        self.X_train_s = self.scaler_X.transform(self.X_train)
        self.X_test_s = self.scaler_X.transform(self.X_test)
        self.Y_train_s = self.scaler_Y.transform(self.Y_train)
        
        self.input_dim = self.X_train_s.shape[1]
        self.output_dim = self.Y_train_s.shape[1]

    def _train_ml_models(self):
        """Machine Learning 모델 학습 (RF, XGB, GP)"""
        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(self.X_train_s, self.Y_train_s)
        self.models['RandomForest'] = rf
        
        # 2. XGBoost
        if HAS_ADVANCED_LIB:
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1)
                xgb_model.fit(self.X_train_s, self.Y_train_s)
                self.models['XGBoost'] = xgb_model
            except: pass
            
        # 3. Gaussian Process (Bayesian Opt용 Surrogate Model)
        # 데이터가 많으면 샘플링
        idx = np.random.choice(len(self.X_train_s), min(500, len(self.X_train_s)), replace=False)
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        gp.fit(self.X_train_s[idx], self.Y_train_s[idx])
        self.models['GaussianProcess'] = gp

    def _train_dl_models(self):
        """Deep Learning 모델 학습 (PyTorch)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset 준비
        train_ds = TensorDataset(torch.FloatTensor(self.X_train_s).to(device), torch.FloatTensor(self.Y_train_s).to(device))
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # 학습 루프 함수
        def train_loop(model, name, epochs=50):
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(epochs):
                for x, y in loader:
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            model.eval()
            self.models[name] = model.cpu()

        # MLP 및 CNN 학습 실행
        train_loop(ALD_MLP(self.input_dim, self.output_dim), "MLP")
        train_loop(ALD_CNN_1D(self.input_dim, self.output_dim), "CNN")

    def predict(self, params: Dict, model_name: str = 'XGBoost') -> Dict[str, float]:
        """단일 조건에 대한 물성 예측"""
        # Input Vector 구성
        input_df = pd.DataFrame([params])
        input_df = pd.get_dummies(input_df)
        
        aligned = pd.DataFrame(0, index=[0], columns=self.feature_cols)
        for c in input_df.columns:
            if c in self.feature_cols: aligned[c] = input_df[c].values
        
        # One-hot manual fix
        if 'Precursor' in params:
            p_col = f"Precursor_{params['Precursor']}"
            if p_col in self.feature_cols: aligned[p_col] = 1.0

        X_in = self.scaler_X.transform(aligned.values)
        
        # Inference
        if model_name in ['MLP', 'CNN']:
            with torch.no_grad():
                pred_s = self.models[model_name](torch.FloatTensor(X_in)).numpy()
        else:
            model = self.models.get(model_name, self.models.get('RandomForest')) # Fallback
            pred_s = model.predict(X_in)
            if pred_s.ndim == 1: pred_s = pred_s.reshape(1, -1)
            
        pred_inv = self.scaler_Y.inverse_transform(pred_s)[0]
        return dict(zip(self.target_cols, pred_inv))

    def optimize_process(self, constraints: Dict, algorithm: str, model_name: str) -> Tuple[Dict, Dict]:
        """
        [최적화 엔진] 사용자의 요구사항(Constraints)을 만족하는 최적 레시피 탐색
        """
        target_th = constraints.get('Thickness (nm)', 20.0)
        
        # 1. 목적 함수 정의 (Cost Function)
        def objective(x):
            # x = [Temp, Pressure, Pulse, Purge]
            curr = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], 
                "Precursor_Pulse Time (s)": x[2], "Purge Time (s)": x[3],
                "Precursor": constraints['Precursor']
            }
            pred = self.predict(curr, model_name)
            
            # 가중치 기반 Cost 계산
            cost = 0.0
            # (1) Roughness 최소화
            cost += pred.get('Surface Roughness (RMS, nm)', 1.0) * 50.0
            # (2) Step Coverage 최대화 (목표: 100%)
            cost += (100.0 - pred.get('Step Coverage (sc, %)', 80.0)) ** 2 * 0.5
            # (3) Uniformity 최대화
            cost += (100.0 - pred.get('Uniformity (%)', 90.0)) ** 2 * 0.2
            
            # (4) 물리적 제약 Penalty
            if x[0] > 450: cost += 5000 # 고온 분해 위험
            if x[0] < 100: cost += 5000 # 미반응 위험
            
            return cost

        bounds = [(100, 450), (0.1, 5.0), (0.1, 4.0), (5, 100)]
        
        # 2. 알고리즘별 탐색 실행
        best_x = None
        
        if algorithm == 'SLSQP':
            res = minimize(objective, x0=[250, 1.0, 1.0, 20], bounds=bounds, method='SLSQP')
            best_x = res.x
            
        elif algorithm == 'Genetic Algorithm':
            res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=20, popsize=15)
            best_x = res.x
            
        elif algorithm == 'Bayesian':
            # Gaussian Process Surrogate를 이용한 탐색
            # 여기서는 편의상 Random Sampling (Monte Carlo) 방식으로 구현
            # 실제 BO는 GP.predict(acq)를 최적화해야 함
            X_cand = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (1000, 4))
            scores = [objective(x) for x in X_cand]
            best_x = X_cand[np.argmin(scores)]
        else:
            best_x = [250, 1.0, 1.0, 20]

        # 3. 최종 레시피 구성
        final_recipe = {
            "Temperature (c)": best_x[0], "Pressure (torr)": best_x[1],
            "Precursor_Pulse Time (s)": best_x[2], "Purge Time (s)": best_x[3],
            "Precursor": constraints['Precursor']
        }
        final_pred = self.predict(final_recipe, model_name)
        
        # Cycle 수 계산 (목표 두께 / GPC)
        gpc = max(final_pred.get('GPC (A/cycle)', 0.1), 0.001)
        cycles = int(target_th * 10.0 / gpc)
        final_recipe['Cycles (n)'] = cycles
        final_pred['Thickness (nm)'] = cycles * gpc / 10.0
        
        return final_recipe, final_pred

    # --- 시각화 지원 메서드 ---
    def simulate_sweep(self, base_recipe: Dict, sweep_param: str, start: float, end: float) -> pd.DataFrame:
        """Trend Sweep 데이터를 생성하는 시뮬레이터"""
        vals = np.linspace(start, end, 30)
        results = []
        for v in vals:
            temp = base_recipe.copy()
            temp[sweep_param] = v
            # AI 예측
            res = self.predict(temp, 'XGBoost')
            res[sweep_param] = v
            # 물리 계산 추가
            phy = PhysicsEngine.calculate_step_coverage(
                temp.get('Temperature (c)', 250), temp.get('Pressure (torr)', 1),
                temp.get('Precursor_Pulse Time (s)', 1), temp.get('Precursor', 'TMA'),
                50, 20
            )
            res['Physics SC (%)'] = phy['SC']
            results.append(res)
        return pd.DataFrame(results)

    def get_shap_values(self):
        """SHAP 값 추출 (차원 오류 해결 버전)"""
        if not HAS_ADVANCED_LIB or 'XGBoost' not in self.models: return None, None
        explainer = shap.TreeExplainer(self.models['XGBoost'])
        # 속도를 위해 일부 샘플만 사용
        shap_values = explainer.shap_values(self.X_train_s[:100])
        return shap_values, self.feature_cols


# ==============================================================================
# [MODULE 5] STREAMLIT USER INTERFACE
# ==============================================================================
def main_gui():
    st.set_page_config(page_title="ALD Grand Master", layout="wide", page_icon="🧪")
    
    # Custom CSS
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .big-font {font-size: 20px !important; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 ALD Grand Master: Ultimate Platform")
    st.markdown("##### AI & Physics-Informed Process Optimization System v4.0")

    # [Sidebar] 설정 패널
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # 시스템 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "AI_ALD1.csv")
        
        @st.cache_resource
        def get_system(): return UltimateALDOptimizer(csv_path)
        
        optimizer = get_system()
        
        st.divider()
        st.subheader("1. Target Specs")
        prec = st.selectbox("Precursor", list(ALDConstants.PRECURSORS.keys()))
        target_th = st.number_input("Target Thickness (nm)", 5.0, 1000.0, 20.0)
        target_ar = st.slider("Target AR (Aspect Ratio)", 1, 100, 20)
        cd_nm = st.number_input("Pattern CD (nm)", 10, 500, 50)
        
        st.divider()
        st.subheader("2. AI Strategy")
        model_name = st.selectbox("Inference Model", ["XGBoost", "RandomForest", "MLP", "CNN", "GaussianProcess"])
        algo = st.selectbox("Optimization Algorithm", ["Bayesian", "Genetic Algorithm", "SLSQP"])
        
        st.divider()
        run_btn = st.button("🚀 Start Optimization", type="primary", use_container_width=True)

    # [Main] 실행 로직
    if run_btn:
        with st.spinner("🚀 최적화 엔진 가동 중... (물리 검증 및 AI 추론 수행)"):
            cons = {"Thickness (nm)": target_th, "Precursor": prec, "Target AR": target_ar, "CD (nm)": cd_nm}
            start_t = time.time()
            recipe, pred = optimizer.optimize_process(cons, algo, model_name)
            duration = time.time() - start_t
            st.session_state['res'] = (recipe, pred, cons, duration)

    # [Main] 결과 표시
    if 'res' in st.session_state:
        recipe, pred, cons, duration = st.session_state['res']
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard (Summary)", 
            "⚗️ Physics Verification", 
            "📈 Trend Simulator", 
            "🧠 XAI & Insights"
        ])
        
        # --- TAB 1: Summary & Radar Chart ---
        with tab1:
            st.success(f"최적화 완료! (소요 시간: {duration:.2f}s)")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📋 Optimal Recipe")
                st.info(f"""
                - **Precursor:** {cons['Precursor']}
                - **Temperature:** {recipe['Temperature (c)']:.1f} °C
                - **Pressure:** {recipe['Pressure (torr)']:.2f} Torr
                - **Pulse / Purge:** {recipe['Precursor_Pulse Time (s)']:.2f}s / {recipe['Purge Time (s)']:.1f}s
                - **Total Cycles:** {int(recipe['Cycles (n)'])}
                """)
            with col2:
                st.subheader("🔮 Predicted Outcomes")
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Result Thickness", f"{pred['Thickness (nm)']:.2f} nm", delta=f"{pred['Thickness (nm)'] - cons['Thickness (nm)']:.2f}")
                c_m1.metric("GPC", f"{pred.get('GPC (A/cycle)', 0):.2f} A/cyc")
                c_m2.metric("Step Coverage", f"{pred.get('Step Coverage (sc, %)', 0):.1f} %")
                c_m2.metric("Roughness", f"{pred.get('Surface Roughness (RMS, nm)', 0):.3f} nm", delta_color="inverse")
                
            st.divider()
            st.markdown("#### 🕸️ Balance Analysis (Radar Chart)")
            
            # 레이더 차트 데이터 준비
            labels = ['Speed (GPC)', 'Uniformity', 'Coverage (SC)', 'Smoothness']
            # 정규화 (0~1)
            values = [
                min(pred.get('GPC (A/cycle)', 0)/2.0, 1.0),
                pred.get('Uniformity (%)', 0)/100.0,
                pred.get('Step Coverage (sc, %)', 0)/100.0,
                1.0 / (pred.get('Surface Roughness (RMS, nm)', 1) + 0.1) # 거칠기는 역수
            ]
            values = [max(0, v) for v in values] # 음수 방지
            
            # 레이더 차트 그리기
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]; angles += angles[:1]
            
            fig_rad, ax_rad = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax_rad.fill(angles, values, color='blue', alpha=0.25)
            ax_rad.plot(angles, values, color='blue', linewidth=2)
            ax_rad.set_yticklabels([])
            ax_rad.set_xticks(angles[:-1])
            ax_rad.set_xticklabels(labels, size=12)
            st.pyplot(fig_rad)

        # --- TAB 2: Physics Verification ---
        with tab2:
            st.markdown("### 🧬 Physics Engine Report")
            st.caption("AI의 예측값이 물리적으로 타당한지 기체 분자 운동론으로 검증합니다.")
            
            phy_res = PhysicsEngine.calculate_step_coverage(
                recipe['Temperature (c)'], recipe['Pressure (torr)'], 
                recipe['Precursor_Pulse Time (s)'], cons['Precursor'],
                cons['CD (nm)'], cons['Target AR']
            )
            
            c_p1, c_p2, c_p3 = st.columns(3)
            c_p1.metric("Mean Free Path", f"{phy_res['MFP']*1e6:.2f} µm")
            c_p2.metric("Knudsen Number", f"{phy_res['Kn']:.2f}")
            c_p3.metric("Required Saturation Time", f"{phy_res['T_req']:.4f} s")
            
            st.markdown("#### ⚖️ Step Coverage Comparison")
            
            fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
            categories = ['Physics Limit (Theory)', 'AI Prediction (Data)']
            vals = [phy_res['SC'], pred.get('Step Coverage (sc, %)', 0)]
            colors = ['#B0BEC5', '#66BB6A']
            
            ax_bar.barh(categories, vals, color=colors)
            ax_bar.set_xlim(0, 105)
            ax_bar.axvline(90, color='red', linestyle='--', label='Target Spec (90%)')
            ax_bar.set_xlabel('Step Coverage (%)')
            ax_bar.legend()
            
            for i, v in enumerate(vals):
                ax_bar.text(v+1, i, f"{v:.1f}%", va='center', fontweight='bold')
            st.pyplot(fig_bar)
            
            if recipe['Precursor_Pulse Time (s)'] < phy_res['T_req']:
                st.warning(f"⚠️ **주의:** 현재 Pulse Time({recipe['Precursor_Pulse Time (s)']:.2f}s)은 이론적 최소 요구 시간({phy_res['T_req']:.4f}s)보다 짧습니다.")

        # --- TAB 3: Trend Simulator ---
        with tab3:
            st.markdown("### 📈 Interactive Parameter Sweep")
            st.info("최적화된 레시피를 기준으로, 특정 변수 하나를 변경했을 때 물성 변화를 시뮬레이션합니다.")
            
            cols = st.columns(3)
            with cols[0]: param_sw = st.selectbox("Parameter to Sweep", ["Temperature (c)", "Pressure (torr)", "Precursor_Pulse Time (s)"])
            with cols[1]: view_targets = st.multiselect("Metrics to View", ["Step Coverage (sc, %)", "Physics SC (%)", "GPC (A/cycle)", "Surface Roughness (RMS, nm)"], default=["Step Coverage (sc, %)", "Physics SC (%)"])
            
            # 시뮬레이션 실행
            curr_val = recipe[param_sw]
            df_sweep = optimizer.simulate_sweep(recipe, param_sw, curr_val*0.5, curr_val*1.5)
            
            # 그래프
            if view_targets:
                fig_line, ax_line = plt.subplots(figsize=(10, 5))
                for target in view_targets:
                    sns.lineplot(data=df_sweep, x=param_sw, y=target, label=target, marker='o', ax=ax_line)
                
                ax_line.axvline(curr_val, color='red', linestyle='--', label='Current Recipe')
                ax_line.set_title(f"Effect of {param_sw}")
                ax_line.grid(True, alpha=0.3)
                ax_line.legend()
                st.pyplot(fig_line)

        # --- TAB 4: XAI & SHAP ---
        with tab4:
            st.markdown("### 🧠 Explainable AI (Model Interpretability)")
            
            if HAS_ADVANCED_LIB and model_name == 'XGBoost':
                shap_vals, features = optimizer.get_shap_values()
                
                if shap_vals is not None:
                    # [CRITICAL FIX] 차원 문제 해결 로직 재확인
                    if isinstance(shap_vals, list): 
                        sv = shap_vals[0] # First output
                    else: 
                        sv = shap_vals
                        if len(sv.shape) == 3: sv = sv[:, :, 0] # (Sample, Feature, Target) -> (Sample, Feature)
                    
                    # Feature Importance Calculation
                    importance = np.abs(sv).mean(axis=0)
                    if importance.ndim > 1: importance = importance.mean(axis=1)
                    
                    # Top 10 Visualization
                    idx = np.argsort(importance)[-10:]
                    feat_names = np.array(features)[idx]
                    feat_vals = importance[idx]
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                    ax_shap.barh(range(len(idx)), feat_vals, color='#4FC3F7', edgecolor='black')
                    ax_shap.set_yticks(range(len(idx)))
                    ax_shap.set_yticklabels(feat_names, fontsize=10)
                    ax_shap.set_xlabel("Mean |SHAP Value| (Impact on Output)", fontsize=12)
                    ax_shap.set_title("Key Process Drivers (Top 10)", fontsize=14)
                    st.pyplot(fig_shap)
                    
                    st.caption("그래프 해석: 막대가 길수록 해당 공정 변수가 결과 품질에 결정적인 영향을 미쳤음을 의미합니다.")
                else:
                    st.warning("SHAP 값을 계산할 수 없습니다. (데이터 부족 등)")
            else:
                st.warning(f"SHAP 분석은 현재 'XGBoost' 모델에서만 지원됩니다. (선택된 모델: {model_name})")

if __name__ == "__main__":
    main_gui()