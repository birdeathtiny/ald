# ==============================================================================
# 3D 반도체 소자 구현을 위한 ALD 공정 설계 및 AI 최적화 시스템
# (AI-Driven ALD Process Optimization System)
# 
# [Final Stable Version]
# 1. Stability Fix: Removed Polynomial Features (Prevent Overfitting)
# 2. Scaler Update: RobustScaler -> StandardScaler (Training Stability)
# 3. Gradient Clipping added to PyTorch Loop
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import shap

# ------------------------------------------------------------------------------
# 1. 환경 설정 및 상수 정의
# ------------------------------------------------------------------------------

import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    try:
        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    except:
        pass 

N_A = 6.022e23
k_B = 1.38e-23

PRECURSOR_CONSTANTS = {
    "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005, "max_sites_q": 1.0e18},
    "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001, "max_sites_q": 0.8e18},
    "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005, "max_sites_q": 0.5e18},
    "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008, "max_sites_q": 0.6e18}
}

COST_WEIGHTS = {
    "gpc": 10000.0,
    "roughness": 10.0,
    "uniformity": 50.0 
}

# ------------------------------------------------------------------------------
# 2. Deep Learning Model Definition (PyTorch)
# ------------------------------------------------------------------------------

class ALDRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ALDRegressor, self).__init__()
        # 모델 구조 단순화 및 안정화
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer_stack(x)

# ------------------------------------------------------------------------------
# 3. Main Optimization Logic Class
# ------------------------------------------------------------------------------

class ALDOptimizer:
    
    def __init__(self, file_path: str, mode: str = "cli", progress_callback=None):
        self.mode = mode
        self.progress_callback = progress_callback
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 1000
        self.best_model_path = 'best_ald_mlp_model.pth'
        self.default_gpc_guess = 1.0 
        
        self.models = {'mlp': None, 'xgboost': None, 'rf': None}
        self.model_weights = {'mlp': 0.33, 'xgboost': 0.33, 'rf': 0.33}
        
        # [수정] StandardScaler 사용 (가장 안정적)
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.X_imputer = KNNImputer(n_neighbors=5)
        self.Y_imputer = KNNImputer(n_neighbors=5)
        
        self.all_input_cols = []
        self.all_output_cols = []
        
        # Pipeline Start
        self._update_progress(0.0, "데이터 로드 및 전처리 중...")
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        
        self._update_progress(0.1, "앙상블 모델 학습 시작...")
        self.performance_df = self._train_ensemble_models()
        self._update_progress(1.0, "학습 완료! 최적화 준비됨.")

    def _update_progress(self, value, text):
        if self.progress_callback:
            self.progress_callback(value, text)

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            msg = f"[Error] Load Failed: {e}"
            if self.mode == "cli": print(msg); sys.exit(1)
            else: raise FileNotFoundError(f"File not found: {file_path}")
        
        df.replace('-', np.nan, inplace=True)
        
        numeric_cols = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)',
            'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)',
            'Co-reactant Flow Rate (cm3/min)', 'Thickness (nm)', 'Surface Roughness (RMS, nm)',
            'Uniformity (%)', 'Step Coverage (sc, %)', 'Density (g/cm3)', 'GPC (A/cycle)',
            'Aspect Ratio (AR)', 'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)'
        ]
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Co-reactant' in df.columns:
            df['Co-reactant'] = df['Co-reactant'].replace({
                'O3?': 'O3', 'H2O (Implied)': 'H2O',
                'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'
            })
        
        drop_cols = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', 'Leakage Current Density (A/cm2)']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if '순서' in df.columns: df = df.drop(columns=['순서'])
        
        cat_cols = ['Precursor', 'Co-reactant', 'Purge Gas']
        df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], dummy_na=False)
        
        return df_encoded

    def _prepare_datasets(self, df: pd.DataFrame):
        target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)',
            'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)', 'Step Coverage (sc, %)'
        ]
        ignore_cols = ['Aspect Ratio (AR)']
        
        available_targets = [c for c in target_cols if c in df.columns]
        drop_for_inputs = available_targets + [c for c in ignore_cols if c in df.columns]
        
        self.all_input_cols = df.drop(columns=drop_for_inputs).columns.tolist()
        self.all_output_cols = available_targets

        X_raw = df[self.all_input_cols].values
        Y_raw = df[self.all_output_cols].values
        
        X_imp = self.X_imputer.fit_transform(X_raw)
        Y_imp = self.Y_imputer.fit_transform(Y_raw)
        
        # Data Augmentation (2배) - 과적합 방지 위해 노이즈 레벨 조정
        X_aug, Y_aug = self._augment_data(X_imp, Y_imp, noise=0.01, multiplier=2)
        
        # Hold-out Split
        X_temp, self.X_test, Y_temp, self.Y_test = train_test_split(X_aug, Y_aug, test_size=0.1, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X_temp, Y_temp, test_size=0.15, random_state=42)
        
        # Scaling (StandardScaler)
        self.X_train_sc = self.X_scaler.fit_transform(self.X_train)
        self.X_val_sc = self.X_scaler.transform(self.X_val)
        self.X_test_sc = self.X_scaler.transform(self.X_test)
        
        self.Y_train_sc = self.Y_scaler.fit_transform(self.Y_train)
        self.Y_val_sc = self.Y_scaler.transform(self.Y_val)
        
        self.input_dim = self.X_train_sc.shape[1]
        self.output_dim = self.Y_train_sc.shape[1]

    def _augment_data(self, X, Y, noise=0.01, multiplier=2):
        X_aug, Y_aug = [X], [Y]
        for _ in range(multiplier):
            n = np.random.normal(0, noise, X.shape)
            X_aug.append(X + n * np.std(X, axis=0))
            Y_aug.append(Y)
        return np.vstack(X_aug), np.vstack(Y_aug)

    def _train_ensemble_models(self):
        # 1. XGBoost
        self._update_progress(0.15, "XGBoost 학습 중... (1/3)")
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, max_depth=5, n_jobs=-1)
        self.models['xgboost'] = MultiOutputRegressor(xgb_model)
        self.models['xgboost'].fit(self.X_train_sc, self.Y_train_sc)
        
        # 2. Random Forest
        self._update_progress(0.35, "Random Forest 학습 중... (2/3)")
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        self.models['rf'] = rf_model
        self.models['rf'].fit(self.X_train_sc, self.Y_train_sc)
        
        # 3. PyTorch MLP
        self._update_progress(0.55, "Deep Learning (PyTorch MLP) 학습 시작... (3/3)")
        self._train_pytorch_mlp()
        
        # 4. Optimize Weights
        self._update_progress(0.95, "모델 가중치 최적화 중...")
        self._optimize_weights()
        
        return self._evaluate_ensemble()

    def _train_pytorch_mlp(self):
        X_t = torch.FloatTensor(self.X_train_sc).to(self.device)
        Y_t = torch.FloatTensor(self.Y_train_sc).to(self.device)
        
        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.models['mlp'] = ALDRegressor(self.input_dim, self.output_dim).to(self.device)
        optimizer = optim.Adam(self.models['mlp'].parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        loss_weights = torch.ones(self.output_dim).to(self.device)
        try:
            uni_idx = self.all_output_cols.index('Uniformity (%)')
            loss_weights[uni_idx] = 2.0 
        except: pass
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.models['mlp'].train()
            epoch_loss = 0.0
            
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.models['mlp'](bx)
                loss = torch.mean(loss_weights * (pred - by) ** 2)
                loss.backward()
                
                # [수정] Gradient Clipping 추가 (학습 안정화)
                torch.nn.utils.clip_grad_norm_(self.models['mlp'].parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            self.models['mlp'].eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(self.X_val_sc).to(self.device)
                val_y = torch.FloatTensor(self.Y_val_sc).to(self.device)
                val_pred = self.models['mlp'](val_x)
                val_loss = torch.mean(loss_weights * (val_pred - val_y) ** 2).item()
            
            scheduler.step(val_loss)
            
            if epoch % 50 == 0:
                progress = 0.55 + (0.35 * (epoch / self.epochs))
                self._update_progress(progress, f"Deep Learning 학습 중... Epoch {epoch}/{self.epochs} (Loss: {val_loss:.5f})")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.models['mlp'].state_dict(), self.best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= 50:
                    break
                    
        self.models['mlp'].load_state_dict(torch.load(self.best_model_path, weights_only=True))
        if os.path.exists(self.best_model_path): os.remove(self.best_model_path)

    def _optimize_weights(self):
        p_xgb = self.models['xgboost'].predict(self.X_val_sc)
        p_rf = self.models['rf'].predict(self.X_val_sc)
        with torch.no_grad():
            t_val = torch.FloatTensor(self.X_val_sc).to(self.device)
            p_mlp = self.models['mlp'](t_val).cpu().numpy()
            
        y_true = self.Y_val_sc
        mse_xgb = mean_squared_error(y_true, p_xgb)
        mse_rf = mean_squared_error(y_true, p_rf)
        mse_mlp = mean_squared_error(y_true, p_mlp)
        
        # MSE 역수 기반 가중치
        total_inv = (1/(mse_xgb+1e-8)) + (1/(mse_rf+1e-8)) + (1/(mse_mlp+1e-8))
        self.model_weights = {
            'xgboost': (1/(mse_xgb+1e-8)) / total_inv,
            'rf': (1/(mse_rf+1e-8)) / total_inv,
            'mlp': (1/(mse_mlp+1e-8)) / total_inv
        }

    def _predict_ensemble(self, X_scaled):
        p_xgb = self.models['xgboost'].predict(X_scaled)
        p_rf = self.models['rf'].predict(X_scaled)
        with torch.no_grad():
            t_x = torch.FloatTensor(X_scaled).to(self.device)
            p_mlp = self.models['mlp'](t_x).cpu().numpy()
            
        final_pred = (p_xgb * self.model_weights['xgboost'] + 
                      p_rf * self.model_weights['rf'] + 
                      p_mlp * self.model_weights['mlp'])
        return final_pred

    def _evaluate_ensemble(self):
        y_pred_sc = self._predict_ensemble(self.X_test_sc)
        y_pred = self.Y_scaler.inverse_transform(y_pred_sc)
        
        r2 = r2_score(self.Y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(self.Y_test, y_pred, multioutput='raw_values'))
        mae = mean_absolute_error(self.Y_test, y_pred, multioutput='raw_values')
        
        y_mean = np.mean(self.Y_test, axis=0)
        # 0에 가까운 값으로 나누는 것 방지
        safe_mean = np.where(np.abs(y_mean) < 1e-6, 1e-6, y_mean)
        rrmse = (rmse / np.abs(safe_mean)) * 100
        
        return pd.DataFrame({
            'RMSE': dict(zip(self.all_output_cols, rmse.round(4))),
            'MAE': dict(zip(self.all_output_cols, mae.round(4))),
            'RRMSE (%)': dict(zip(self.all_output_cols, rrmse.round(2))),
            'R2': dict(zip(self.all_output_cols, r2.round(4)))
        })

    # --------------------------------------------------------------------------
    # Physics & Optimization Methods
    # --------------------------------------------------------------------------
    
    @staticmethod
    def _calc_physics(T_c, P_torr, pulse_s, AR, precursor, CD_m):
        const = PRECURSOR_CONSTANTS.get(precursor, PRECURSOR_CONSTANTS["TMA"])
        d, M = const["diameter_m"], const["mass_g_mol"] / 1000.0 / N_A
        T_K, P_Pa = T_c + 273.15, P_torr * 133.322
        L = AR * CD_m
        
        v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M))
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
        Kn = lambda_m / L
        
        D_eff = 1.0 / ((1/(1/3 * lambda_m * v_avg + 1e-30)) + (1/(1/3 * v_avg * CD_m + 1e-30)))
        phi = L / (np.sqrt(D_eff * pulse_s + 1e-30) + 1e-30)
        
        sc = 1.0/(1.0+phi) if phi < 1.0 else np.exp(-phi)
        mode = "Reaction Limited" if phi < 1.0 else "Diffusion Limited"
        return float(np.clip(sc * 100, 0, 100)), lambda_m, Kn, phi, mode

    def _predict_recipe(self, params, precursor, co_reactant, purge_gas):
        input_df = pd.DataFrame(columns=self.all_input_cols); input_df.loc[0] = 0.0
        for k, v in params.items():
            if k in input_df.columns: input_df.at[0, k] = v
        for col, val in [("Precursor", precursor), ("Co-reactant", co_reactant), ("Purge Gas", purge_gas)]:
            if f"{col}_{val}" in input_df.columns: input_df.at[0, f"{col}_{val}"] = 1.0
            
        X_sc = self.X_scaler.transform(input_df.values)
        Y_sc = self._predict_ensemble(X_sc)
        Y_real = self.Y_scaler.inverse_transform(Y_sc)[0]
        return pd.Series(Y_real, index=self.all_output_cols)

    def optimize(self, user_input):
        pre, th = user_input["Precursor"], user_input["Thickness (nm)"]
        co, purge = ('H2O' if pre in ['TMA', 'TDMAH'] else 'O3'), "N2"
        
        def objective(x):
            params = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], 
                "Precursor_Pulse Time (s)": x[2], "Purge Time (s)": x[3], 
                "Purge Gas Flow Rate (cm3/min)": x[4], "Cycles (n)": 100, 
                "Co-reactant_Pulse Time (s)": x[2]
            }
            try:
                pred = self._predict_recipe(params, pre, co, purge)
                gpc = pred.get('GPC (A/cycle)', 0.1)
                cycles = th / (gpc + 1e-9)
                cost = (COST_WEIGHTS["roughness"] * (pred.get('Surface Roughness (RMS, nm)', 10))**2) + \
                       (COST_WEIGHTS["uniformity"] * (pred.get('Uniformity (%)', 100))**2)
                est_th = gpc * cycles
                cost += 500 * (est_th - th)**2 
                return cost
            except: return 1e9

        def constraint(x):
            sc, _, _, _, _ = self._calc_physics(x[0], x[1], x[2], user_input["Target AR"], pre, user_input["CD (nm)"]*1e-9)
            return sc - 90.0

        bounds = [(150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500)]
        res = minimize(objective, [250, 0.1, 0.5, 5.0, 100], method='SLSQP', bounds=bounds, constraints={'type':'ineq', 'fun':constraint})
        
        x = res.x
        rounded_vals = [round(v, 3) for v in x]
        
        temp_params = {
            "Temperature (c)": rounded_vals[0], "Pressure (torr)": rounded_vals[1], 
            "Precursor_Pulse Time (s)": rounded_vals[2], "Co-reactant_Pulse Time (s)": rounded_vals[2],
            "Purge Time (s)": rounded_vals[3], "Purge Gas Flow Rate (cm3/min)": rounded_vals[4], "Cycles (n)": 100
        }
        pred_res = self._predict_recipe(temp_params, pre, co, purge)
        gpc = max(0.001, pred_res.get('GPC (A/cycle)', 0.1))
        final_cycles = int(round(th / gpc))
        
        opt_recipe = temp_params.copy()
        opt_recipe["Cycles (n)"] = final_cycles
        opt_recipe["Precursor"] = pre; opt_recipe["Co-reactant"] = co; opt_recipe["Purge Gas"] = purge
        
        final_pred = self._predict_recipe(opt_recipe, pre, co, purge)
        final_pred['Thickness (nm)'] = gpc * final_cycles
        
        sc_val, lam, kn, phi, mode = self._calc_physics(rounded_vals[0], rounded_vals[1], rounded_vals[2], user_input["Target AR"], pre, user_input["CD (nm)"]*1e-9)
        phy_info = {"Mean Free Path (λ)": f"{lam:.2e} m", "Knudsen": f"{kn:.2f}", "Thiele Modulus": f"{phi:.4f}", "Mode": mode, "Physics SC": f"{sc_val:.2f}%"}
        return opt_recipe, final_pred, phy_info, res

    def analyze_sensitivity(self, recipe, user_input, x_col, y_col):
        if x_col not in recipe: return pd.DataFrame()
        base_val = recipe[x_col]
        values = np.linspace(base_val * 0.7, base_val * 1.3, 20)
        results = []
        pre, co, purge = user_input["Precursor"], recipe["Co-reactant"], recipe["Purge Gas"]
        
        for v in values:
            temp = recipe.copy(); temp[x_col] = v
            pred = self._predict_recipe(temp, pre, co, purge)
            phys_sc, _, _, _, _ = self._calc_physics(temp["Temperature (c)"], temp["Pressure (torr)"], temp["Precursor_Pulse Time (s)"], user_input["Target AR"], pre, user_input["CD (nm)"]*1e-9)
            row = {x_col: v}; row.update(pred.to_dict()); row['Physics SC (%)'] = phys_sc
            results.append(row)
        return pd.DataFrame(results)

    def get_shap(self):
        try: sc_idx = self.all_output_cols.index('Step Coverage (sc, %)')
        except: sc_idx = 0
        model = self.models['xgboost'].estimators_[sc_idx]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(self.X_test_sc)
        return explainer, shap_vals, self.X_test_sc, self.all_input_cols

# ------------------------------------------------------------------------------
# 4. Streamlit GUI
# ------------------------------------------------------------------------------

def main_gui():
    st.set_page_config(page_title="AI 기반 ALD 공정 최적화", layout="wide")
    st.title("✨ AI 기반 ALD 공정 최적화 시스템 (Pro Ver.)")
    
    # Session State Init
    if 'optimizer' not in st.session_state:
        csv = "AI_ALD1.csv"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv)
        if not os.path.exists(path): path = csv
        
        if not os.path.exists(path):
            st.error(f"❌ 데이터 파일 '{csv}'을(를) 찾을 수 없습니다."); st.stop()

        # Progress UI
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(val, text):
            prog_bar.progress(val)
            status_text.text(text)

        try:
            opt_instance = ALDOptimizer(path, mode="gui", progress_callback=update_progress)
            st.session_state['optimizer'] = opt_instance
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {e}"); st.stop()
            
        prog_bar.empty(); status_text.empty()
        st.success("✅ AI 모델 학습 완료! 시스템이 준비되었습니다.")
        
    optimizer = st.session_state['optimizer']

    # Sidebar Input
    st.sidebar.header("🎯 공정 목표 설정")
    pre = st.sidebar.selectbox("전구체 (Precursor)", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("목표 두께 (nm)", 1.0, 200.0, 15.0)
    ar = st.sidebar.number_input("Target AR (종횡비)", 1.0, 100.0, 10.0)
    cd = st.sidebar.number_input("CD (nm)", 1.0, 500.0, 100.0)
    
    if st.sidebar.button("🚀 최적 레시피 도출"):
        user_input = {"Precursor": pre, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("AI가 최적 조건을 탐색 중입니다... (Ensemble Prediction)"):
            recipe, pred, phy, res = optimizer.optimize(user_input)
            st.session_state.res = (recipe, pred, phy, res, user_input)
            
    if 'res' in st.session_state:
        recipe, pred, phy, res, u_in = st.session_state.res
        
        t1, t2, t3 = st.tabs(["📄 최적 레시피 리포트", "📊 공정 민감도 분석", "🔍 XAI 해석"])
        
        with t1:
            st.markdown("#### 🏆 AI 모델 성능 (Ensemble Test Result)")
            st.dataframe(optimizer.performance_df, use_container_width=True)
            st.info(f"가중치 최적화 결과: XGB({optimizer.model_weights['xgboost']:.2f}) / RF({optimizer.model_weights['rf']:.2f}) / MLP({optimizer.model_weights['mlp']:.2f})")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🧪 도출된 공정 레시피")
                st.dataframe(pd.DataFrame.from_dict(recipe, orient='index', columns=['Value']), use_container_width=True)
                st.success(f"최적화 비용 함수 값: {res.fun:.4f}")
            with c2:
                st.markdown("### 🔮 AI 예측 물성")
                pred_disp = pred.to_frame(name='Predicted')
                target_th = u_in['Thickness (nm)']
                pred_th = pred['Thickness (nm)']
                st.metric("Thickness (nm)", f"{pred_th:.4f}", f"{pred_th - target_th:.4f}")
                st.dataframe(pred_disp.drop('Thickness (nm)'), use_container_width=True)
                
            st.markdown("### ⚛️ 물리 모델 검증 (Physics Check)")
            st.info(f"Step Coverage: {phy['Physics SC']} ({phy['Mode']}) | Knudsen Number: {phy['Knudsen']}")

        with t2:
            st.markdown("### 📈 Sensitivity Analysis (민감도 분석)")
            opts = {
                "GPC vs Temperature": ("Temperature (c)", "GPC (A/cycle)"),
                "GPC vs Pulse Time": ("Precursor_Pulse Time (s)", "GPC (A/cycle)"),
                "SC vs Pulse Time": ("Precursor_Pulse Time (s)", "Step Coverage (sc, %)"),
                "GPC vs Pressure": ("Pressure (torr)", "GPC (A/cycle)")
            }
            choice = st.selectbox("분석할 관계 선택:", list(opts.keys()))
            xk, yk = opts[choice]
            
            df_sens = optimizer.analyze_sensitivity(recipe, u_in, xk, yk)
            
            if not df_sens.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_sens, x=xk, y=yk, marker='o', ax=ax, label='Trend')
                
                opt_x, opt_y = recipe.get(xk), pred.get(yk)
                if opt_x is not None:
                    ax.scatter([opt_x], [opt_y], color='red', s=150, zorder=5, label='Optimal')
                
                ymin, ymax = df_sens[yk].min(), df_sens[yk].max()
                if ymax - ymin < 1e-4:
                    ax.set_ylim(ymin - 0.0001, ymax + 0.0001)
                
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                
                if 'Step Coverage (sc, %)' in df_sens.columns:
                    st.markdown("#### ⚖️ Step Coverage Verification (AI vs Physics)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=df_sens, x=xk, y='Step Coverage (sc, %)', marker='o', ax=ax2, label='AI')
                    sns.lineplot(data=df_sens, x=xk, y='Physics SC (%)', marker='x', linestyle='--', ax=ax2, label='Physics', color='red')
                    
                    sc_min = min(df_sens['Step Coverage (sc, %)'].min(), df_sens['Physics SC (%)'].min())
                    sc_max = max(df_sens['Step Coverage (sc, %)'].max(), df_sens['Physics SC (%)'].max())
                    margin = (sc_max - sc_min) * 0.1 if sc_max != sc_min else 1.0
                    ax2.set_ylim(max(0, sc_min - margin), min(100.5, sc_max + margin))
                    
                    ax2.set_ylabel("Step Coverage (%)")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)

        with t3:
            st.markdown("### 🧠 Explainable AI (SHAP)")
            if st.button("SHAP 분석 시작"):
                with st.spinner("Calculating feature importance..."):
                    exp, vals, X_test, feats = optimizer.get_shap()
                    fig, ax = plt.subplots()
                    shap.summary_plot(vals, X_test, feature_names=feats, show=False)
                    st.pyplot(fig)

if __name__ == "__main__":
    main_gui()