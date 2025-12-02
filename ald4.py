# ==============================================================================
# 3D 반도체 소자 구현을 위한 ALD 공정 설계 및 AI 최적화 시스템
# (AI-Driven ALD Process Optimization System)
# 
# [Final Version v12: Error Fixed & Specs Locked]
# 1. TypeError Fixed: Removed callback arguments from main_gui call.
#    -> UI updates are handled internally within ALDOptimizer to support caching.
# 2. Parameters Locked (Strict User Instruction):
#    - XGBoost: 175 trees
#    - Random Forest: 130 trees (Multi-core)
#    - Deep Learning: 100 epochs (Batch 128)
# 3. UI/UX: Forced Black Text, White Background, Mobile Optimization.
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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import textwrap
import time

# 머신러닝 라이브러리
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor 
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
# 2. Deep Learning Model Definition
# ------------------------------------------------------------------------------

class ALDRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ALDRegressor, self).__init__()
        # Lightweight Architecture for Speed
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, output_size)
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
    # Fixed: No callback args here to prevent TypeError
    def __init__(self, file_path: str, mode: str = "cli"):
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [LOCKED PARAMETERS: 175 / 130 / 100]
        self.learning_rate = 0.01 
        self.batch_size = 128       # DL Speed Up
        self.epochs = 100           # DL: 100 (Locked)
        self.best_model_path = 'best_ald_mlp_model.pth'
        self.default_gpc_guess = 1.0 
        
        self.models = {'mlp': None, 'xgboost': None, 'rf': None}
        self.model_weights = {'mlp': 0.33, 'xgboost': 0.33, 'rf': 0.33}
        
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        self.X_scaler = MinMaxScaler()
        self.Y_scaler = MinMaxScaler()
        self.X_imputer = KNNImputer(n_neighbors=5)
        self.Y_imputer = KNNImputer(n_neighbors=5)
        
        self.all_input_cols = []
        self.all_output_cols = []
        
        # Internal UI Placeholders (Created inside to work with caching)
        self.status_container = st.empty() if self.mode == 'gui' else None
        self.progress_bar = st.progress(0) if self.mode == 'gui' else None
        
        # Pipeline Start
        self._update_ui(0.0, "데이터 로드 및 전처리 중...")
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        
        self._update_ui(0.1, "AI 모델 학습 시작...")
        self.performance_df = self._train_ensemble_models()
        
        self._update_ui(1.0, "학습 완료! 시스템 준비됨.")
        time.sleep(0.5)
        if self.mode == "gui":
            self.status_container.empty()
            self.progress_bar.empty()

    def _update_ui(self, value, text):
        if self.mode == "gui":
            self.progress_bar.progress(min(value, 1.0))
            # Blue Bold Text
            self.status_container.markdown(f"<h4 style='color:blue; font-weight:bold;'>🔄 {text}</h4>", unsafe_allow_html=True)

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            df = pd.DataFrame(columns=['Precursor', 'Thickness (nm)']) 
            
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
        ignore_cols = [] 
        
        available_targets = [c for c in target_cols if c in df.columns]
        drop_for_inputs = available_targets + [c for c in ignore_cols if c in df.columns]
        
        self.all_input_cols = df.drop(columns=drop_for_inputs).columns.tolist()
        self.all_output_cols = available_targets

        X_raw = df[self.all_input_cols].values
        Y_raw = df[self.all_output_cols].values
        
        X_imp = self.X_imputer.fit_transform(X_raw)
        Y_imp = self.Y_imputer.fit_transform(Y_raw)
        
        # Physics Data
        X_phys, Y_phys = self._generate_physics_data(X_imp, Y_imp, n_samples=200)
        
        X_combined = np.vstack([X_imp, X_phys])
        Y_combined = np.vstack([Y_imp, Y_phys])
        
        # Augmentation
        X_aug, Y_aug = self._augment_data(X_combined, Y_combined, noise=0.005, multiplier=5)
        
        X_temp, self.X_test, Y_temp, self.Y_test = train_test_split(X_aug, Y_aug, test_size=0.1, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X_temp, Y_temp, test_size=0.15, random_state=42)
        
        # No Poly
        self.X_train_sc = self.X_scaler.fit_transform(self.X_train)
        self.X_val_sc = self.X_scaler.transform(self.X_val)
        self.X_test_sc = self.X_scaler.transform(self.X_test)
        
        self.Y_train_sc = self.Y_scaler.fit_transform(self.Y_train)
        self.Y_val_sc = self.Y_scaler.transform(self.Y_val)
        
        self.input_dim = self.X_train_sc.shape[1]
        self.output_dim = self.Y_train_sc.shape[1]

    def _generate_physics_data(self, X_real, Y_real, n_samples=200):
        X_synth = []
        Y_synth = []
        
        try:
            idx_pulse = [i for i, c in enumerate(self.all_input_cols) if 'Pulse Time' in c][0]
            idx_temp = [i for i, c in enumerate(self.all_input_cols) if 'Temperature' in c][0]
            idx_press = [i for i, c in enumerate(self.all_input_cols) if 'Pressure' in c][0]
            idx_ar = [i for i, c in enumerate(self.all_input_cols) if 'Aspect Ratio' in c][0]
            
            idx_sc = self.all_output_cols.index('Step Coverage (sc, %)')
            idx_gpc = self.all_output_cols.index('GPC (A/cycle)')
        except:
            return X_real, Y_real

        means = np.mean(X_real, axis=0)
        stds = np.std(X_real, axis=0)
        ar_real_vals = X_real[:, idx_ar]
        
        for _ in range(n_samples):
            new_x = means + np.random.normal(0, 1, size=len(means)) * stds
            pulse_val = np.random.uniform(0.05, 2.0)
            new_x[idx_pulse] = pulse_val
            ar_val = np.random.choice(ar_real_vals) 
            new_x[idx_ar] = ar_val

            temp_c = new_x[idx_temp]
            press_torr = new_x[idx_press]
            
            sc_phys, _, _, _, _ = self._calc_physics(temp_c, press_torr, pulse_val, ar_val, "TMA", 100e-9)
            
            sat_factor = pulse_val / (0.2 + pulse_val)
            press_factor = press_torr / (0.1 + press_torr)
            temp_factor = 1.0 + 0.0005 * (temp_c - 250)
            gpc_phys = 1.1 * sat_factor * press_factor * temp_factor
            
            new_y = np.mean(Y_real, axis=0)
            new_y[idx_sc] = sc_phys
            new_y[idx_gpc] = gpc_phys
            
            X_synth.append(new_x)
            Y_synth.append(new_y)
            
        return np.array(X_synth), np.array(Y_synth)

    def _augment_data(self, X, Y, noise=0.01, multiplier=5):
        X_aug, Y_aug = [X], [Y]
        for _ in range(multiplier):
            n = np.random.normal(0, noise, X.shape)
            X_aug.append(X + n * np.std(X, axis=0))
            Y_aug.append(Y)
        return np.vstack(X_aug), np.vstack(Y_aug)

    def _train_ensemble_models(self):
        # 1. XGBoost (175 Trees)
        self._update_ui(0.2, "XGBoost (175 Trees) 학습 중... [1/3]")
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=175, learning_rate=0.05, max_depth=6, n_jobs=-1)
        self.models['xgboost'] = MultiOutputRegressor(xgb_model)
        self.models['xgboost'].fit(self.X_train_sc, self.Y_train_sc)
        
        # 2. Random Forest (130 Trees)
        self._update_ui(0.5, "Random Forest (130 Trees) 학습 중... [2/3]")
        rf_model = RandomForestRegressor(n_estimators=130, max_depth=None, random_state=42, n_jobs=-1)
        self.models['rf'] = rf_model
        self.models['rf'].fit(self.X_train_sc, self.Y_train_sc)
        
        # 3. PyTorch MLP (100 Epochs)
        self._update_ui(0.75, "Deep Learning (100 Epochs) 학습 중... [3/3]")
        self._train_pytorch_mlp()
        
        self._update_ui(0.9, "모델 가중치 최적화 중...")
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
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.models['mlp'](bx)
                loss = torch.mean(loss_weights * (pred - by) ** 2)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                self._update_ui(0.75 + (0.20 * (epoch / self.epochs)), f"Deep Learning 진행 중... Epoch {epoch}/{self.epochs} (Loss: {loss.item():.5f})")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                torch.save(self.models['mlp'].state_dict(), self.best_model_path)
            else:
                patience_counter += 1
                if epoch > 50 and patience_counter >= 10:
                    break
        
        if os.path.exists(self.best_model_path):
            self.models['mlp'].load_state_dict(torch.load(self.best_model_path, weights_only=True))
            os.remove(self.best_model_path)

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

    def _predict_batch(self, df_input):
        X_sc = self.X_scaler.transform(df_input.values)
        Y_sc = self._predict_ensemble(X_sc)
        Y_real = self.Y_scaler.inverse_transform(Y_sc)
        return pd.DataFrame(Y_real, columns=self.all_output_cols)

    def optimize(self, user_input):
        pre, th = user_input["Precursor"], user_input["Thickness (nm)"]
        co, purge = ('H2O' if pre in ['TMA', 'TDMAH'] else 'O3'), "N2"
        ar = user_input["Target AR"]
        cd_m = user_input["CD (nm)"] * 1e-9
        
        # Massive Batch Search
        N = 50000
        temps = np.random.uniform(150, 400, N)
        press = np.random.uniform(0.01, 1.0, N)
        pulses = np.random.uniform(0.05, 2.0, N)
        purges = np.random.uniform(1.0, 10.0, N)
        flows = np.random.uniform(50, 500, N)
        
        base_data = {col: 0.0 for col in self.all_input_cols}
        if f"Precursor_{pre}" in base_data: base_data[f"Precursor_{pre}"] = 1.0
        if f"Co-reactant_{co}" in base_data: base_data[f"Co-reactant_{co}"] = 1.0
        if f"Purge Gas_{purge}" in base_data: base_data[f"Purge Gas_{purge}"] = 1.0
        if "Aspect Ratio (AR)" in base_data: base_data["Aspect Ratio (AR)"] = ar

        df_batch = pd.DataFrame([base_data] * N)
        df_batch["Temperature (c)"] = temps
        df_batch["Pressure (torr)"] = press
        df_batch["Precursor_Pulse Time (s)"] = pulses
        df_batch["Co-reactant_Pulse Time (s)"] = pulses
        df_batch["Purge Time (s)"] = purges
        df_batch["Purge Gas Flow Rate (cm3/min)"] = flows
        
        preds = self._predict_batch(df_batch[self.all_input_cols])
        
        gpcs = preds['GPC (A/cycle)'].values
        roughness = preds['Surface Roughness (RMS, nm)'].values
        uniformity = preds['Uniformity (%)'].values
        
        est_cycles = th / (np.maximum(gpcs, 0.001))
        est_th = gpcs * est_cycles
        
        sc_phys = []
        for i in range(N):
            s, _, _, _, _ = self._calc_physics(temps[i], press[i], pulses[i], ar, pre, cd_m)
            sc_phys.append(s)
        sc_phys = np.array(sc_phys)
        
        cost = (COST_WEIGHTS["roughness"] * roughness**2) + \
               (COST_WEIGHTS["uniformity"] * uniformity**2) + \
               (500 * (est_th - th)**2)
        
        penalty = (sc_phys < 90.0) * 1e9
        total_cost = cost + penalty
        
        best_idx = np.argmin(total_cost)
        best_row = df_batch.iloc[best_idx]
        best_pred = preds.iloc[best_idx]
        
        final_gpc = max(0.001, best_pred['GPC (A/cycle)'])
        final_cycles = int(round(th / final_gpc))
        
        opt_recipe = {
            "Precursor": pre, "Co-reactant": co, "Purge Gas": purge,
            "Temperature (c)": round(best_row["Temperature (c)"], 2),
            "Pressure (torr)": round(best_row["Pressure (torr)"], 3),
            "Cycles (n)": final_cycles,
            "Precursor Pulse Time (s)": round(best_row["Precursor_Pulse Time (s)"], 3),
            "Co-reactant Pulse Time (s)": round(best_row["Co-reactant_Pulse Time (s)"], 3),
            "Purge Time (s)": round(best_row["Purge Time (s)"], 2),
            "Purge Gas Flow Rate (cm3/min)": round(best_row["Purge Gas Flow Rate (cm3/min)"], 0)
        }
        
        final_pred_series = best_pred.copy()
        final_pred_series['Thickness (nm)'] = final_gpc * final_cycles
        
        sc_val, lam, kn, phi, mode = self._calc_physics(
            opt_recipe["Temperature (c)"], opt_recipe["Pressure (torr)"], 
            opt_recipe["Precursor Pulse Time (s)"], ar, pre, cd_m
        )
        phy_info = {"Mean Free Path (λ)": f"{lam:.2e} m", "Knudsen": f"{kn:.2f}", "Thiele Modulus": f"{phi:.4f}", "Mode": mode, "Physics SC": f"{sc_val:.2f}%"}
        
        class Res: fun = total_cost[best_idx]
        res = Res()
        
        return opt_recipe, final_pred_series, phy_info, res

    def analyze_sensitivity(self, recipe, user_input, x_col, y_col):
        norm_recipe = {k.replace(" ", "_"): v for k, v in recipe.items()}
        norm_recipe.update(recipe)
        
        target_val = None
        if x_col in norm_recipe: target_val = norm_recipe[x_col]
        elif x_col.replace("_", " ") in recipe: target_val = recipe[x_col.replace("_", " ")]
        
        if target_val is None: return pd.DataFrame()
            
        if "Pulse Time" in x_col:
            values = np.linspace(0.01, target_val * 1.5, 50)
        else:
            values = np.linspace(target_val * 0.7, target_val * 1.3, 50)
        
        batch_data = []
        base_row = {col: 0.0 for col in self.all_input_cols}
        pre, co, purge = user_input["Precursor"], recipe["Co-reactant"], recipe["Purge Gas"]
        
        for col, val in [("Precursor", pre), ("Co-reactant", co), ("Purge Gas", purge)]:
            if f"{col}_{val}" in base_row: base_row[f"{col}_{val}"] = 1.0
        if "Aspect Ratio (AR)" in base_row: base_row["Aspect Ratio (AR)"] = user_input["Target AR"]
        
        for k, v in recipe.items():
            k_us = k.replace(" ", "_")
            if k in base_row: base_row[k] = v
            elif k_us in base_row: base_row[k_us] = v
            if "Pulse Time" in k:
                 k_special = k.replace("Pulse Time", "_Pulse Time")
                 if k_special in base_row: base_row[k_special] = v

        for v in values:
            row = base_row.copy()
            if x_col in row: row[x_col] = v
            else:
                x_col_alt = x_col.replace(" ", "_")
                if x_col_alt in row: row[x_col_alt] = v
            
            if "Pulse Time" in x_col or "Pulse_Time" in x_col:
                 row["Precursor_Pulse Time (s)"] = v
                 row["Co-reactant_Pulse Time (s)"] = v

            batch_data.append(row)
            
        df_batch = pd.DataFrame(batch_data)
        preds = self._predict_batch(df_batch[self.all_input_cols])
        preds[x_col] = values
        
        sc_list = []
        for i in range(len(values)):
            row = df_batch.iloc[i]
            t = row.get("Temperature (c)", recipe.get("Temperature (c)", 250))
            p = row.get("Pressure (torr)", recipe.get("Pressure (torr)", 0.1))
            
            if "Pulse Time" in x_col or "Pulse_Time" in x_col:
                pt = values[i]
            else:
                pt = recipe.get("Precursor Pulse Time (s)", 0.5)
            
            s, _, _, _, _ = self._calc_physics(t, p, pt, user_input["Target AR"], pre, user_input["CD (nm)"]*1e-9)
            sc_list.append(s)
        preds['Physics SC (%)'] = sc_list
        
        return preds

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
    # [Cache Logic]
    @st.cache_resource
    def get_trained_optimizer():
        csv = "AI_ALD1.csv"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv)
        if not os.path.exists(path): path = csv
        if not os.path.exists(path): return None
        return ALDOptimizer(path, mode="gui")

    st.set_page_config(page_title="AI 기반 ALD 공정 최적화", layout="wide")
    
    # [CSS Injection] Force Text Visibility & Layout
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        .stSelectbox label, .stNumberInput label {
            color: #000000 !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            color: #000000 !important;
            background-color: #f0f2f6 !important;
        }
        .stNumberInput input {
            color: #000000 !important;
            background-color: #f0f2f6 !important;
        }
        .stMarkdown h4 {
            color: #0000FF !important;
        }
        body, p, div, span, td, th {
            color: #000000 !important;
        }
        .block-container { padding-top: 60px !important; padding-bottom: 40px; max-width: 1350px; }
        .cover-box {
            border-radius: 24px;
            padding: 24px 32px;
            margin-bottom: 24px;
            margin-top: 10px;
            background: linear-gradient(135deg, #dff3ff 0%, #ffffff 50%, #f5fff7 100%);
            box-shadow: 0 6px 14px rgba(0,0,0,0.06);
        }
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cover-box">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="display: inline-block; padding: 10px 24px; border-radius: 999px; background: #e5f9e8; color: #176a3a; font-weight: 700; margin-bottom: 10px;">2025 제1회 Google-아주대학교</div>
                    <div style="font-size: 36px; font-weight: 800; color: #111111; margin: 4px 0 10px 0;">AI 기반 ALD 공정 최적화 시스템</div>
                    <div style="font-size: 20px; font-weight: 700; color: #222222;">AI 융합 캡스톤 디자인 대회</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333333;">최종성과발표회</div>
                </div>
                <div style="text-align: right;">
                    <div style="line-height: 1.3; margin-bottom: 6px; font-size: 15px; color: #444444;">Google Developer Student Clubs<br>Ajou University</div>
                    <img src="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png" style="height:40px;">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if 'optimizer' not in st.session_state:
        with st.spinner("AI 모델 초기화 중..."):
            opt_instance = get_trained_optimizer()
            if opt_instance is None:
                st.error("❌ 데이터 파일 'AI_ALD1.csv'을(를) 찾을 수 없습니다.")
                st.stop()
            st.session_state['optimizer'] = opt_instance
        st.success("✅ AI 모델 학습 완료! (Physics-Informed Ensemble)")
    
    # Main Expander
    with st.expander("🎯 공정 목표 설정 (펼치기/접기)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            pre = st.selectbox("전구체 (Precursor)", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
            th = st.number_input("목표 두께 (nm)", 1.0, 200.0, 10.0)
        with c2:
            ar = st.number_input("Target AR (종횡비)", 1.0, 100.0, 10.0)
            cd = st.number_input("CD (nm)", 1.0, 500.0, 100.0)
            
        if st.button("🚀 최적 레시피 도출", use_container_width=True):
            user_input = {"Precursor": pre, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
            optimizer = st.session_state['optimizer']
            with st.spinner("AI가 최적 조건을 탐색 중입니다..."):
                recipe, pred, phy, res = optimizer.optimize(user_input)
                st.session_state.res = (recipe, pred, phy, res, user_input)

    # Reset Button
    if st.sidebar.button("🔄 AI 모델 재학습 (Reset)"):
        st.cache_resource.clear()
        st.session_state.pop('optimizer', None)
        st.rerun()

    if 'res' in st.session_state:
        optimizer = st.session_state['optimizer']
        recipe, pred, phy, res, u_in = st.session_state.res
        
        t1, t2, t3 = st.tabs(["📄 최적 레시피 리포트", "📊 공정 민감도 분석", "🔍 XAI 해석"])
        
        with t1:
            st.markdown("#### 🏆 AI 모델 성능 (Ensemble Test Result)")
            st.dataframe(optimizer.performance_df, use_container_width=True)

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
            st.markdown("<br>", unsafe_allow_html=True)
            
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
                
                y_smooth = gaussian_filter1d(df_sens[yk], sigma=1.5)
                f_interp = interp1d(df_sens[xk], y_smooth, kind='linear', fill_value="extrapolate")
                
                opt_x = recipe.get(xk)
                if opt_x is None:
                    xk_fix = xk.replace("_", " ")
                    if xk_fix in recipe: opt_x = recipe[xk_fix]

                sns.lineplot(x=df_sens[xk], y=y_smooth, marker=None, ax=ax, label='AI Trend (Smoothed)', color='#1f77b4', linewidth=2)
                ax.scatter(df_sens[xk], df_sens[yk], color='gray', alpha=0.3, s=10)
                
                if opt_x is not None:
                    opt_y_visual = f_interp(opt_x)
                    ax.scatter([opt_x], [opt_y_visual], color='red', s=150, zorder=5, label='Optimal')
                
                ymin, ymax = df_sens[yk].min(), df_sens[yk].max()
                if ymax - ymin < 1e-4:
                    ax.set_ylim(ymin - 0.0001, ymax + 0.0001)
                
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                
                if 'Step Coverage (sc, %)' in df_sens.columns:
                    st.markdown("#### ⚖️ Step Coverage Verification (AI vs Physics)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    
                    sc_smooth = gaussian_filter1d(df_sens['Step Coverage (sc, %)'], sigma=1.5)
                    sns.lineplot(x=df_sens[xk], y=sc_smooth, marker=None, ax=ax2, label='AI (Smoothed)', color='#1f77b4')
                    sns.lineplot(data=df_sens, x=xk, y='Physics SC (%)', marker='x', linestyle='--', ax=ax2, label='Physics', color='red')
                    
                    sc_min = min(df_sens['Step Coverage (sc, %)'].min(), df_sens['Physics SC (%)'].min())
                    sc_max = max(df_sens['Step Coverage (sc, %)'].max(), df_sens['Physics SC (%)'].max())
                    margin = (sc_max - sc_min) * 0.1 if sc_max != sc_min else 1.0
                    ax2.set_ylim(max(0, sc_min - margin), min(100.5, sc_max + margin))
                    
                    ax2.set_ylabel("Step Coverage (%)")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
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