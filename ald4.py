# --- 0. 기본 라이브러리 및 Streamlit 임포트 ---
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker  # 축 서식 정밀 제어를 위해 추가
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import itertools
import xgboost as xgb
import shap

# 한글 폰트 설정
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

# --- 1. 물리/화학 상수 테이블 정의 ---
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
    "roughness": 10.0
}

# --- 2. AI 모델 클래스 정의 (MLP) ---
class ALDRegressor_Optimized(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ALDRegressor_Optimized, self).__init__()
        self.output_size = output_size
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.layer_stack(x)

# --- 3. ALD 최적화 메인 클래스 ---
class ALDOptimizer:
    
    def __init__(self, file_path: str, mode: str = "cli"):
        self.mode = mode
        
        self.final_learning_rate = 0.00195
        self.final_dropout_rate = 0.28
        self.final_batch_size = 16
        self.final_epochs = 500
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 30
        self.WEIGHT_DECAY = 1e-5
        self.BEST_MODEL_PATH = 'best_ald_model.pth'
        self.DEFAULT_GPC_GUESS_A = 1.0 
        self.model_comparison_results = {}
        self.performance_df = None
        self.xgb_model = None

        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.final_model = None
        self.ALL_INPUT_FEATURES_ORDERED = []
        self.ALL_OUTPUT_FEATURES_ORDERED = []
        
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        
        self._hyperparameter_search() 
        self._compare_with_rf()
        self._train_model()
        
        _, _, _, _ = self._evaluate_model(torch.from_numpy(self.X_test_scaled).float(), torch.from_numpy(self.Y_test_scaled).float())

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            msg = f"\n[치명적 오류] 파일 로드 실패: {file_path}. 오류: {e}"
            if self.mode == "cli": print(msg); sys.exit(1)
            else: raise FileNotFoundError(f"'{file_path}' 파일을 찾거나 로드할 수 없습니다. 경로를 확인하세요.")
        
        df.replace('-', np.nan, inplace=True)
        
        cols_to_convert = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)',
            'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)',
            'Co-reactant Flow Rate (cm3/min)', 'Thickness (nm)', 'Surface Roughness (RMS, nm)',
            'Uniformity (%)', 'Step Coverage (sc, %)', 'Density (g/cm3)', 'GPC (A/cycle)',
            'Aspect Ratio (AR)', 'Dielectric Constant (ε)',
            'Breakdown Field (MV/cm)'
        ]
        for col in cols_to_convert:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Co-reactant' in df.columns:
            df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
            df['Co-reactant'] = df['Co-reactant'].replace({'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'})
        
        cols_to_drop_high_nan = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', 'Leakage Current Density (A/cm2)']
        existing_cols_to_drop = [c for c in cols_to_drop_high_nan if c in df.columns]
        df_processed = df.drop(columns=existing_cols_to_drop)
        
        categorical_cols = ['Precursor', 'Co-reactant', 'Purge Gas']
        existing_cat_cols = [c for c in categorical_cols if c in df_processed.columns]
        if '순서' in df_processed.columns: df_processed = df_processed.drop(columns=['순서'])
        
        df_encoded = pd.get_dummies(df_processed, columns=existing_cat_cols, dummy_na=False)
        return df_encoded

    def _prepare_datasets(self, df_encoded: pd.DataFrame):
        target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)',
            'Dielectric Constant (ε)',
            'Breakdown Field (MV/cm)', 'Step Coverage (sc, %)'
        ]
        cols_to_ignore_for_ai = ['Aspect Ratio (AR)']
        available_target_cols = [col for col in target_cols if col in df_encoded.columns]
        
        cols_to_drop = available_target_cols + [c for c in cols_to_ignore_for_ai if c in df_encoded.columns]
        self.ALL_INPUT_FEATURES_ORDERED = df_encoded.drop(columns=cols_to_drop).columns.tolist()
        self.ALL_OUTPUT_FEATURES_ORDERED = available_target_cols

        X = df_encoded[self.ALL_INPUT_FEATURES_ORDERED].values
        Y = df_encoded[self.ALL_OUTPUT_FEATURES_ORDERED].values
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_imputer = KNNImputer(n_neighbors=5); X_train_imputed = X_imputer.fit_transform(X_train); X_test_imputed = X_imputer.transform(X_test)
        Y_imputer = KNNImputer(n_neighbors=5); Y_train_imputed = Y_imputer.fit_transform(Y_train); self.Y_test_unscaled = Y_imputer.transform(Y_test)
        
        self.X_test_scaled = self.X_scaler.fit_transform(X_test_imputed)
        self.Y_test_scaled = self.Y_scaler.fit_transform(self.Y_test_unscaled)
        
        self.X_train_scaled = self.X_scaler.fit_transform(X_train_imputed)
        self.Y_train_scaled = self.Y_scaler.fit_transform(Y_train_imputed)
        self.Y_train_unscaled = Y_train_imputed
        
        X_train_final_idx, X_val_idx, _, _ = train_test_split(
            np.arange(self.X_train_scaled.shape[0]), np.arange(self.Y_train_scaled.shape[0]), test_size=self.VALIDATION_SPLIT, random_state=42
        )
        self.X_train_final = self.X_train_scaled[X_train_final_idx]
        self.Y_train_final = self.Y_train_scaled[X_train_final_idx]
        self.X_val = self.X_train_scaled[X_val_idx]
        self.Y_val = self.Y_train_scaled[X_val_idx]

        self.INPUT_SIZE = self.X_train_scaled.shape[1]
        self.OUTPUT_SIZE = self.Y_train_scaled.shape[1]

    def _hyperparameter_search(self):
        LR_CANDIDATES = [0.001, 0.002]; DO_CANDIDATES = [0.2, 0.3]; best_val_loss = float('inf')
        
        for lr, do in itertools.product(LR_CANDIDATES, DO_CANDIDATES):
            model_temp = ALDRegressor_Optimized(self.INPUT_SIZE, self.OUTPUT_SIZE, do)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model_temp.parameters(), lr=lr)

            X_val_tensor = torch.from_numpy(self.X_val).float(); Y_val_tensor = torch.from_numpy(self.Y_val).float()
            X_train_tensor = torch.from_numpy(self.X_train_final).float(); Y_train_tensor = torch.from_numpy(self.Y_train_final).float()

            train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.final_batch_size, shuffle=True)
            
            for epoch in range(50):
                model_temp.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad(); outputs = model_temp(inputs); loss = criterion(outputs, targets); loss.backward(); optimizer.step()

            model_temp.eval()
            with torch.no_grad():
                val_loss = criterion(model_temp(X_val_tensor), Y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.final_learning_rate = lr
                self.final_dropout_rate = do

    def _compare_with_rf(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        X_combined = np.concatenate([self.X_train_final, self.X_val])
        Y_unscaled_combined = self.Y_scaler.inverse_transform(np.concatenate([self.Y_train_final, self.Y_val]))

        rf_model.fit(X_combined, Y_unscaled_combined)
        Y_pred_rf = rf_model.predict(self.X_test_scaled)

        r2_rf = r2_score(self.Y_test_unscaled, Y_pred_rf, multioutput='raw_values')
        rmse_rf = np.sqrt(mean_squared_error(self.Y_test_unscaled, Y_pred_rf, multioutput='raw_values'))
        
        self.model_comparison_results['RandomForest'] = {
            'R2_mean': r2_rf.mean().round(4), 
            'RMSE_mean': rmse_rf.mean().round(4), 
            'R2_GPC': r2_rf[self.ALL_OUTPUT_FEATURES_ORDERED.index('GPC (A/cycle)')].round(4)
        }

    def _train_and_compare_xgboost(self):
        xgb_estimator = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            n_jobs=-1
        )
        self.xgb_model = MultiOutputRegressor(xgb_estimator)
        self.xgb_model.fit(self.X_train_final, self.Y_train_final)
        
        Y_pred_xgb_scaled = self.xgb_model.predict(self.X_test_scaled)
        Y_pred_xgb = self.Y_scaler.inverse_transform(Y_pred_xgb_scaled)
        
        r2_xgb = r2_score(self.Y_test_unscaled, Y_pred_xgb, multioutput='raw_values')
        rmse_xgb = np.sqrt(mean_squared_error(self.Y_test_unscaled, Y_pred_xgb, multioutput='raw_values'))
        
        self.model_comparison_results['XGBoost'] = {
            'R2_mean': r2_xgb.mean().round(4),
            'RMSE_mean': rmse_xgb.mean().round(4),
            'R2_GPC': r2_xgb[self.ALL_OUTPUT_FEATURES_ORDERED.index('GPC (A/cycle)')].round(4)
        }
        return self.model_comparison_results

    def get_shap_explainer(self):
        if self.xgb_model is None:
            self._train_and_compare_xgboost()

        try:
            sc_index = self.ALL_OUTPUT_FEATURES_ORDERED.index('Step Coverage (sc, %)')
        except ValueError:
            sc_index = 0
        
        target_model = self.xgb_model.estimators_[sc_index]
        explainer = shap.TreeExplainer(target_model)
        shap_values = explainer.shap_values(self.X_test_scaled)
        return explainer, shap_values, self.X_test_scaled

    def _train_model(self):
        X_train_tensor = torch.from_numpy(self.X_train_final).float(); Y_train_tensor = torch.from_numpy(self.Y_train_final).float()
        X_val_tensor = torch.from_numpy(self.X_val).float(); Y_val_tensor = torch.from_numpy(self.Y_val).float()
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor); val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.final_batch_size, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=self.final_batch_size, shuffle=False)
        
        model = ALDRegressor_Optimized(self.INPUT_SIZE, self.OUTPUT_SIZE, self.final_dropout_rate)
        criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=self.final_learning_rate, weight_decay=self.WEIGHT_DECAY)
        best_val_loss = float('inf'); patience_counter = 0

        for epoch in range(self.final_epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets); loss.backward(); optimizer.step()
            
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs); loss = criterion(outputs, targets); val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss; torch.save(model.state_dict(), self.BEST_MODEL_PATH); patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.PATIENCE: break
        
        model.load_state_dict(torch.load(self.BEST_MODEL_PATH)); model.eval()
        self.final_model = model
        if os.path.exists(self.BEST_MODEL_PATH): os.remove(self.BEST_MODEL_PATH)
        
        self.model_comparison_results['MLP (Deep Learning)'] = {'R2_mean': 0, 'RMSE_mean': 0, 'R2_GPC': 0}

    def _evaluate_model(self, X_test_tensor, Y_test_tensor):
        if self.final_model is None: return 0.0, {}, {}, {}

        self.final_model.eval()
        with torch.no_grad(): Y_pred_scaled_tensor = self.final_model(X_test_tensor)
            
        Y_pred_unscaled = self.Y_scaler.inverse_transform(Y_pred_scaled_tensor.numpy())
        test_rmse = np.sqrt(mean_squared_error(self.Y_test_unscaled, Y_pred_unscaled))
        R2_scores = r2_score(self.Y_test_unscaled, Y_pred_unscaled, multioutput='raw_values')
        RMSE_scores = np.sqrt(mean_squared_error(self.Y_test_unscaled, Y_pred_unscaled, multioutput='raw_values'))
        
        Y_test_avg = np.mean(self.Y_test_unscaled, axis=0)
        RRMSE_scores = (RMSE_scores / (Y_test_avg + 1e-8)) * 100
        
        R2_dict = dict(zip(self.ALL_OUTPUT_FEATURES_ORDERED, R2_scores.round(4)))
        RMSE_dict = dict(zip(self.ALL_OUTPUT_FEATURES_ORDERED, RMSE_scores.round(4)))
        RRMSE_dict = dict(zip(self.ALL_OUTPUT_FEATURES_ORDERED, RRMSE_scores.round(2)))
        
        self.model_comparison_results['MLP (Deep Learning)'] = {
            'R2_mean': R2_scores.mean().round(4), 
            'RMSE_mean': test_rmse.round(4), 
            'R2_GPC': R2_dict.get('GPC (A/cycle)', 0)
        }
        
        self.performance_df = pd.DataFrame({'RMSE': RMSE_dict, 'R^2': R2_dict, 'RRMSE (%)': RRMSE_dict})

        return test_rmse, RMSE_dict, R2_dict, RRMSE_dict

    @staticmethod
    def _calculate_physical_parameters(T_celsius, P_torr, precursor_name, L_feature_m):
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        d_precursor_m = const["diameter_m"]; T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
        Kn = lambda_m / L_feature_m
        return lambda_m, Kn

    @staticmethod
    def _calculate_physics_sc_details(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        d = const["diameter_m"]; M_kg = const["mass_g_mol"] / 1000.0 / N_A
        T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322
        L_m = AR_value * CD_m
        v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_kg))
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
        D_A = (1.0 / 3.0) * lambda_m * v_avg
        D_Kn = (1.0 / 3.0) * v_avg * CD_m
        D_eff = 1.0 / ((1.0 / (D_A + 1e-30)) + (1.0 / (D_Kn + 1e-30)))
        lambda_pen = np.sqrt(D_eff * Pulse_Time_s + 1e-30)
        phi = L_m / (lambda_pen + 1e-30)
        
        if phi < 1.0: SC_fraction = 1.0 / (1.0 + phi); mode = "Reaction Limited (RDS, φ < 1)"
        else: SC_fraction = np.exp(-phi); mode = "Diffusion Limited (RDS, φ ≥ 1)"
            
        return float(np.clip(SC_fraction * 100.0, 0.0, 100.0)), float(phi), mode

    def _calculate_physics_sc(self, P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
        sc, _, _ = self._calculate_physics_sc_details(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m)
        return sc

    def _create_model_input(self, recipe_params, precursor_name, co_reactant_name, purge_gas_name) -> pd.DataFrame:
        input_df = pd.DataFrame(columns=self.ALL_INPUT_FEATURES_ORDERED); input_df.loc[0] = 0.0
        for key, value in recipe_params.items():
            if key in input_df.columns: input_df.at[0, key] = value
        
        if f"Precursor_{precursor_name}" in input_df.columns: input_df.at[0, f"Precursor_{precursor_name}"] = 1.0
        if f"Co-reactant_{co_reactant_name}" in input_df.columns: input_df.at[0, f"Co-reactant_{co_reactant_name}"] = 1.0
        if f"Purge Gas_{purge_gas_name}" in input_df.columns: input_df.at[0, f"Purge Gas_{purge_gas_name}"] = 1.0
        return input_df

    def _predict_from_recipe(self, recipe_params, precursor_name, co_reactant_name, purge_gas_name) -> pd.Series:
        input_df = self._create_model_input(recipe_params, precursor_name, co_reactant_name, purge_gas_name)
        X_scaled = self.X_scaler.transform(input_df.values); X_tensor = torch.from_numpy(X_scaled).float()
        self.final_model.eval()
        with torch.no_grad(): Y_pred_scaled_tensor = self.final_model(X_tensor)
        Y_pred_unscaled = self.Y_scaler.inverse_transform(Y_pred_scaled_tensor.numpy())[0]
        return pd.Series(Y_pred_unscaled, index=self.ALL_OUTPUT_FEATURES_ORDERED)

    def _constraint_sc(self, x, user_input, co_reactant_name, purge_gas_name, cost_weights, fixed_cycles_n) -> float:
        target_ar = user_input["Target AR"]
        CD_m = user_input["CD (nm)"] * 1e-9 
        TARGET_SC_MIN = 98.0 if target_ar <= 5 else 90.0 if target_ar <= 15 else 85.0
        phys_sc = self._calculate_physics_sc(x[1], x[0], x[2], target_ar, user_input["Precursor"], CD_m)
        return phys_sc - TARGET_SC_MIN

    def _objective_function(self, x, user_input, co_reactant_name, purge_gas_name, cost_weights, fixed_cycles_n) -> float:
        recipe_params = {
            "Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
            "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
            "Cycles (n)": fixed_cycles_n, "Co-reactant_Pulse Time (s)": x[2]
        }
        try:
            pred = self._predict_from_recipe(recipe_params, user_input["Precursor"], co_reactant_name, purge_gas_name)
        except: return 1e9
        
        target_gpc = (user_input["Thickness (nm)"] * 10) / (fixed_cycles_n + 1e-6)
        
        cost = (cost_weights["gpc"] * (pred.get('GPC (A/cycle)', 0) - target_gpc)**2) + \
               (cost_weights["roughness"] * (pred.get('Surface Roughness (RMS, nm)', 10)/5.0)**2)
        
        return cost

    def generate_optimal_recipe(self, user_input: Dict[str, Any], silent: bool = False):
        precursor = user_input["Precursor"]; thickness = user_input["Thickness (nm)"]
        co_reactant = 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3'; purge_gas = "N2"
        initial_cycles = max(10, int(round((thickness * 10) / self.DEFAULT_GPC_GUESS_A)))
        
        bounds = [(150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500)]
        initial_guess = [np.random.uniform(l, h) for l, h in bounds]

        args = (user_input, co_reactant, purge_gas, COST_WEIGHTS, initial_cycles)
        
        result = minimize(self._objective_function, initial_guess, args=args, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': self._constraint_sc, 'args': args}, 
                          options={'maxiter': 100, 'eps': 1e-6, 'ftol': 1e-7})
        
        x = result.x
        
        check_params = {"Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                        "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4], "Cycles (n)": initial_cycles, "Co-reactant_Pulse Time (s)": x[2]}
        check_res = self._predict_from_recipe(check_params, precursor, co_reactant, purge_gas)
        final_gpc = check_res.get('GPC (A/cycle)', 1.0); final_gpc = max(0.001, final_gpc)
        final_cycles = max(10, int(round((thickness * 10) / final_gpc)))

        final_params = check_params.copy(); final_params["Cycles (n)"] = final_cycles
        final_pred = self._predict_from_recipe(final_params, precursor, co_reactant, purge_gas)
        final_pred['Thickness (nm)'] = (final_gpc * final_cycles) / 10.0 

        opt_recipe = {"Precursor": precursor, "Co-reactant": co_reactant, "Temperature (c)": round(x[0], 2), "Pressure (torr)": round(x[1], 3),
                      "Cycles (n)": final_cycles, "Precursor Pulse Time (s)": round(x[2], 3), "Co-reactant Pulse Time (s)": round(x[2], 3),
                      "Purge Time (s)": round(x[3], 2), "Purge Gas Flow Rate (cm3/min)": round(x[4], 0), "Purge Gas": purge_gas}
        
        sc_val, phi, mode = self._calculate_physics_sc_details(x[1], x[0], x[2], user_input["Target AR"], precursor, user_input["CD (nm)"]*1e-9)
        lambda_m, Kn = self._calculate_physical_parameters(x[0], x[1], precursor, user_input["CD (nm)"]*1e-9)
        valid_data = {"Mean Free Path (λ) [m]": f"{lambda_m:.2e}", "Knudsen Number (Kn)": f"{Kn:.2f}", 
                      "Thiele Modulus (φ)": f"{phi:.4f}", "Transport Mode": mode, "SC (Full Model)": f"{sc_val:.4f} %"}
                              
        stats = {"Optimization Success": result.success, "Message": result.message, "Iterations": result.nit, "Final Cost": f"{result.fun:.6f}"}

        return opt_recipe, final_pred, valid_data, stats

    # --- NEW: 공정 민감도 분석 (목표값 고정, 변수 변화) ---
    def analyze_process_sensitivity(self, opt_recipe, user_input, x_col, y_col):
        try:
            current_x_val = opt_recipe[x_col]
        except KeyError:
            return pd.DataFrame() 

        # 스윕 범위 설정 (현재 값 기준 ±20%)
        sweep_range = np.linspace(current_x_val * 0.8, current_x_val * 1.2, 20)
        
        results = []
        
        for val in sweep_range:
            temp_recipe = opt_recipe.copy()
            temp_recipe[x_col] = val
            
            sim_params = {
                "Temperature (c)": temp_recipe.get("Temperature (c)", opt_recipe["Temperature (c)"]),
                "Pressure (torr)": temp_recipe.get("Pressure (torr)", opt_recipe["Pressure (torr)"]),
                "Precursor_Pulse Time (s)": temp_recipe.get("Precursor Pulse Time (s)", opt_recipe["Precursor Pulse Time (s)"]),
                "Co-reactant_Pulse Time (s)": temp_recipe.get("Co-reactant Pulse Time (s)", opt_recipe["Co-reactant Pulse Time (s)"]),
                "Purge Time (s)": temp_recipe.get("Purge Time (s)", opt_recipe["Purge Time (s)"]),
                "Purge Gas Flow Rate (cm3/min)": temp_recipe.get("Purge Gas Flow Rate (cm3/min)", opt_recipe["Purge Gas Flow Rate (cm3/min)"]),
                "Cycles (n)": temp_recipe.get("Cycles (n)", opt_recipe["Cycles (n)"])
            }
            
            pred = self._predict_from_recipe(sim_params, user_input["Precursor"], temp_recipe["Co-reactant"], temp_recipe["Purge Gas"])
            
            phys_sc = self._calculate_physics_sc(
                sim_params["Pressure (torr)"],
                sim_params["Temperature (c)"],
                sim_params["Precursor_Pulse Time (s)"],
                user_input["Target AR"],
                user_input["Precursor"],
                user_input["CD (nm)"] * 1e-9
            )
            
            row = {x_col: val}
            row.update(pred.to_dict())
            row['Physics SC (%)'] = phys_sc
            
            for k, v in sim_params.items():
                row[k] = v
                
            results.append(row)
            
        return pd.DataFrame(results)


# ==========================================
# 🌐 Streamlit GUI 모드 실행 함수
# ==========================================
def main_gui():
    st.set_page_config(page_title="AI 기반 ALD 공정 최적화", layout="wide")
    st.title("✨ AI 기반 ALD 공정 최적화 시스템")

    @st.cache_resource(show_spinner="AI 모델 및 데이터 준비 중 (초기 1회만 실행)...")
    def load_optimizer(): 
        csv_file_name = "AI_ALD1.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_file_path = os.path.join(current_dir, csv_file_name)
        if not os.path.exists(full_file_path):
            full_file_path = csv_file_name 
        if not os.path.exists(full_file_path):
            st.error(f"❌ 데이터 파일 '{csv_file_name}'을(를) 찾을 수 없습니다.")
            st.stop()
        return ALDOptimizer(file_path=full_file_path, mode="gui") 

    try: optimizer = load_optimizer()
    except Exception as e: st.error(f"모델 로드/학습 실패: {e}"); st.stop()

    if 'xgb_trained' not in st.session_state:
        with st.spinner("AI 모델 고도화 진행 중... (MLP vs XGBoost 성능 비교 & SHAP 준비)"):
            optimizer._train_and_compare_xgboost()
        st.session_state.xgb_trained = True

    st.sidebar.header("🎯 목표 조건 입력")
    sel_p = st.sidebar.selectbox("전구체 선택", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"], key='precursor_select')
    th = st.sidebar.number_input("목표 두께 (nm)", 1.0, 200.0, 15.0, key='thickness_input')
    ar = st.sidebar.number_input("목표 AR (Aspect Ratio)", 1.0, 100.0, 10.0, key='ar_input')
    cd = st.sidebar.number_input("CD (Critical Dimension, nm)", 1.0, 500.0, 100.0, key='cd_input')

    if 'opt_result' not in st.session_state:
        st.session_state.opt_result = None

    if st.sidebar.button("🚀 최적 레시피 생성", type="primary"):
        user_input = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("최적화 진행 중 (SLSQP)..."):
            opt_recipe, pred_results, val_data, opt_stats = optimizer.generate_optimal_recipe(user_input=user_input)
            st.session_state.opt_result = (opt_recipe, pred_results, val_data, opt_stats, user_input)
        st.sidebar.success("✅ 최적화 완료!")
        
    if st.session_state.opt_result:
        opt_recipe, pred_results, val_data, opt_stats, user_input = st.session_state.opt_result
        
        tab1, tab2, tab3 = st.tabs(["📄 결과 리포트", "📊 공정 민감도 분석", "🔍 AI 해석 (XAI)"])

        with tab1:
            st.subheader("모델 선택 근거 및 성능 평가")
            col_perf1, col_perf2 = st.columns([1, 2])
            with col_perf1:
                st.markdown("##### 1) 모델별 성능 비교 (R² Score)")
                comp_df = pd.DataFrame(optimizer.model_comparison_results).T
                comp_df.index.name = "Model"
                st.dataframe(comp_df, use_container_width=True)
            with col_perf2:
                st.markdown("##### 2) 최종 선택 모델(MLP) 상세 성능")
                st.dataframe(optimizer.performance_df, use_container_width=True)
            st.divider()
            st.subheader(f"💡 AI 최적 레시피 (목표 두께: {user_input['Thickness (nm)']:.1f} nm)")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown("##### 레시피 파라미터")
                recipe_df = pd.DataFrame.from_dict({k: opt_recipe[k] for k in ['Cycles (n)', 'Temperature (c)', 'Pressure (torr)', 'Precursor Pulse Time (s)', 'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)']}, orient='index', columns=['Value'])
                st.dataframe(recipe_df, use_container_width=True)
            with c2:
                st.markdown("##### 예측 물성 결과")
                st.metric("최종 예측 두께 (nm)", f"{pred_results['Thickness (nm)']:.4f}", delta=f"{pred_results['Thickness (nm)'] - user_input['Thickness (nm)']:.4f} (nm)")
                st.dataframe(pred_results.to_frame(name='Predicted').drop('Thickness (nm)'), use_container_width=True)
            with c3:
                st.markdown("##### 물리/최적화 검증")
                st.dataframe(pd.DataFrame.from_dict(val_data, orient='index', columns=['Value']), use_container_width=True)
                st.caption(f"최종 비용: {opt_stats['Final Cost']}")

        with tab2:
            st.header("📊 공정 민감도 분석 (Sensitivity Analysis)")
            st.info("최적화된 레시피를 고정하고, 단일 변수 변화에 따른 물성 변화를 정밀하게 분석합니다.")
            
            # [수정] 그래프 옵션: 요청하신 Pulse Time vs SC 포함
            plot_options = {
                "GPC vs Temperature (온도)": ("Temperature (c)", "GPC (A/cycle)"),
                "GPC vs Pulse Time (펄스 시간)": ("Precursor Pulse Time (s)", "GPC (A/cycle)"),
                "GPC vs Pressure (압력)": ("Pressure (torr)", "GPC (A/cycle)"),
                "Step Coverage vs Pulse Time (피복성)": ("Precursor Pulse Time (s)", "Step Coverage (sc, %)")
            }
            
            selected_plot = st.selectbox("📈 표시할 그래프 선택:", list(plot_options.keys()))
            x_col_name, y_col_name = plot_options[selected_plot]
            
            with st.spinner(f"'{x_col_name}' 변화에 따른 시뮬레이션 수행 중..."):
                sensitivity_df = optimizer.analyze_process_sensitivity(opt_recipe, user_input, x_col_name, y_col_name)

            if not sensitivity_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                opt_x = opt_recipe.get(x_col_name)
                opt_y = pred_results.get(y_col_name) if y_col_name in pred_results else opt_recipe.get(y_col_name)
                
                sns.lineplot(data=sensitivity_df, x=x_col_name, y=y_col_name, marker='o', ax=ax, color='steelblue', label='Trend')
                
                if opt_x is not None:
                    ax.scatter([opt_x], [opt_y], color='red', s=100, zorder=5, label='Optimal Point')
                
                ax.set_xlabel(x_col_name)
                ax.set_ylabel(y_col_name)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # [수정] Y축 정밀 서식 및 자동 스케일링 (Zoom-in)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f')) # 소수점 5자리 표시
                
                # 데이터 범위가 너무 작으면(Noise 수준) 강제로 Zoom-in
                y_values = sensitivity_df[y_col_name]
                y_min, y_max = y_values.min(), y_values.max()
                
                if (y_max - y_min) < 1e-4: # 변화폭이 0.0001 미만일 때
                    mid = (y_max + y_min) / 2
                    ax.set_ylim(mid - 0.0001, mid + 0.0001) # 아주 미세한 변화도 보이게 설정
                
                ax.legend()
                
                plt.title(f"Process Sensitivity: {selected_plot}")
                st.pyplot(fig)
                
                # --- SC Trend (보조 그래프) ---
                if 'Step Coverage (sc, %)' in sensitivity_df.columns:
                    st.divider()
                    st.subheader(f"⚖️ Step Coverage 변화 ({x_col_name} 변화 시)")
                    fig2, ax_sc = plt.subplots(figsize=(10, 4))
                    ax_sc.plot(sensitivity_df[x_col_name], sensitivity_df['Step Coverage (sc, %)'], 'g-^', label='AI Prediction')
                    ax_sc.plot(sensitivity_df[x_col_name], sensitivity_df['Physics SC (%)'], 'k--x', label='Physics Model')
                    ax_sc.set_xlabel(x_col_name)
                    ax_sc.set_ylabel("Step Coverage (%)")
                    
                    # 동적 Y축 범위 설정 (Zoom-In)
                    sc_min = min(sensitivity_df['Step Coverage (sc, %)'].min(), sensitivity_df['Physics SC (%)'].min())
                    sc_max = max(sensitivity_df['Step Coverage (sc, %)'].max(), sensitivity_df['Physics SC (%)'].max())
                    margin = (sc_max - sc_min) * 0.1 if sc_max != sc_min else 1.0 
                    ax_sc.set_ylim(max(0, sc_min - margin), min(100.5, sc_max + margin))

                    ax_sc.legend()
                    ax_sc.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig2)
            else:
                st.warning("선택한 변수에 대한 시뮬레이션 데이터를 생성할 수 없습니다.")

        with tab3:
            st.header("🔍 AI 판단 근거 분석 (SHAP)")
            st.info("AI 모델(XGBoost)이 'Step Coverage'를 예측할 때 어떤 공정 변수가 가장 큰 영향을 미쳤는지 분석합니다.")
            if st.button("SHAP 분석 실행"):
                with st.spinner("SHAP 값을 계산 중입니다..."):
                    try:
                        explainer, shap_values, X_test_data = optimizer.get_shap_explainer()
                        feature_names = optimizer.ALL_INPUT_FEATURES_ORDERED
                        st.markdown("##### 1. 변수 중요도 (Feature Importance)")
                        fig_shap, ax_shap = plt.subplots()
                        shap.summary_plot(shap_values, X_test_data, feature_names=feature_names, show=False)
                        st.pyplot(fig_shap)
                    except Exception as e:
                        st.error(f"SHAP 분석 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main_gui()