# --- 0. 기본 라이브러리 및 Streamlit 임포트 ---
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

# --- 0. 물리/화학 상수 테이블 정의 ---
# (제공해주신 내용과 동일)
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

# --- 2. AI 모델 클래스 정의 ---
class ALDRegressor_Optimized(nn.Module):
    # (제공해주신 내용과 동일)
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
    
    # (제공해주신 클래스 내용 전체를 여기에 그대로 붙여넣습니다)
    # --- 1. 초기화 및 모델 학습 ---
    
    def __init__(self, file_path: str):
        print("--- 1단계: ALD 최적화 시스템 초기화 시작 ---")
        
        # 하이퍼파라미터 설정
        self.final_learning_rate = 0.00195
        self.final_dropout_rate = 0.28
        self.final_batch_size = 16
        self.final_epochs = 500
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 30
        self.WEIGHT_DECAY = 1e-5
        self.BEST_MODEL_PATH = 'best_ald_model.pth'
        
        self.DEFAULT_GPC_GUESS_A = 1.0 

        # 클래스 속성 초기화
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.final_model = None
        self.ALL_INPUT_FEATURES_ORDERED = []
        self.ALL_OUTPUT_FEATURES_ORDERED = []
        
        # 데이터 로드, 전처리, 모델 학습 자동 실행
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        self._train_model()

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        """데이터를 로드하고 전처리합니다."""
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            print(f"\n[치명적 오류] 파일 로드 실패: {e}. 프로그램을 종료합니다.")
            # Streamlit 환경에서는 sys.exit() 대신 st.error()와 st.stop()을 사용합니다.
            # (이 클래스는 Streamlit 외부에서 호출될 것이므로 일단 유지)
            raise(e) # 오류를 발생시켜 상위 호출자(Streamlit)가 처리하도록 함

        # (데이터 전처리 로직은 이전과 동일)
        df.replace('-', np.nan, inplace=True)
        cols_to_convert = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)',
            'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)',
            'Co-reactant Flow Rate (cm3/min)', 'Thickness (nm)', 'Surface Roughness (RMS, nm)',
            'Uniformity (%)', 'Step Coverage (sc, %)', 'Density (g/cm3)', 'GPC (A/cycle)',
            'Aspect Ratio (AR)', 'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)',
            'Breakdown Field (MV/cm)'
        ]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
        df['Co-reactant'] = df['Co-reactant'].replace({'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'})
        cols_to_drop_high_nan = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)']
        df_processed = df.drop(columns=cols_to_drop_high_nan)
        categorical_cols = ['Precursor', 'Co-reactant', 'Purge Gas']
        df_encoded = pd.get_dummies(df_processed.drop(columns=['순서']), columns=categorical_cols, dummy_na=False)
        
        return df_encoded

    def _prepare_datasets(self, df_encoded: pd.DataFrame):
        """AI 모델의 입/출력 정의 및 데이터셋을 준비합니다."""
        
        target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)',
            'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)',
            'Breakdown Field (MV/cm)',
            'Step Coverage (sc, %)'
        ]
        cols_to_ignore_for_ai = ['Aspect Ratio (AR)']

        try:
            self.ALL_INPUT_FEATURES_ORDERED = df_encoded.drop(
                columns=target_cols + cols_to_ignore_for_ai
            ).columns.tolist()
            self.ALL_OUTPUT_FEATURES_ORDERED = target_cols
        except KeyError:
            print("\n[치명적 오류] target_cols 또는 cols_to_ignore_for_ai에 CSV에 없는 컬럼명이 포함되어 있습니다.")
            raise # 오류 발생

        X = df_encoded[self.ALL_INPUT_FEATURES_ORDERED].values
        Y = df_encoded[self.ALL_OUTPUT_FEATURES_ORDERED].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_imputer = KNNImputer(n_neighbors=5); X_train = X_imputer.fit_transform(X_train); X_test = X_imputer.transform(X_test)
        Y_imputer = KNNImputer(n_neighbors=5); Y_train = Y_imputer.fit_transform(Y_train); Y_test = Y_imputer.transform(Y_test)
        self.X_train_scaled = self.X_scaler.fit_transform(X_train)
        self.X_test_scaled = self.X_scaler.transform(X_test)
        self.Y_train_scaled = self.Y_scaler.fit_transform(Y_train)
        self.Y_test_scaled = self.Y_scaler.transform(Y_test)
        self.INPUT_SIZE = self.X_train_scaled.shape[1]
        self.OUTPUT_SIZE = self.Y_train_scaled.shape[1]
        print(f"AI 모델 입력 피처 수: {self.INPUT_SIZE} (AR 제외)")
        print(f"AI 모델 출력 피처 수: {self.OUTPUT_SIZE} (SC 포함)")

    def _train_model(self):
        """AI 모델을 학습시키고 self.final_model에 저장합니다."""
        print(f"\n--- 2단계: AI 모델 학습 시작 (최대 {self.final_epochs} 에포크, 조기 종료 적용) ---")
        print(f"AI가 예측할 물성 (총 {self.OUTPUT_SIZE}개): {self.ALL_OUTPUT_FEATURES_ORDERED}")

        X_train_tensor = torch.from_numpy(self.X_train_scaled).float()
        Y_train_tensor = torch.from_numpy(self.Y_train_scaled).float()
        X_test_tensor = torch.from_numpy(self.X_test_scaled).float()
        Y_test_tensor = torch.from_numpy(self.Y_test_scaled).float()

        X_train_final, X_val, Y_train_final, Y_val = train_test_split(
            X_train_tensor, Y_train_tensor, test_size=self.VALIDATION_SPLIT, random_state=42
        )
        train_dataset = TensorDataset(X_train_final, Y_train_final)
        val_dataset = TensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.final_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.final_batch_size, shuffle=False)
        
        model = ALDRegressor_Optimized(self.INPUT_SIZE, self.OUTPUT_SIZE, self.final_dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.final_learning_rate, weight_decay=self.WEIGHT_DECAY)

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
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.final_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss; torch.save(model.state_dict(), self.BEST_MODEL_PATH); patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.PATIENCE:
                print(f"\n[조기 종료] {self.PATIENCE} 에포크 동안 검증 손실이 개선되지 않아 Epoch {epoch+1}에서 학습을 중단합니다.")
                break

        print(f"\n✅ 모델 학습 완료. 최고 성능 모델(Val Loss: {best_val_loss:.4f})을 불러옵니다.")
        model.load_state_dict(torch.load(self.BEST_MODEL_PATH)); model.eval()

        with torch.no_grad():
            Y_test_pred = model(X_test_tensor); test_loss = criterion(Y_test_pred, Y_test_tensor)
        print(f"--- 🚀 최종 모델 테스트셋 MSE ({self.OUTPUT_SIZE}개 물성): {test_loss.item():.6f} ---")
        
        self.final_model = model

        if os.path.exists(self.BEST_MODEL_PATH):
            os.remove(self.BEST_MODEL_PATH); print(f"\n[정리] 임시 모델 파일 ({self.BEST_MODEL_PATH})이 삭제되었습니다.")

    # --- 3. 물리 모델 함수 정의 (SC 전담) ---
    
    @staticmethod
    def _calculate_physical_parameters(T_celsius, P_torr, precursor_name, L_feature_m):
        # (이전과 동일)
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        d_precursor_m = const["diameter_m"]; T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
        Kn = lambda_m / L_feature_m
        return lambda_m, Kn

    @staticmethod
    def _calculate_physics_sc(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
        # (이전과 동일)
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        c = const["sticking_c"]; q = const["max_sites_q"]; d_precursor_m = const["diameter_m"]; M_A_kg = const["mass_g_mol"] / 1000.0 / N_A
        T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322; L_m = AR_value * CD_m
        v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_A_kg)); lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
        D_A = (1/3) * lambda_m * v_avg; D_Kn = (1/3) * v_avg * CD_m; D_eff = 1 / ((1 / D_A) + (1 / D_Kn))
        Q = 1 / np.sqrt(2 * np.pi * M_A_kg * k_B * T_K); lambda_D = np.sqrt(D_eff * Pulse_Time_s); L_over_lambda_D = L_m / (lambda_D + 1e-12)
        constant_term = (c * Q * P_Pa * Pulse_Time_s) / q; theta_0 = 1.0 - np.exp(-constant_term)
        exp_inner_term = -constant_term * np.exp(-L_over_lambda_D); theta_L = 1.0 - np.exp(exp_inner_term)
        SC_full_model = theta_L / (theta_0 + 1e-12)
        return np.clip(SC_full_model * 100.0, 0.0, 100.0)

    # --- 4. AI 모델 입/출력 변환기 ---
    
    def _create_model_input(
        self,
        recipe_params: Dict[str, Any],
        precursor_name: str,
        co_reactant_name: str,
        purge_gas_name: str
    ) -> pd.DataFrame:
        """레시피 딕셔너리를 AI 모델 입력용 DataFrame으로 변환합니다."""
        # (이전과 동일)
        input_df = pd.DataFrame(columns=self.ALL_INPUT_FEATURES_ORDERED); input_df.loc[0] = 0.0
        for key, value in recipe_params.items():
            if key in input_df.columns: input_df.at[0, key] = value
        precursor_col = f"Precursor_{precursor_name}";
        if precursor_col in input_df.columns: input_df.at[0, precursor_col] = 1.0
        coreactant_col = f"Co-reactant_{co_reactant_name}";
        if coreactant_col in input_df.columns: input_df.at[0, coreactant_col] = 1.0
        purge_gas_col = f"Purge Gas_{purge_gas_name}";
        if purge_gas_col in input_df.columns: input_df.at[0, purge_gas_col] = 1.0
        return input_df

    def _predict_from_recipe(
        self,
        recipe_params: Dict[str, Any],
        precursor_name: str,
        co_reactant_name: str,
        purge_gas_name: str
    ) -> pd.Series:
        """하나의 레시피로 AI 예측을 수행하고 9개 결과를 반환합니다."""
        # (이전과 동일)
        input_df = self._create_model_input(recipe_params, precursor_name, co_reactant_name, purge_gas_name)
        X_scaled = self.X_scaler.transform(input_df.values); X_tensor = torch.from_numpy(X_scaled).float()
        self.final_model.eval()
        with torch.no_grad():
            Y_pred_scaled_tensor = self.final_model(X_tensor)
        Y_pred_unscaled = self.Y_scaler.inverse_transform(Y_pred_scaled_tensor.numpy())[0]
        predicted_results = pd.Series(Y_pred_unscaled, index=self.ALL_OUTPUT_FEATURES_ORDERED).round(4)
        return predicted_results


    # --- 5. 최적화 목적/제약 함수 (수정됨) ---

    def _constraint_sc(
        self,
        x: np.ndarray, 
        user_input: Dict[str, Any],
        co_reactant_name: str,
        purge_gas_name: str,
        cost_weights: Dict[str, float],
        fixed_cycles_n: int 
    ) -> float:
        """물리 모델 기반 SC 제약조건 함수 (SLSQP용)"""
        
        target_ar = user_input["Target AR"]
        if target_ar <= 5: TARGET_SC_MIN = 98.0
        elif target_ar <= 15: TARGET_SC_MIN = 90.0
        else: TARGET_SC_MIN = 85.0
        
        T_celsius = x[0]
        P_torr = x[1]
        Pulse_Time_s = x[2]
        
        phys_sc = self._calculate_physics_sc(
            P_torr=P_torr, T_celsius=T_celsius, Pulse_Time_s=Pulse_Time_s,
            AR_value=user_input["Target AR"],
            precursor_name=user_input["Precursor"],
            CD_m=user_input["CD (nm)"] * 1e-9
        )
        return phys_sc - TARGET_SC_MIN

    def _objective_function(
        self,
        x: np.ndarray, 
        user_input: Dict[str, Any],
        co_reactant_name: str,
        purge_gas_name: str,
        cost_weights: Dict[str, float],
        fixed_cycles_n: int 
    ) -> float:
        """AI 예측 기반 비용 함수 (SLSQP용) - GPC와 거칠기 최소화가 목표"""
        
        # 1. 최적화 변수(x, 5개)를 레시피(dict)로 매핑
        recipe_params = {
            "Temperature (c)": x[0],
            "Pressure (torr)": x[1],
            "Precursor_Pulse Time (s)": x[2],
            "Purge Time (s)": x[3],
            "Purge Gas Flow Rate (cm3/min)": x[4],
            "Cycles (n)": fixed_cycles_n, # 💡 고정된 사이클 값 사용
            "Co-reactant_Pulse Time (s)": x[2]
        }
        
        # 2. 레시피로 AI 예측 수행
        try:
            predicted_results = self._predict_from_recipe(
                recipe_params, user_input["Precursor"], co_reactant_name, purge_gas_name
            )
        except Exception as e:
            return 1e9 
            
        # 3. 목표값 정의 (GPC)
        target_thickness = user_input["Thickness (nm)"]
        target_gpc_ideal = (target_thickness * 10) / (fixed_cycles_n + 1e-6)
        
        # 4. 오차 (Cost) 계산
        w_gpc = cost_weights.get("gpc", 1.0)
        w_roughness = cost_weights.get("roughness", 1.0)

        pred_gpc = predicted_results.get('GPC (A/cycle)', 0)
        pred_roughness = predicted_results.get('Surface Roughness (RMS, nm)', 10)
        
        cost_gpc = (pred_gpc - target_gpc_ideal)**2
        cost_roughness = (pred_roughness / 5.0)**2

        total_cost = (
            w_gpc * cost_gpc +
            w_roughness * cost_roughness
        )
        
        return total_cost


    # --- 6. 최적화 실행 및 리포트 (수정됨) ---

    def generate_optimal_recipe(self, user_input: Dict[str, Any]):
        """
        [수정된 2단계 최적화]
        1. SLSQP로 5개 변수를 최적화하여 '최적 GPC'를 찾습니다.
        2. '최적 GPC'를 기반으로 '최종 Cycles'를 역산합니다.
        """
        precursor = user_input["Precursor"]
        target_thickness = user_input["Thickness (nm)"]
        target_ar = user_input["Target AR"]
        CD_m = user_input["CD (nm)"] * 1e-9
        
        # 💡 Streamlit에서는 print 대신 UI로 피드백을 줍니다.
        # print("\n--- ⏳ 최적의 ALD 공정 조건을 탐색 중입니다. (SLSQP, 5개 변수, GPC/Roughness 타겟) ---")
        
        co_reactant = 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3'
        purge_gas = "N2"
        
        initial_cycles_n = int(round((target_thickness * 10) / self.DEFAULT_GPC_GUESS_A))
        initial_cycles_n = max(10, initial_cycles_n) # 최소 10 사이클 보장
        
        bounds = [
            (150, 400),     # Temperature (c)
            (0.01, 1.0),    # Pressure (torr)
            (0.05, 2.0),    # Precursor_Pulse Time (s)
            (1.0, 10.0),    # Purge Time (s)
            (50, 500),      # Purge Gas Flow Rate (cm3/min)
        ]
        initial_guess = [
            300, 0.5, 0.1, 5.0, 300
        ]
        
        args = (user_input, co_reactant, purge_gas, COST_WEIGHTS, initial_cycles_n)
        
        constraints = ({
            'type': 'ineq', 
            'fun': self._constraint_sc, 
            'args': args
        })
        
        result = minimize(
            self._objective_function, 
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'eps': 1e-6}
        )
        
        if not result.success:
            print(f"\n[경고] 최적화가 수렴에 실패했습니다 (혹은 제약조건 위반): {result.message}")
            
        optimal_x_5_vars = result.x
        
        # --- 최종 검증 및 Cycles (n) 역산 ---
        
        recipe_params_for_gpc_check = {
            "Temperature (c)": optimal_x_5_vars[0],
            "Pressure (torr)": optimal_x_5_vars[1],
            "Precursor_Pulse Time (s)": optimal_x_5_vars[2],
            "Purge Time (s)": optimal_x_5_vars[3],
            "Purge Gas Flow Rate (cm3/min)": optimal_x_5_vars[4],
            "Cycles (n)": initial_cycles_n, 
            "Co-reactant_Pulse Time (s)": optimal_x_5_vars[2]
        }
        
        predicted_results_gpc_check = self._predict_from_recipe(
            recipe_params_for_gpc_check, precursor, co_reactant, purge_gas
        )
        
        final_gpc_A = predicted_results_gpc_check.get('GPC (A/cycle)', self.DEFAULT_GPC_GUESS_A)
        if final_gpc_A <= 0: final_gpc_A = self.DEFAULT_GPC_GUESS_A
            
        final_optimal_cycles_n = int(round((target_thickness * 10) / final_gpc_A))
        final_optimal_cycles_n = max(10, final_optimal_cycles_n)

        # (이 print문들은 터미널에 출력됩니다)
        print(f"\n--- 💡 GPC 기반 Cycles 재계산 ---")
        print(f"  - AI 예측 최적 GPC: {final_gpc_A:.4f} A/cycle")
        print(f"  - 목표 두께 {target_thickness} nm 달성을 위한 최종 Cycles: {final_optimal_cycles_n} (n)")
        
        # --- 리포트 생성 ---
        
        T = optimal_x_5_vars[0]; P = optimal_x_5_vars[1]; Pulse_Time = optimal_x_5_vars[2]
        SC_full_model_value = self._calculate_physics_sc(P, T, Pulse_Time, target_ar, precursor, CD_m)
        lambda_m, Kn = self._calculate_physical_parameters(T, P, precursor, CD_m)

        optimal_recipe_report = {
            "Precursor": precursor, "Co-reactant": co_reactant,
            "Temperature (c)": round(T, 2), "Pressure (torr)": round(P, 3),
            "Cycles (n)": final_optimal_cycles_n, 
            "Precursor Pulse Time (s)": round(Pulse_Time, 3),
            "Co-reactant Pulse Time (s)": round(Pulse_Time, 3),
            "Purge Time (s)": round(optimal_x_5_vars[3], 2),
            "Purge Gas Flow Rate (cm3/min)": round(optimal_x_5_vars[4], 0),
            "Purge Gas": purge_gas
        }
        
        final_recipe_params_for_report = {
            "Temperature (c)": T,
            "Pressure (torr)": P,
            "Precursor_Pulse Time (s)": Pulse_Time,
            "Purge Time (s)": optimal_x_5_vars[3],
            "Purge Gas Flow Rate (cm3/min)": optimal_x_5_vars[4],
            "Cycles (n)": final_optimal_cycles_n, 
            "Co-reactant_Pulse Time (s)": Pulse_Time
        }
        predicted_results_for_report = self._predict_from_recipe(
            final_recipe_params_for_report, precursor, co_reactant, purge_gas
        )
        
        validation_data = {
            "Mean Free Path (λ) [m]": f"{lambda_m:.2e}", "Knudsen Number (Kn)": f"{Kn:.2f}",
            "Sticking Coeff. (c) 사용 ": f"{PRECURSOR_CONSTANTS.get(precursor, PRECURSOR_CONSTANTS['TMA'])['sticking_c']:.3e}",
            "SC (Full Model)": f"{SC_full_model_value:.4f} %",
        }
        
        optimization_stats = {
            "Optimization Success": result.success,
            "Optimization Message": result.message,
            "Function Evaluations (nfev)": result.nfev,
            "Iterations (nit)": result.nit,
            "Final Cost (fun)": f"{result.fun:.6f}"
        }
        
        # 💡 [핵심 수정] _print_report 대신 4개의 결과 객체를 반환합니다.
        return optimal_recipe_report, predicted_results_for_report, validation_data, optimization_stats

    # 💡 _print_report 함수는 Streamlit UI에서 직접 처리하므로 여기서는 제거하거나 주석 처리합니다.
    # def _print_report(self, ...):
    #     ... 


# --- 7. Streamlit UI 및 실행 로직 ---

@st.cache_resource(show_spinner="AI 모델 및 데이터 로딩 중... (최초 1회 시간이 걸릴 수 있습니다)")
def initialize_optimizer(file_path):
    """
    ALDOptimizer 객체를 초기화하고 Streamlit 캐시에 저장합니다.
    이 함수는 모델 학습을 포함하며, 앱 실행 시 한 번만 실행됩니다.
    """
    try:
        optimizer = ALDOptimizer(file_path=file_path)
        return optimizer
    except FileNotFoundError:
        st.error(f"[치명적 오류] 데이터 파일 '{file_path}'를 찾을 수 없습니다.")
        st.stop()
    except Exception as e:
        st.error(f"[초기화 오류] 모델 로딩 중 문제가 발생했습니다: {e}")
        st.exception(e) # 자세한 오류 로그 표시
        st.stop()


def main_app():
    """Streamlit 메인 애플리케이션 함수"""
    
    st.set_page_config(page_title="AI 기반 ALD 공정 최적화", layout="wide")
    st.title("✨ AI 기반 ALD 공정 최적화 시스템")

    # --- 1. Optimizer 객체 로드 (캐시 사용) ---
    # 이 부분에서 __init__이 실행되며 모델 학습이 진행됩니다. (최초 1회)
    optimizer = initialize_optimizer(file_path="AI_ALD1.csv.csv")
    
    if optimizer is None:
        st.warning("모델이 정상적으로 로드되지 않았습니다. 파일 경로를 확인하세요.")
        st.stop()

    # --- 2. 사용자 목표 입력 (Streamlit 사이드바) ---
    st.sidebar.header("🎯 3단계: 목표 조건 입력")

    available_precursors = {1: "TMA", 2: "TDMAH", 3: "TEMAHf", 4: "Zr(NEt2)4"}
    precursor_options = list(available_precursors.values())
    
    selected_precursor_name = st.sidebar.selectbox(
        "1. 사용할 전구체를 선택해 주세요:",
        precursor_options,
        index=0
    )

    thickness = st.sidebar.number_input(
        "2. 목표 박막 두께 (Thickness, nm):",
        min_value=1.0, max_value=200.0, value=15.0, step=0.5,
        help="예: 15.0"
    )

    target_ar = st.sidebar.number_input(
        "3. 목표 종횡비 (Aspect Ratio, AR):",
        min_value=1.0, max_value=100.0, value=10.0, step=0.1,
        help="예: 10.0"
    )

    critical_dimension_nm = st.sidebar.number_input(
        "4. 채널 폭 (Critical Dimension, CD, nm):",
        min_value=1.0, max_value=500.0, value=100.0, step=1.0,
        help="예: 100"
    )

    # 입력값을 딕셔너리로 묶기
    user_target_input = {
        "Precursor": selected_precursor_name,
        "Thickness (nm)": thickness,
        "Target AR": target_ar,
        "CD (nm)": critical_dimension_nm
    }

    st.markdown("---")
    st.header(f"[입력된 목표: {selected_precursor_name}, {thickness} nm]")
    st.subheader(f"[구조적 조건: AR={target_ar}, CD={critical_dimension_nm} nm]")

    # --- 3. 최적화 실행 버튼 ---
    if st.button("🚀 최적 레시피 생성하기", type="primary"):
        
        # 스피너(로딩 표시)와 함께 최적화 함수 실행
        with st.spinner("--- ⏳ 최적의 ALD 공정 조건을 탐색 중입니다. (SLSQP, 5개 변수, GPC/Roughness 타겟) ---"):
            try:
                # 3. 최적화 실행 (결과 4개를 반환받음)
                optimal_recipe, predicted_results, validation_data, optimization_stats = optimizer.generate_optimal_recipe(user_target_input)

                st.success("✅ 최적화 완료! 결과 보고서를 확인하세요.")
                
                st.markdown("---")
                st.header("📄 AI 기반 ALD 공정 최적화 최종 결과 보고서")

                col1, col2 = st.columns(2)

                # --- 결과 리포트 (기존 _print_report 대체) ---
                with col1:
                    st.subheader("💡 AI 제안 최적 공정 레시피")
                    recipe_df = pd.DataFrame.from_dict(optimal_recipe, orient='index', columns=['Value'])
                    st.dataframe(recipe_df)

                    st.subheader("🔬 물리 기반 검증 (SC 모델)")
                    validation_df = pd.DataFrame.from_dict(validation_data, orient='index', columns=['Value'])
                    st.dataframe(validation_df)

                with col2:
                    st.subheader("📈 AI 예측 박막 특성 (9가지)")
                    st.dataframe(predicted_results.to_frame(name='Predicted Value'))
                    
                    st.subheader("📊 최적화(SLSQP) 수렴 리포트")
                    stats_df = pd.DataFrame.from_dict(optimization_stats, orient='index', columns=['Value'])
                    st.dataframe(stats_df)
                
                st.markdown("---")
                st.subheader("🔍 핵심 결과 요약")
                
                # 두께 검증
                pred_thickness = predicted_results.get('Thickness (nm)', 0)
                st.metric(
                    label=f"두께 검증 (목표: {user_target_input['Thickness (nm)']} nm)",
                    value=f"{pred_thickness:.4f} nm",
                    delta=f"{pred_thickness - user_target_input['Thickness (nm)']:.4f} nm",
                    # delta_color="inverse" # 오차가 0에 가까울수록 좋으므로 "inverse"
                )

                # SC 이중 검증
                st.markdown("**[SC 이중 검증 요약]**")
                sc_ai = predicted_results.get('Step Coverage (sc, %)', 'N/A')
                sc_phys = validation_data['SC (Full Model)']
                st.text(f"  - 1. AI 예측 SC: {sc_ai:.4f} % (데이터 기반 학습 결과)")
                st.text(f"  - 2. 물리 모델 SC: {sc_phys} (최적화 제약조건)")
                
                st.info(f"최적화 목표 오차 (Cost): {optimization_stats['Final Cost (fun)']} (GPC(x10k), Roughness(x10) 기준)")

            except Exception as e:
                st.error(f"최적화 실행 중 오류가 발생했습니다: {e}")
                st.exception(e) # 자세한 오류 로그 표시

    else:
        st.info("왼쪽 사이드바에서 목표 조건을 입력하고 '최적 레시피 생성하기' 버튼을 눌러주세요.")


# --- 8. 시스템 실행 ---
if __name__ == "__main__":
    # (기존 CLI 실행 코드는 main_app() 호출로 대체됨)
    main_app()