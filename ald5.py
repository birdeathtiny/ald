# --- 0. 기본 라이브러리 및 Streamlit 임포트 ---
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

# --- 0. 물리/화학 상수 테이블 정의 ---
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
        if self.mode == "cli":
            print("--- 1단계: ALD 최적화 시스템 초기화 시작 ---")
        
        self.final_learning_rate = 0.00195
        self.final_dropout_rate = 0.28
        self.final_batch_size = 16
        self.final_epochs = 500
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 30
        self.WEIGHT_DECAY = 1e-5
        self.BEST_MODEL_PATH = 'best_ald_model.pth'
        self.DEFAULT_GPC_GUESS_A = 1.0 

        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.final_model = None
        self.ALL_INPUT_FEATURES_ORDERED = []
        self.ALL_OUTPUT_FEATURES_ORDERED = []
        
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        self._train_model()

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            try:
                df = pd.read_csv(file_path + ".csv", encoding='CP949')
            except:
                msg = f"\n[치명적 오류] 파일 로드 실패: {e}"
                if self.mode == "cli": print(msg); sys.exit(1)
                else: raise e

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
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Co-reactant' in df.columns:
            df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
            df['Co-reactant'] = df['Co-reactant'].replace({'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'})
        
        cols_to_drop_high_nan = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)']
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
            'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)',
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
        X_imputer = KNNImputer(n_neighbors=5); X_train = X_imputer.fit_transform(X_train); X_test = X_imputer.transform(X_test)
        Y_imputer = KNNImputer(n_neighbors=5); Y_train = Y_imputer.fit_transform(Y_train); Y_test = Y_imputer.transform(Y_test)
        self.X_train_scaled = self.X_scaler.fit_transform(X_train)
        self.X_test_scaled = self.X_scaler.transform(X_test)
        self.Y_train_scaled = self.Y_scaler.fit_transform(Y_train)
        self.Y_test_scaled = self.Y_scaler.transform(Y_test)
        self.INPUT_SIZE = self.X_train_scaled.shape[1]
        self.OUTPUT_SIZE = self.Y_train_scaled.shape[1]
        
        if self.mode == "cli":
            print(f"AI 모델 입력 피처 수: {self.INPUT_SIZE}")
            print(f"AI 모델 출력 피처 수: {self.OUTPUT_SIZE}")

    def _train_model(self):
        if self.mode == "cli":
            print(f"\n--- 2단계: AI 모델 학습 시작 (최대 {self.final_epochs} 에포크) ---")
        
        X_train_tensor = torch.from_numpy(self.X_train_scaled).float()
        Y_train_tensor = torch.from_numpy(self.Y_train_scaled).float()
        X_test_tensor = torch.from_numpy(self.X_test_scaled).float()
        Y_test_tensor = torch.from_numpy(self.Y_test_scaled).float()

        X_train_final, X_val, Y_train_final, Y_val = train_test_split(X_train_tensor, Y_train_tensor, test_size=self.VALIDATION_SPLIT, random_state=42)
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
            
            if self.mode == "cli" and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.final_epochs}], Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss; torch.save(model.state_dict(), self.BEST_MODEL_PATH); patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.PATIENCE:
                break
        
        model.load_state_dict(torch.load(self.BEST_MODEL_PATH)); model.eval()
        self.final_model = model
        if os.path.exists(self.BEST_MODEL_PATH): os.remove(self.BEST_MODEL_PATH)
        
        if self.mode == "cli":
             print(f"\n✅ 모델 학습 완료. Val Loss: {best_val_loss:.4f}")

    @staticmethod
    def _calculate_physical_parameters(T_celsius, P_torr, precursor_name, L_feature_m):
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        d_precursor_m = const["diameter_m"]; T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
        Kn = lambda_m / L_feature_m
        return lambda_m, Kn

    @staticmethod
    def _calculate_physics_sc_details(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
        import numpy as np
        const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
        d = const["diameter_m"]; M_kg = const["mass_g_mol"] / 1000.0 / N_A
        T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322
        L_m = AR_value * CD_m
        v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_kg)); lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
        D_A = (1.0 / 3.0) * lambda_m * v_avg; D_Kn = (1.0 / 3.0) * v_avg * CD_m
        D_eff = 1.0 / ((1.0 / (D_A + 1e-30)) + (1.0 / (D_Kn + 1e-30)))
        lambda_pen = np.sqrt(D_eff * Pulse_Time_s + 1e-30)
        phi = L_m / (lambda_pen + 1e-30)
        if phi < 1.0:
            SC_fraction = 1.0 / (1.0 + phi); mode = "Reaction Limited (RDS, φ < 1)"
        else:
            SC_fraction = np.exp(-phi); mode = "Diffusion Limited (RDS, φ ≥ 1)"
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
        return pd.Series(Y_pred_unscaled, index=self.ALL_OUTPUT_FEATURES_ORDERED).round(4)

    def _constraint_sc(self, x, user_input, co_reactant_name, purge_gas_name, cost_weights, fixed_cycles_n) -> float:
        target_ar = user_input["Target AR"]
        TARGET_SC_MIN = 98.0 if target_ar <= 5 else 90.0 if target_ar <= 15 else 85.0
        phys_sc = self._calculate_physics_sc(x[1], x[0], x[2], user_input["Target AR"], user_input["Precursor"], user_input["CD (nm)"] * 1e-9)
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
        cost = (cost_weights["gpc"] * (pred.get('GPC (A/cycle)', 0) - target_gpc)**2) + (cost_weights["roughness"] * (pred.get('Surface Roughness (RMS, nm)', 10)/5.0)**2)
        return cost

    def generate_optimal_recipe(self, user_input: Dict[str, Any], silent: bool = False):
        precursor = user_input["Precursor"]; thickness = user_input["Thickness (nm)"]
        co_reactant = 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3'; purge_gas = "N2"
        initial_cycles = max(10, int(round((thickness * 10) / self.DEFAULT_GPC_GUESS_A)))
        
        bounds = [(150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500)]
        initial_guess = [np.random.uniform(l, h) for l, h in bounds]

        if self.mode == "cli" and not silent:
            print(f"\n--- 🔍 '무작위' 탐색 시작점 설정: {np.round(initial_guess, 3)} ---")

        args = (user_input, co_reactant, purge_gas, COST_WEIGHTS, initial_cycles)
        result = minimize(self._objective_function, initial_guess, args=args, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': self._constraint_sc, 'args': args}, options={'maxiter': 100, 'eps': 1e-6})
        
        x = result.x
        check_params = {"Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                        "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4], "Cycles (n)": initial_cycles, "Co-reactant_Pulse Time (s)": x[2]}
        check_res = self._predict_from_recipe(check_params, precursor, co_reactant, purge_gas)
        final_gpc = check_res.get('GPC (A/cycle)', 1.0); final_gpc = max(0.001, final_gpc)
        final_cycles = max(10, int(round((thickness * 10) / final_gpc)))

        if self.mode == "cli" and not silent:
            print(f"--- 🏁 탐색 완료! GPC: {final_gpc:.4f}, 최종 Cycles: {final_cycles} ---")

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

    def simulate_target_sweep(self, base_user_input, target_param_name, range_values):
        results = []
        for val in range_values:
            current_input = base_user_input.copy()
            current_input[target_param_name] = val
            
            # 조용히 최적화 실행
            opt_recipe, pred_results, _, _ = self.generate_optimal_recipe(current_input, silent=True)
            
            phys_sc = self._calculate_physics_sc(
                opt_recipe['Pressure (torr)'], opt_recipe['Temperature (c)'], opt_recipe['Precursor Pulse Time (s)'],
                current_input['Target AR'], current_input['Precursor'], current_input['CD (nm)'] * 1e-9
            )

            row = {target_param_name: val}
            row.update(opt_recipe)
            row.update(pred_results.to_dict())
            row['Physics SC (%)'] = phys_sc
            results.append(row)
        return pd.DataFrame(results)


# ==========================================
# 🖥️ 1. CLI 모드 실행 함수 (터미널)
# ==========================================
def main_cli():
    print("\n" + "="*50 + "\n  [CLI 모드] AI 기반 ALD 공정 최적화 (터미널)\n" + "="*50)
    
    import matplotlib
    try: matplotlib.use('TkAgg')
    except: pass 

    csv_file = "AI_ALD1.csv" 
    if not os.path.exists(csv_file):
        print(f"[오류] '{csv_file}' 파일이 없습니다."); return
    
    optimizer = ALDOptimizer(file_path=csv_file, mode="cli")

    precursors = {1: "TMA", 2: "TDMAH", 3: "TEMAHf", 4: "Zr(NEt2)4"}
    print("\n[전구체 선택]"); [print(f"{k}: {v}") for k, v in precursors.items()]
    try:
        sel_p = precursors.get(int(input("1. 전구체 번호 입력: ")), "TMA")
        th = float(input("2. 목표 두께 (Thickness, nm): "))
        ar = float(input("3. 목표 AR (Aspect Ratio): "))
        cd = float(input("4. CD (nm) (예: 100): "))
    except: print("[입력 오류]"); return

    user_input = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
    recipe, pred, valid, stats = optimizer.generate_optimal_recipe(user_input)

    print("\n[💡 AI 최적 레시피]\n", pd.Series(recipe).to_string())
    print("\n[📈 예측 물성]\n", pred.to_string())
    print("\n[🔬 물리 검증]\n", pd.Series(valid).to_string())
    print(f"\n✅ 최종 두께: {pred['Thickness (nm)']:.4f} nm")

    # --- CLI 시각화 ---
    print("\n" + "="*50 + "\n📊 그래프 시각화 설정 (윈도우 창으로 표시됩니다)\n" + "="*50)
    x_options = ["Thickness (nm)", "Target AR"]
    print("[1] X축 (변화시킬 목표값) 선택:")
    for i, opt in enumerate(x_options, 1): print(f"  {i}. {opt}")
    try: x_idx = int(input("  => 번호 입력 (기본 1): ")) - 1
    except: x_idx = 0
    target_param = x_options[x_idx] if 0 <= x_idx < len(x_options) else x_options[0]

    y_left_opts = ["Temperature (c)", "Pressure (torr)", "Precursor Pulse Time (s)", "Purge Time (s)", "Cycles (n)"]
    print("\n[2] 왼쪽 Y축 (최적 공정 조건) 선택:")
    for i, opt in enumerate(y_left_opts, 1): print(f"  {i}. {opt}")
    try: yl_idx = int(input("  => 번호 입력 (기본 1): ")) - 1
    except: yl_idx = 0
    y_left = y_left_opts[yl_idx] if 0 <= yl_idx < len(y_left_opts) else y_left_opts[0]

    y_right_opts = ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)", "Uniformity (%)"]
    print("\n[3] 오른쪽 Y축 (예측 물성) 선택:")
    for i, opt in enumerate(y_right_opts, 1): print(f"  {i}. {opt}")
    try: yr_idx = int(input("  => 번호 입력 (기본 1): ")) - 1
    except: yr_idx = 0
    y_right = y_right_opts[yr_idx] if 0 <= yr_idx < len(y_right_opts) else y_right_opts[0]

    print(f"\n📈 '{target_param}' 변화에 따른 시뮬레이션 진행 중...")
    
    cur_val = user_input[target_param]
    sweep_range = np.linspace(cur_val * 0.5, cur_val * 1.5, 10)
    sweep_df = optimizer.simulate_target_sweep(user_input, target_param, sweep_range)

    plt.figure(figsize=(14, 5))

    # Graph 1
    ax1 = plt.subplot(1, 2, 1)
    line1 = ax1.plot(sweep_df[target_param], sweep_df[y_left], 'r-o', label=f"Recipe: {y_left}")
    ax1.set_xlabel(f"Target {target_param}"); ax1.set_ylabel(f"Optimal {y_left}", color='r')
    ax1.tick_params(axis='y', labelcolor='r'); ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()
    line2 = ax2.plot(sweep_df[target_param], sweep_df[y_right], 'b--s', label=f"Property: {y_right}")
    ax2.set_ylabel(f"Predicted {y_right}", color='b'); ax2.tick_params(axis='y', labelcolor='b')
    lines = line1 + line2; labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax1.set_title(f"Trend: {y_left} & {y_right}")

    # Graph 2
    plt.subplot(1, 2, 2)
    plt.plot(sweep_df[target_param], sweep_df['Step Coverage (sc, %)'], 'g-^', label='AI Prediction')
    plt.plot(sweep_df[target_param], sweep_df['Physics SC (%)'], 'k--x', label='Physics Model')
    plt.xlabel(f"Target {target_param}"); plt.ylabel("Step Coverage (%)"); plt.ylim(0, 110)
    plt.title(f"SC Trend: AI vs Physics")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    try: plt.show(); print("   (그래프 창을 닫으면 종료됩니다)")
    except Exception as e: print(f"\n⚠️ 그래프 팝업 불가: {e}"); plt.savefig('result_graph.png')


# ==========================================
# 🌐 2. GUI 모드 실행 함수 (Streamlit)
# ==========================================
def main_gui():
    st.set_page_config(page_title="AI 기반 ALD 공정 최적화", layout="wide")
    st.title("✨ AI 기반 ALD 공정 최적화 시스템")

    @st.cache_resource(show_spinner="AI 모델 로딩 중...")
    def load_optimizer(): return ALDOptimizer(file_path="AI_ALD1.csv", mode="gui") 

    try: optimizer = load_optimizer()
    except Exception as e: st.error(f"모델 로드 실패: {e}"); st.stop()

    st.sidebar.header("🎯 목표 조건 입력")
    sel_p = st.sidebar.selectbox("전구체 선택", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("목표 두께 (nm)", 1.0, 200.0, 15.0)
    ar = st.sidebar.number_input("목표 AR", 1.0, 100.0, 10.0)
    cd = st.sidebar.number_input("CD (nm)", 1.0, 500.0, 100.0)

    # 💡 [핵심 수정] Session State 초기화 (재실행 시 데이터 유지)
    if 'opt_done' not in st.session_state:
        st.session_state['opt_done'] = False
        st.session_state['opt_recipe'] = None
        st.session_state['pred_results'] = None
        st.session_state['val_data'] = None
        st.session_state['opt_stats'] = None
        st.session_state['user_input'] = None

    if st.button("🚀 최적 레시피 생성", type="primary"):
        user_input = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("최적화 진행 중..."):
            # 결과 저장
            rec, pred, val, stats = optimizer.generate_optimal_recipe(user_input=user_input)
            st.session_state['opt_recipe'] = rec
            st.session_state['pred_results'] = pred
            st.session_state['val_data'] = val
            st.session_state['opt_stats'] = stats
            st.session_state['user_input'] = user_input
            st.session_state['opt_done'] = True

    # 💡 결과가 있으면 화면 표시 (버튼 안 눌러도 유지됨)
    if st.session_state['opt_done']:
        opt_recipe = st.session_state['opt_recipe']
        pred_results = st.session_state['pred_results']
        val_data = st.session_state['val_data']
        
        st.success("완료!")
        
        tab1, tab2 = st.tabs(["📄 결과 리포트", "📊 최적화 경향 분석"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💡 최적 레시피")
                st.dataframe(pd.DataFrame.from_dict(opt_recipe, orient='index', columns=['Value']))
                st.subheader("🔬 물리 검증")
                st.dataframe(pd.DataFrame.from_dict(val_data, orient='index', columns=['Value']))
            with c2:
                st.subheader("📈 예측 물성")
                st.dataframe(pred_results.to_frame(name='Predicted'))
                st.metric("두께 검증", f"{pred_results['Thickness (nm)']:.2f} nm", 
                          delta=f"{pred_results['Thickness (nm)'] - st.session_state['user_input']['Thickness (nm)']:.2f} nm")

        with tab2:
            st.header("📊 최적 공정 경향 분석")
            
            col1, col2, col3 = st.columns(3)
            with col1: target_param = st.selectbox("1. X축: 목표값", ["Thickness (nm)", "Target AR"])
            with col2: y_left = st.selectbox("2. 왼쪽 Y축: 공정 조건", ["Temperature (c)", "Pressure (torr)", "Precursor Pulse Time (s)", "Purge Time (s)", "Cycles (n)"])
            with col3: y_right = st.selectbox("3. 오른쪽 Y축: 예측 물성", ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)", "Uniformity (%)"])

            curr_val = st.session_state['user_input'][target_param]
            sweep_range = np.linspace(curr_val * 0.5, curr_val * 1.5, 10)
            
            # 시각화용 데이터 생성 (이미 로드된 optimizer 사용)
            if st.button("🔄 그래프 업데이트"):
                 with st.spinner("시뮬레이션 중..."):
                    sweep_df = optimizer.simulate_target_sweep(st.session_state['user_input'], target_param, sweep_range)
                    st.session_state['sweep_df'] = sweep_df # 이것도 저장
            
            if 'sweep_df' in st.session_state:
                sweep_df = st.session_state['sweep_df']
                
                fig, ax1 = plt.subplots(figsize=(10, 5))
                line1 = ax1.plot(sweep_df[target_param], sweep_df[y_left], 'r-o', label=f"Recipe: {y_left}")
                ax1.set_xlabel(f"Target {target_param}"); ax1.set_ylabel(f"Optimal {y_left}", color='r')
                ax1.tick_params(axis='y', labelcolor='r'); ax1.grid(True, linestyle='--', alpha=0.5)
                
                ax2 = ax1.twinx()
                line2 = ax2.plot(sweep_df[target_param], sweep_df[y_right], 'b--s', label=f"Property: {y_right}")
                ax2.set_ylabel(f"Predicted {y_right}", color='b'); ax2.tick_params(axis='y', labelcolor='b')
                
                lines = line1 + line2; labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
                st.pyplot(fig)

                st.divider()
                st.subheader(f"⚖️ Step Coverage Trend (vs {target_param})")
                fig2, ax_sc = plt.subplots(figsize=(10, 4))
                ax_sc.plot(sweep_df[target_param], sweep_df['Step Coverage (sc, %)'], 'g-^', label='AI Prediction')
                ax_sc.plot(sweep_df[target_param], sweep_df['Physics SC (%)'], 'k--x', label='Physics Model')
                ax_sc.set_xlabel(f"Target {target_param}"); ax_sc.set_ylabel("Step Coverage (%)"); ax_sc.set_ylim(0, 110)
                ax_sc.legend(); ax_sc.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig2)
            else:
                st.info("👆 '그래프 업데이트' 버튼을 눌러 시각화를 시작하세요.")


# ==========================================
# 🚦 3. 실행 모드 자동 감지
# ==========================================
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx() is not None
    except ImportError:
        is_streamlit = False

    if is_streamlit:
        main_gui()
    else:
        main_cli()