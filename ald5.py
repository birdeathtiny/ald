# --- 0. 기본 라이브러리 및 Streamlit 임포트 ---
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple

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

# --- 1. ALD 최적화 메인 클래스 ---
class ALDOptimizer:
    
    def __init__(self, file_path: str, mode: str = "cli"):
        self.mode = mode
        if self.mode == "cli":
            print(f"--- [Smart Tuning Mode] 데이터 로드 및 스마트 학습 시작 ---")
        
        self.DEFAULT_GPC_GUESS_A = 1.0 
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.model = None
        self.ALL_INPUT_FEATURES_ORDERED = []
        self.ALL_OUTPUT_FEATURES_ORDERED = []
        self.performance_metrics = {}
        self.best_params = {} 
        
        # 1. 데이터 로드
        df_encoded = self._load_and_preprocess(file_path)
        
        # 2. 데이터셋 준비
        self._prepare_datasets(df_encoded)
        
        # 3. 모델 학습 (스마트 튜닝: 대표 타겟으로 파라미터 찾기)
        self._train_model_smart()

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(current_dir, file_path)
                df = pd.read_csv(full_path, encoding='CP949')
            except:
                msg = f"[오류] 파일을 찾을 수 없습니다: {file_path}"
                if self.mode == "cli": print(msg); sys.exit(1)
                else: st.error(msg); st.stop()

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
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Co-reactant' in df.columns:
            df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O', 'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'})
        
        cols_to_drop = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', '순서']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        categorical_cols = [c for c in ['Precursor', 'Co-reactant', 'Purge Gas'] if c in df.columns]
        return pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    def _prepare_datasets(self, df_encoded: pd.DataFrame):
        target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)', 'Leakage Current Density (A/cm2)', 
            'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)', 'Step Coverage (sc, %)'
        ]
        cols_to_ignore = ['Aspect Ratio (AR)']
        
        avail_targets = [c for c in target_cols if c in df_encoded.columns]
        avail_drops = [c for c in cols_to_ignore if c in df_encoded.columns]
        
        self.ALL_OUTPUT_FEATURES_ORDERED = avail_targets
        self.ALL_INPUT_FEATURES_ORDERED = df_encoded.drop(columns=avail_targets + avail_drops).columns.tolist()

        X = df_encoded[self.ALL_INPUT_FEATURES_ORDERED].values
        Y = df_encoded[self.ALL_OUTPUT_FEATURES_ORDERED].values
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        imputer_X = KNNImputer(n_neighbors=5)
        self.X_train = imputer_X.fit_transform(X_train)
        self.X_test = imputer_X.transform(X_test)
        
        imputer_Y = KNNImputer(n_neighbors=5)
        self.Y_train = imputer_Y.fit_transform(Y_train)
        self.Y_test = imputer_Y.transform(Y_test)
        
        self.X_train_scaled = self.X_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.X_scaler.transform(self.X_test)
        self.Y_train_scaled = self.Y_scaler.fit_transform(self.Y_train)
        self.Y_test_scaled = self.Y_scaler.transform(self.Y_test)

        if self.mode == "cli":
            print(f"✅ 데이터 로드 완료 (입력: {X.shape[1]}개, 출력: {Y.shape[1]}개)")

    def _train_model_smart(self):
        """
        [Smart Tuning]
        전체 타겟을 다 튜닝하면 느립니다. 
        대표 타겟(Thickness) 하나로 최적 파라미터를 찾고(약 3초), 전체 모델에 적용합니다.
        """
        if self.mode == "cli": print("--- 🧠 AI가 최적의 파라미터를 탐색 중입니다 (약 3~5초 소요)... ---")
        
        param_dist = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.03, 0.05, 0.1],
            'max_depth': [4, 5, 6],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }
        
        # 1. 대표 타겟(0번: Thickness)으로만 튜닝 -> 속도 9배 향상
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(n_jobs=-1, random_state=42),
            param_distributions=param_dist,
            n_iter=10,   # 10번 실험 (충분함)
            cv=2,        # 2-Fold 검증
            scoring='neg_mean_squared_error',
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(self.X_train_scaled, self.Y_train_scaled[:, 0]) 
        self.best_params = search.best_params_
        
        if self.mode == "cli":
            print(f"✨ 최적 파라미터 발견: {self.best_params}")
            print("--- 🤖 전체 모델 학습 적용 중... ---")
        
        # 2. 찾은 파라미터로 전체 타겟 학습
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**self.best_params, n_jobs=-1, random_state=42))
        self.model.fit(self.X_train_scaled, self.Y_train_scaled)
        
        # 3. 평가
        Y_pred_scaled = self.model.predict(self.X_test_scaled)
        Y_pred = self.Y_scaler.inverse_transform(Y_pred_scaled)
        
        r2 = r2_score(self.Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(self.Y_test, Y_pred))
        
        RMSE_scores = np.sqrt(mean_squared_error(self.Y_test, Y_pred, multioutput='raw_values'))
        R2_scores = r2_score(self.Y_test, Y_pred, multioutput='raw_values')
        
        R2_dict = dict(zip(self.ALL_OUTPUT_FEATURES_ORDERED, R2_scores.round(4)))
        RMSE_dict = dict(zip(self.ALL_OUTPUT_FEATURES_ORDERED, RMSE_scores.round(4)))
        
        self.performance_metrics = {'R2': r2, 'RMSE': rmse}
        self.performance_df = pd.DataFrame({'RMSE': RMSE_dict, 'R^2': R2_dict})
        
        if self.mode == "cli":
            print(f"✅ 학습 완료 | 평균 R2 Score: {r2:.4f}")

    # --- 물리 모델 (SC) ---
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
        v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_kg))
        lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
        D_eff = 1.0 / ((1.0 / ((1/3)*lambda_m*v_avg + 1e-30)) + (1.0 / ((1/3)*v_avg*CD_m + 1e-30)))
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
            print(f"\n--- 🔍 최적화 탐색 시작 ---")

        args = (user_input, co_reactant, purge_gas, COST_WEIGHTS, initial_cycles)
        result = minimize(self._objective_function, initial_guess, args=args, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': self._constraint_sc, 'args': args}, options={'maxiter': 30, 'eps': 1e-6})
        
        x = result.x
        check_params = {"Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                        "Co-reactant_Pulse Time (s)": x[2], "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
                        "Cycles (n)": initial_cycles}
        check_pred = self.predict(check_params, user_input["Precursor"], co_reactant, "N2")
        final_gpc = max(0.001, check_pred.get("GPC (A/cycle)", 1.0))
        final_cycles = int(round((user_input["Thickness (nm)"] * 10) / final_gpc))
        
        final_params = check_params.copy()
        final_params["Cycles (n)"] = final_cycles
        final_pred = self.predict(final_params, user_input["Precursor"], co_reactant, "N2")
        final_pred["Thickness (nm)"] = (final_gpc * final_cycles) / 10.0 

        recipe = {
            "Precursor": user_input["Precursor"], "Co-reactant": co_reactant,
            "Temperature (c)": round(x[0], 1), "Pressure (torr)": round(x[1], 3),
            "Cycles (n)": final_cycles, "Pulse Time (s)": round(x[2], 2),
            "Purge Time (s)": round(x[3], 1), "Purge Flow (sccm)": int(x[4])
        }
        
        phys_sc = self._calculate_physics_sc(x[1], x[0], x[2], user_input["Target AR"], user_input["Precursor"], user_input["CD (nm)"]*1e-9)
        sc_val, phi, mode = self._calculate_physics_sc_details(x[1], x[0], x[2], user_input["Target AR"], user_input["Precursor"], user_input["CD (nm)"]*1e-9)
        lambda_m, Kn = self._calculate_physical_parameters(x[0], x[1], user_input["Precursor"], user_input["CD (nm)"]*1e-9)
        valid = {"Mean Free Path (λ) [m]": f"{lambda_m:.2e}", "Knudsen Number (Kn)": f"{Kn:.2f}", 
                 "Thiele Modulus (φ)": f"{phi:.4f}", "Transport Mode": mode, "Physics SC (%)": f"{sc_val:.2f}%", "Cost": f"{res.fun:.4f}"}
        
        if self.mode == "cli" and not silent:
            print("✅ 최적화 완료!")

        return recipe, final_pred, valid

    def simulate_target_sweep(self, base_user_input, target_param_name, range_values):
        results = []
        for val in range_values:
            current_input = base_user_input.copy()
            current_input[target_param_name] = val
            rec, pred, val_data = self.generate_optimal_recipe(current_input, silent=True)
            
            phys_sc = self._calculate_physics_sc(
                rec['Pressure (torr)'], rec['Temperature (c)'], rec['Pulse Time (s)'],
                current_input['Target AR'], current_input['Precursor'], current_input['CD (nm)'] * 1e-9
            )
            row = {target_param_name: val}
            row.update({k: v for k, v in rec.items() if isinstance(v, (int, float))})
            row.update(pred.to_dict())
            row["Physics SC (%)"] = phys_sc
            results.append(row)
        return pd.DataFrame(results)


# ==========================================
# 🖥️ CLI 모드 실행 함수
# ==========================================
def main_cli():
    print("\n" + "="*50 + "\n  [CLI] ALD AI Optimizer (Smart Tuning)\n" + "="*50)
    
    import matplotlib
    try: matplotlib.use('TkAgg')
    except: pass 

    csv_file = "AI_ALD1.csv" 
    if not os.path.exists(csv_file):
        print(f"[오류] '{csv_file}' 파일이 없습니다."); return
    
    optimizer = ALDOptimizer(file_path=csv_file, mode="cli")
    print(f"📊 모델 정확도 (R2): {optimizer.performance_metrics['R2']:.4f}")

    try:
        p_list = ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"]
        print("\n[전구체 목록]: " + ", ".join([f"{i+1}.{p}" for i, p in enumerate(p_list)]))
        p_idx = int(input("=> 번호 입력: ")) - 1
        sel_p = p_list[p_idx]
        th = float(input("=> 목표 두께 (nm): "))
        ar = float(input("=> 목표 AR: "))
        cd = float(input("=> CD (nm): "))
    except: print("[입력 오류]"); return

    user_input = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
    recipe, pred, valid = optimizer.generate_optimal_recipe(user_input)
    
    print("\n" + "-"*30)
    print(f"💡 최적 레시피:\n{pd.Series(recipe).to_string()}")
    print("\n📈 예측 물성:\n{pred.to_string()}")
    print("-" * 30)

    # 시각화
    print("\n📊 [시각화] 목표값 변화에 따른 경향 분석")
    x_opts = ["Thickness (nm)", "Target AR"]
    print(f"1. {x_opts[0]}  2. {x_opts[1]}")
    try: x_idx = int(input("=> X축 선택 (1/2): ")) - 1
    except: x_idx = 0
    target_param = x_opts[x_idx]

    print(f"📈 '{target_param}' 변화 시뮬레이션 중...")
    curr = user_input[target_param]
    sweep_range = np.linspace(curr * 0.5, curr * 1.5, 10)
    df = optimizer.simulate_target_sweep(user_input, target_param, sweep_range)

    plt.figure(figsize=(14, 5))
    ax1 = plt.subplot(1, 2, 1)
    l1 = ax1.plot(df[target_param], df["Temperature (c)"], 'r-o', label="Temp (c)")
    ax1.set_xlabel(target_param); ax1.set_ylabel("Temp (c)", color='r')
    ax2 = ax1.twinx()
    l2 = ax2.plot(df[target_param], df["GPC (A/cycle)"], 'b--s', label="GPC")
    ax2.set_ylabel("GPC (A/cycle)", color='b')
    lns = l1 + l2; labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title("Temp & GPC Trend")

    plt.subplot(1, 2, 2)
    plt.plot(df[target_param], df["Step Coverage (sc, %)"], 'g-^', label="AI SC")
    plt.plot(df[target_param], df["Physics SC (%)"], 'k--x', label="Physics SC")
    plt.legend(); plt.title("Step Coverage Trend")
    
    plt.tight_layout()
    try: plt.show(); print("   (그래프 창을 닫으면 종료됩니다)")
    except Exception as e: print(f"\n⚠️ 그래프 팝업 불가: {e}"); plt.savefig('result_graph.png')


# ==========================================
# 🌐 GUI 모드 (Streamlit)
# ==========================================
def main_gui():
    st.set_page_config(page_title="ALD Optimizer", layout="wide")
    st.title("🚀 AI 기반 ALD 공정 최적화 시스템")

    @st.cache_resource(show_spinner="AI 모델 스마트 튜닝 중 (약 3초)...")
    def load_model():
        csv_file_name = "AI_ALD1.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, csv_file_name)
        if not os.path.exists(full_path): full_path = csv_file_name
        return ALDOptimizer(full_path, mode="gui")

    try: optimizer = load_model()
    except Exception as e: st.error(f"오류: {e}"); st.stop()

    st.sidebar.header("🎯 목표 설정")
    sel_p = st.sidebar.selectbox("전구체", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("두께 (nm)", 1.0, 500.0, 15.0)
    ar = st.sidebar.number_input("AR", 1.0, 100.0, 10.0)
    cd = st.sidebar.number_input("CD (nm)", 1.0, 1000.0, 100.0)

    if 'opt_done' not in st.session_state:
        st.session_state['opt_done'] = False
        st.session_state['opt_recipe'] = None
        st.session_state['pred_results'] = None
        st.session_state['val_data'] = None
        st.session_state['opt_stats'] = None
        st.session_state['user_input'] = None

    if st.sidebar.button("최적화 실행", type="primary"):
        user_input = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("최적해 탐색 중..."):
            rec, pred, val, stats = optimizer.generate_optimal_recipe(user_input=user_input)
            st.session_state['opt_recipe'] = rec
            st.session_state['pred_results'] = pred
            st.session_state['val_data'] = val
            st.session_state['opt_stats'] = stats
            st.session_state['user_input'] = user_input
            st.session_state['opt_done'] = True

    if st.session_state['opt_done']:
        rec = st.session_state.opt_recipe
        pred = st.session_state.pred_results
        val = st.session_state.val_data
        u_in = st.session_state.user_input
        
        tab1, tab2 = st.tabs(["결과 리포트", "시뮬레이션 그래프"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💡 최적 레시피")
                st.dataframe(pd.DataFrame([rec]).T.rename(columns={0:"Value"}))
                st.info(f"물리검증 SC: {val['Physics SC (%)']}")
            with c2:
                st.subheader("📈 AI 예측 결과")
                st.dataframe(pred.to_frame("Predicted"))
                st.metric("두께 검증", f"{pred['Thickness (nm)']:.2f} nm", delta=f"{pred['Thickness (nm)'] - u_in['Thickness (nm)']:.2f} nm")
            
            st.markdown("---")
            st.caption(f"AI Best Params: {optimizer.best_params}")
            st.dataframe(optimizer.performance_df.T)

        with tab2:
            st.header("📊 목표값 변화 시뮬레이션")
            col1, col2, col3 = st.columns(3)
            target = col1.selectbox("X축 (목표)", ["Thickness (nm)", "Target AR"])
            y1 = col2.selectbox("Y1 (좌측)", ["Temperature (c)", "Pressure (torr)", "Pulse Time (s)", "Cycles (n)"])
            y2 = col3.selectbox("Y2 (우측)", ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)"])

            if st.button("🔄 그래프 업데이트"):
                with st.spinner("시뮬레이션 중..."):
                    curr = u_in[target]
                    rng = np.linspace(curr*0.5, curr*1.5, 10)
                    df = optimizer.simulate_target_sweep(u_in, target, rng)
                    st.session_state['sweep_df'] = df
            
            if 'sweep_df' in st.session_state:
                df = st.session_state['sweep_df']
                if target not in df.columns:
                    st.warning("⚠️ 설정이 변경되었습니다. 업데이트 버튼을 눌러주세요.")
                else:
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(df[target], df[y1], 'r-o', label=f"Recipe: {y_left}")
                    ax1.set_ylabel(y1, color='r'); ax1.tick_params(axis='y', labelcolor='r')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(df[target], df[y2], 'b--s', label=f"Property: {y_right}")
                    ax2.set_ylabel(y2, color='b'); ax2.tick_params(axis='y', labelcolor='b')
                    
                    lines = ax1.get_lines() + ax2.get_lines()
                    ax1.legend(lines, [l.get_label() for l in lines])
                    st.pyplot(fig)
                    
                    st.subheader("⚖️ SC: AI vs Physics")
                    fig2, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(df[target], df["Step Coverage (sc, %)"], 'g-', label="AI")
                    ax.plot(df[target], df["Physics SC (%)"], 'k--', label="Physics")
                    ax.set_xlabel(target); ax.set_ylabel("SC (%)")
                    ax.legend()
                    st.pyplot(fig2)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(): main_gui()
        else: main_cli()
    except: main_cli()