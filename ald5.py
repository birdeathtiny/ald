# --- 0. 기본 라이브러리 및 Streamlit 임포트 ---
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
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

# --- 1. ALD 최적화 메인 클래스 (XGBoost 적용) ---
class ALDOptimizer:
    
    def __init__(self, file_path: str, mode: str = "cli"):
        self.mode = mode
        if self.mode == "cli":
            print(f"--- [시스템 시작] 데이터 파일 로드 중: {file_path} ---")
        
        self.DEFAULT_GPC_GUESS_A = 1.0 
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.model = None
        self.ALL_INPUT_FEATURES_ORDERED = []
        self.ALL_OUTPUT_FEATURES_ORDERED = []
        self.performance_metrics = {} # 💡 [수정됨] 오타 수정 (performance_ metrics -> performance_metrics)
        
        # 데이터 로드 및 모델 학습
        df_encoded = self._load_and_preprocess(file_path)
        self._prepare_datasets(df_encoded)
        self._train_model_xgboost()

    def _load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, encoding='CP949')
        except Exception as e:
            # 파일명만 있는 경우 현재 폴더에서 다시 시도
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
            df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O', 'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma'})
        
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
        
        # 결측치 처리 (KNN)
        imputer_X = KNNImputer(n_neighbors=5)
        self.X_train = imputer_X.fit_transform(X_train)
        self.X_test = imputer_X.transform(X_test)
        
        imputer_Y = KNNImputer(n_neighbors=5)
        self.Y_train = imputer_Y.fit_transform(Y_train)
        self.Y_test = imputer_Y.transform(Y_test)
        
        # 스케일링
        self.X_train_scaled = self.X_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.X_scaler.transform(self.X_test)
        self.Y_train_scaled = self.Y_scaler.fit_transform(self.Y_train)
        self.Y_test_scaled = self.Y_scaler.transform(self.Y_test)

        if self.mode == "cli":
            print(f"✅ 데이터 준비 완료 (입력: {X.shape[1]}개, 출력: {Y.shape[1]}개)")

    def _train_model_xgboost(self):
        """XGBoost 모델 학습 (MultiOutput)"""
        if self.mode == "cli": print("--- 🤖 XGBoost 모델 학습 시작 ---")
        
        # XGBoost Regressor 설정 (성능 튜닝된 파라미터)
        xgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42
        }
        
        # 다중 출력 회귀 모델 래핑
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        self.model.fit(self.X_train_scaled, self.Y_train_scaled)
        
        # 평가
        Y_pred_scaled = self.model.predict(self.X_test_scaled)
        Y_pred = self.Y_scaler.inverse_transform(Y_pred_scaled)
        
        r2 = r2_score(self.Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(self.Y_test, Y_pred))
        
        self.performance_metrics = {'R2': r2, 'RMSE': rmse}
        
        if self.mode == "cli":
            print(f"✅ 모델 학습 완료 | R2 Score: {r2:.4f} (높을수록 좋음)")

    # --- 물리 모델 (SC) ---
    @staticmethod
    def _calculate_physics_sc(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
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
        
        if phi < 1.0: sc = 1.0 / (1.0 + phi)
        else: sc = np.exp(-phi)
        
        return float(np.clip(sc * 100.0, 0.0, 100.0))

    # --- 예측 및 최적화 ---
    def _create_input_df(self, params, precursor, co_reactant, purge_gas):
        df = pd.DataFrame(0.0, index=[0], columns=self.ALL_INPUT_FEATURES_ORDERED)
        for k, v in params.items():
            if k in df.columns: df.at[0, k] = v
        
        for col in [f"Precursor_{precursor}", f"Co-reactant_{co_reactant}", f"Purge Gas_{purge_gas}"]:
            if col in df.columns: df.at[0, col] = 1.0
        return df

    def predict(self, params, precursor, co_reactant, purge_gas):
        input_df = self._create_input_df(params, precursor, co_reactant, purge_gas)
        X_scaled = self.X_scaler.transform(input_df.values)
        Y_pred_scaled = self.model.predict(X_scaled)
        Y_pred = self.Y_scaler.inverse_transform(Y_pred_scaled)[0]
        return pd.Series(Y_pred, index=self.ALL_OUTPUT_FEATURES_ORDERED).round(4)

    def optimize(self, user_input, silent=False):
        # 1. 초기값 및 경계 설정
        initial_cycles = max(10, int(user_input["Thickness (nm)"] * 10))
        co_reactant = 'H2O' if user_input["Precursor"] in ['TMA', 'TDMAH'] else 'O3'
        
        # [Temp, Pressure, Pulse, PurgeT, PurgeF]
        bounds = [(150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500)]
        x0 = [np.random.uniform(l, h) for l, h in bounds]
        
        if self.mode == "cli" and not silent:
            print(f"\n--- 🔍 최적화 탐색 시작 (Start: {np.round(x0, 2)}) ---")

        # 2. 목적 함수 (Cost Function)
        def objective(x):
            params = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], 
                "Precursor_Pulse Time (s)": x[2], "Co-reactant_Pulse Time (s)": x[2],
                "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
                "Cycles (n)": initial_cycles
            }
            try:
                pred = self.predict(params, user_input["Precursor"], co_reactant, "N2")
                gpc_pred = pred.get("GPC (A/cycle)", 0.1)
                rough_pred = pred.get("Surface Roughness (RMS, nm)", 10)
                
                # GPC 목표: (두께 * 10) / 사이클
                target_gpc = (user_input["Thickness (nm)"] * 10) / initial_cycles
                
                cost = 10000 * (gpc_pred - target_gpc)**2 + 10 * (rough_pred**2)
                return cost
            except: return 1e9

        # 3. 제약 조건 (SC)
        def constraint(x):
            sc = self._calculate_physics_sc(
                x[1], x[0], x[2], user_input["Target AR"], 
                user_input["Precursor"], user_input["CD (nm)"]*1e-9
            )
            target_sc = 90.0 if user_input["Target AR"] <= 15 else 85.0
            return sc - target_sc

        # 4. 최적화 실행
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': constraint}, options={'maxiter': 50})
        
        # 5. 결과 정리 (역산 포함)
        x = res.x
        check_params = {
            "Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
            "Co-reactant_Pulse Time (s)": x[2], "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
            "Cycles (n)": initial_cycles
        }
        check_pred = self.predict(check_params, user_input["Precursor"], co_reactant, "N2")
        
        final_gpc = max(0.001, check_pred.get("GPC (A/cycle)", 1.0))
        final_cycles = int(round((user_input["Thickness (nm)"] * 10) / final_gpc))
        
        # 최종 예측
        final_params = check_params.copy()
        final_params["Cycles (n)"] = final_cycles
        final_pred = self.predict(final_params, user_input["Precursor"], co_reactant, "N2")
        final_pred["Thickness (nm)"] = (final_gpc * final_cycles) / 10.0 # 두께 보정

        # 레시피 포맷팅
        recipe = {
            "Precursor": user_input["Precursor"], "Co-reactant": co_reactant,
            "Temperature (c)": round(x[0], 1), "Pressure (torr)": round(x[1], 3),
            "Cycles (n)": final_cycles, "Pulse Time (s)": round(x[2], 2),
            "Purge Time (s)": round(x[3], 1), "Purge Flow (sccm)": int(x[4])
        }
        
        phys_sc = self._calculate_physics_sc(x[1], x[0], x[2], user_input["Target AR"], user_input["Precursor"], user_input["CD (nm)"]*1e-9)
        valid = {"Physics SC (%)": f"{phys_sc:.2f}%", "Cost": f"{res.fun:.4f}"}
        
        if self.mode == "cli" and not silent:
            print("✅ 최적화 완료!")

        return recipe, final_pred, valid

    # 시뮬레이션 (그래프용)
    def simulate_sweep(self, user_input, target_col, sweep_values):
        results = []
        for val in sweep_values:
            temp_input = user_input.copy()
            temp_input[target_col] = val
            rec, pred, val_data = self.optimize(temp_input, silent=True)
            
            row = {target_col: val}
            # 레시피 저장 (숫자만)
            row.update({k: v for k, v in rec.items() if isinstance(v, (int, float))})
            # 물성 저장
            row.update(pred.to_dict())
            # 물리 SC 저장
            row["Physics SC (%)"] = float(val_data["Physics SC (%)"].replace('%', ''))
            results.append(row)
        return pd.DataFrame(results)


# ==========================================
# 🖥️ CLI 모드 (터미널)
# ==========================================
def main_cli():
    print("\n" + "="*50 + "\n  [CLI] High-Accuracy XGBoost ALD Optimizer\n" + "="*50)
    
    # Matplotlib 팝업 설정
    import matplotlib
    try: matplotlib.use('TkAgg')
    except: pass

    file_name = "AI_ALD1.csv"
    if not os.path.exists(file_name):
        print(f"[오류] '{file_name}' 파일이 없습니다."); return

    optimizer = ALDOptimizer(file_path=file_name, mode="cli")
    print(f"📊 모델 정확도 (R2): {optimizer.performance_metrics['R2']:.4f}")

    # 입력
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
    
    # 최적화
    recipe, pred, valid = optimizer.optimize(user_input)
    
    print("\n" + "-"*30)
    print(f"💡 최적 레시피:\n{pd.Series(recipe).to_string()}")
    print("\n📈 예측 물성:\n{pred.to_string()}")
    print("-" * 30)

    # 시각화
    print("\n📊 [시각화] 목표값 변화에 따른 경향 분석")
    x_opts = ["Thickness (nm)", "Target AR"]
    print(f"1. {x_opts[0]}  2. {x_opts[1]}")
    x_idx = int(input("=> X축 선택 (1/2): ")) - 1
    target_param = x_opts[x_idx]

    print(f"📈 '{target_param}' 변화 시뮬레이션 중...")
    curr = user_input[target_param]
    sweep_range = np.linspace(curr * 0.5, curr * 1.5, 10)
    df = optimizer.simulate_sweep(user_input, target_param, sweep_range)

    plt.figure(figsize=(12, 5))
    
    # Graph 1: Recipe & Property (Dual Axis)
    ax1 = plt.subplot(1, 2, 1)
    l1 = ax1.plot(df[target_param], df["Temperature (c)"], 'r-o', label="Temp (c)")
    ax1.set_xlabel(target_param); ax1.set_ylabel("Temp (c)", color='r')
    
    ax2 = ax1.twinx()
    l2 = ax2.plot(df[target_param], df["GPC (A/cycle)"], 'b--s', label="GPC")
    ax2.set_ylabel("GPC (A/cycle)", color='b')
    
    lns = l1 + l2; labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title("Temp & GPC Trend")

    # Graph 2: SC Comparison
    plt.subplot(1, 2, 2)
    plt.plot(df[target_param], df["Step Coverage (sc, %)"], 'g-^', label="AI SC")
    plt.plot(df[target_param], df["Physics SC (%)"], 'k--x', label="Physics SC")
    plt.legend(); plt.title("Step Coverage Trend")
    
    plt.tight_layout()
    try: plt.show()
    except: print("⚠️ 팝업 불가. result.png 저장"); plt.savefig("result.png")


# ==========================================
# 🌐 GUI 모드 (Streamlit)
# ==========================================
def main_gui():
    st.set_page_config(page_title="High-Acc ALD Optimizer", layout="wide")
    st.title("🚀 고성능 AI ALD 공정 최적화 (XGBoost Ver.)")

    @st.cache_resource
    def load_model():
        return ALDOptimizer("AI_ALD1.csv", mode="gui")

    try: optimizer = load_model()
    except Exception as e: st.error(f"오류: {e}"); st.stop()

    # 사이드바 입력
    st.sidebar.header("🎯 목표 설정")
    sel_p = st.sidebar.selectbox("전구체", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("두께 (nm)", 1.0, 500.0, 15.0)
    ar = st.sidebar.number_input("AR", 1.0, 100.0, 10.0)
    cd = st.sidebar.number_input("CD (nm)", 1.0, 1000.0, 100.0)

    if 'res' not in st.session_state: st.session_state.res = None

    if st.sidebar.button("최적화 실행", type="primary"):
        with st.spinner("최적해 탐색 중..."):
            u_in = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
            rec, pred, val = optimizer.optimize(u_in)
            st.session_state.res = (rec, pred, val, u_in)

    if st.session_state.res:
        rec, pred, val, u_in = st.session_state.res
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💡 최적 레시피")
            st.dataframe(pd.DataFrame([rec]).T.rename(columns={0:"Value"}))
            st.info(f"물리검증 SC: {val['Physics SC (%)']}")
        with c2:
            st.subheader("📈 AI 예측 결과")
            st.dataframe(pred.to_frame("Predicted"))
            st.metric("두께 검증", f"{pred['Thickness (nm)']:.2f} nm", delta=f"{pred['Thickness (nm)'] - u_in['Thickness (nm)']:.2f} nm")

        st.divider()
        st.header("📊 시뮬레이션 (Dual Axis Chart)")
        
        col1, col2, col3 = st.columns(3)
        target = col1.selectbox("X축 (목표)", ["Thickness (nm)", "Target AR"])
        y1 = col2.selectbox("Y1 (좌측/빨강)", ["Temperature (c)", "Pressure (torr)", "Pulse Time (s)", "Cycles (n)"])
        y2 = col3.selectbox("Y2 (우측/파랑)", ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)"])

        if st.button("🔄 그래프 업데이트"):
            with st.spinner("시뮬레이션..."):
                curr = u_in[target]
                rng = np.linspace(curr*0.5, curr*1.5, 10)
                df = optimizer.simulate_sweep(u_in, target, rng)
                
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df[target], df[y1], 'r-o', label=y1)
                ax1.set_ylabel(y1, color='r'); ax1.tick_params(axis='y', labelcolor='r')
                
                ax2 = ax1.twinx()
                ax2.plot(df[target], df[y2], 'b--s', label=y2)
                ax2.set_ylabel(y2, color='b'); ax2.tick_params(axis='y', labelcolor='b')
                
                st.pyplot(fig)
                
                # SC 비교 그래프
                st.subheader("⚖️ SC: AI vs Physics")
                fig2, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df[target], df["Step Coverage (sc, %)"], 'g-', label="AI")
                ax.plot(df[target], df["Physics SC (%)"], 'k--', label="Physics")
                ax.legend()
                st.pyplot(fig2)

# ==========================================
# 🚦 실행 모드
# ==========================================
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(): main_gui()
        else: main_cli()
    except: main_cli()