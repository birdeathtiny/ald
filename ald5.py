# ==============================================================================
#  [Enterprise-Grade] AI ALD Process Optimizer (XGBoost + Feature Engineering)
# ==============================================================================
#  1. Data Management: Loading, Preprocessing, Polynomial Feature Generation
#  2. Model Management: XGBoost w/ RandomizedSearchCV, Feature Importance Analysis
#  3. Physics Engine: Step Coverage & Knudsen Diffusion Calculations
#  4. Optimization Engine: Constrained SLSQP Optimization
#  5. UI/UX: CLI (Terminal) & GUI (Streamlit) Support
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib
if "streamlit" not in sys.modules:
    try: matplotlib.use('TkAgg')
    except: pass
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple

# --- 0. Constants & Configuration ---
PRECURSOR_CONSTANTS = {
    "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005},
    "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001},
    "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005},
    "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008}
}
N_A = 6.022e23
k_B = 1.38e-23

# ==========================================
# 1. Data Manager Class
# ==========================================
class ALDDataManager:
    def __init__(self, file_path: str, mode: str):
        self.mode = mode
        self.file_path = file_path
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        # 💡 [Feature Engineering] 변수 간의 상호작용(곱하기 등)을 추가하여 AI의 통찰력 강화
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        self._load_data()
        self._process_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
             # Try finding in current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(current_dir, os.path.basename(self.file_path))
            if os.path.exists(alt_path):
                self.file_path = alt_path
            else:
                msg = f"❌ [Error] Data file not found: {self.file_path}"
                if self.mode == "cli": print(msg); sys.exit(1)
                else: st.error(msg); st.stop()
                
        try: self.df = pd.read_csv(self.file_path, encoding='CP949')
        except: self.df = pd.read_csv(self.file_path)
        
        if self.mode == "cli": print(f"✅ Data loaded successfully: {len(self.df)} rows")

    def _process_data(self):
        # 1. Clean Data
        self.df.replace('-', np.nan, inplace=True)
        
        cols_num = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)',
            'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Thickness (nm)', 
            'Surface Roughness (RMS, nm)', 'Uniformity (%)', 'Step Coverage (sc, %)', 
            'Density (g/cm3)', 'GPC (A/cycle)', 'Aspect Ratio (AR)', 
            'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)'
        ]
        for c in cols_num:
            if c in self.df.columns: self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # 2. Encode Categoricals
        if 'Co-reactant' in self.df.columns:
            self.df['Co-reactant'] = self.df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
            
        drop_cols = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', '순서']
        self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)
        
        cat_cols = [c for c in ['Precursor', 'Co-reactant', 'Purge Gas'] if c in self.df.columns]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, dummy_na=False)

        # 3. Split X/Y
        self.target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)', 'Leakage Current Density (A/cm2)', 
            'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)', 'Step Coverage (sc, %)'
        ]
        ignore_cols = ['Aspect Ratio (AR)']
        
        self.targets = [c for c in self.target_cols if c in self.df_encoded.columns]
        self.inputs = self.df_encoded.drop(columns=self.targets + [c for c in ignore_cols if c in self.df_encoded.columns]).columns.tolist()

        X_raw = self.df_encoded[self.inputs].values
        Y_raw = self.df_encoded[self.targets].values
        
        # 4. Impute & Feature Engineering
        imputer_x = KNNImputer(n_neighbors=5)
        X_imputed = imputer_x.fit_transform(X_raw)
        
        # 💡 다항 특성 생성 (Poly Features) -> 데이터 컬럼 수 증가
        X_poly = self.poly.fit_transform(X_imputed) 
        
        imputer_y = KNNImputer(n_neighbors=5)
        Y_imputed = imputer_y.fit_transform(Y_raw)

        # 5. Scale & Split
        X_scaled = self.X_scaler.fit_transform(X_poly)
        Y_scaled = self.Y_scaler.fit_transform(Y_imputed)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_scaled, Y_scaled, test_size=0.2, random_state=42
        )
        self.Y_test_raw = self.Y_scaler.inverse_transform(self.Y_test)
        
        # feature names update (poly features 포함)
        self.feature_names = self.poly.get_feature_names_out(self.inputs)
        
        if self.mode == "cli": 
            print(f"✅ Feature Engineering Completed. Input Features: {X_raw.shape[1]} -> {X_poly.shape[1]} (Expanded)")


# ==========================================
# 2. Model Manager Class
# ==========================================
class ALDXGBoostModel:
    def __init__(self, data_manager: ALDDataManager, mode: str):
        self.dm = data_manager
        self.mode = mode
        self.model = None
        self.best_params = {}
        self.metrics = {}
        self._train()

    def _train(self):
        if self.mode == "cli": 
            print("--- 🧠 AI Auto-Tuning & Training Started (Wait for optimization...) ---")
        
        # 1. Define Search Space
        param_dist = {
            'n_estimators': [300, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'min_child_weight': [1, 3]
        }
        
        # 2. Randomized Search (Single Target Optimization for Speed)
        # 대표적으로 Thickness(0번)를 기준으로 최적 파라미터를 찾음
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(n_jobs=-1, random_state=42),
            param_distributions=param_dist,
            n_iter=10, # 10 experiments
            cv=3,      # 3-fold CV
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1
        )
        
        search.fit(self.dm.X_train, self.dm.Y_train[:, 0])
        self.best_params = search.best_params_
        
        if self.mode == "cli": print(f"✨ Optimized Params Found: {self.best_params}")
        
        # 3. Final Training (Multi-Output)
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**self.best_params, n_jobs=-1, random_state=42))
        self.model.fit(self.dm.X_train, self.dm.Y_train)
        
        # 4. Evaluation
        Y_pred_scaled = self.model.predict(self.dm.X_test)
        Y_pred = self.dm.Y_scaler.inverse_transform(Y_pred_scaled)
        
        r2 = r2_score(self.dm.Y_test_raw, Y_pred)
        rmse = np.sqrt(mean_squared_error(self.dm.Y_test_raw, Y_pred))
        self.metrics = {'R2': r2, 'RMSE': rmse}
        
        if self.mode == "cli": print(f"✅ Model Training Finished | R2 Score: {r2:.4f}")

    def predict(self, input_vector):
        # Feature Engineering 적용 (Poly)
        input_poly = self.dm.poly.transform(input_vector)
        # Scaling
        input_scaled = self.dm.X_scaler.transform(input_poly)
        # Prediction
        pred_scaled = self.model.predict(input_scaled)
        # Inverse Scaling
        return self.dm.Y_scaler.inverse_transform(pred_scaled)[0]

    def get_feature_importance(self):
        # MultiOutput이므로 첫 번째 Estimator(예: GPC)의 중요도를 가져옴
        importances = self.model.estimators_[4].feature_importances_ # 4번: GPC
        indices = np.argsort(importances)[::-1]
        top_n = 10
        return [self.dm.feature_names[i] for i in indices[:top_n]], importances[indices[:top_n]]


# ==========================================
# 3. Physics Engine Class
# ==========================================
class ALDPhysics:
    @staticmethod
    def calc_sc(P, T, Pulse, AR, Precursor, CD_nm):
        try:
            const = PRECURSOR_CONSTANTS.get(Precursor, PRECURSOR_CONSTANTS["TMA"])
            T_K = T + 273.15; P_Pa = P * 133.322; L = AR * (CD_nm * 1e-9)
            d = const["diameter_m"]; m_kg = const["mass_g_mol"] / 1000 / N_A
            
            v_avg = np.sqrt(8 * k_B * T_K / (np.pi * m_kg))
            lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
            
            # Knudsen Diffusion
            D_Kn = (1/3) * v_avg * (CD_nm * 1e-9)
            D_bulk = (1/3) * lambda_m * v_avg
            D_eff = 1 / (1/D_Kn + 1/D_bulk)
            
            lambda_pen = np.sqrt(D_eff * Pulse + 1e-12)
            phi = L / (lambda_pen + 1e-12)
            
            if phi < 1.0: return 100.0 / (1.0 + phi)
            else: return np.exp(-phi) * 100.0
        except: return 0.0


# ==========================================
# 4. Optimizer Engine Class
# ==========================================
class ALDOptimizer:
    def __init__(self, data_manager, model_manager, mode="cli"):
        self.dm = data_manager
        self.mm = model_manager
        self.mode = mode

    def _construct_input(self, params, precursor, co_reactant, purge):
        # DataFrame 생성 -> OneHot -> Poly -> Scale
        row = pd.DataFrame(0.0, index=[0], columns=self.dm.inputs)
        for k, v in params.items():
            if k in row.columns: row.at[0, k] = v
        
        for col in [f"Precursor_{precursor}", f"Co-reactant_{co_reactant}", f"Purge Gas_{purge}"]:
            if col in row.columns: row.at[0, col] = 1.0
        
        return row.values

    def optimize(self, user_in):
        init_cycles = max(10, int(user_in["Thickness (nm)"] * 10))
        
        def objective(x):
            params = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
                "Cycles (n)": init_cycles, "Co-reactant_Pulse Time (s)": x[2]
            }
            try:
                vec = self._construct_input(params, user_in["Precursor"], 'H2O', 'N2')
                pred = self.mm.predict(vec)
                # Index 4: GPC, Index 1: Roughness (Based on target list order)
                gpc_pred = pred[4]; rough_pred = pred[1]
                target_gpc = (user_in["Thickness (nm)"] * 10) / init_cycles
                return 10000 * (gpc_pred - target_gpc)**2 + 10 * (rough_pred**2)
            except: return 1e9

        bounds = [(150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500)]
        x0 = [250, 0.5, 0.1, 5.0, 200] # Smart Initial Guess
        
        # SLSQP Optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 30, 'eps': 1e-6})
        
        # Final Calculation
        x = res.x
        final_params = {
            "Temperature (c)": round(x[0], 1), "Pressure (torr)": round(x[1], 3),
            "Precursor Pulse Time (s)": round(x[2], 2), "Purge Time (s)": round(x[3], 1),
            "Purge Gas Flow Rate (cm3/min)": int(x[4]), "Cycles (n)": init_cycles
        }
        
        vec = self._construct_input(final_params, user_in["Precursor"], 'H2O', 'N2')
        final_pred_vals = self.mm.predict(vec)
        final_pred_dict = dict(zip(self.dm.targets, final_pred_vals))
        
        # GPC Correction
        gpc = max(0.001, final_pred_dict['GPC (A/cycle)'])
        real_cycles = int(round((user_in["Thickness (nm)"] * 10) / gpc))
        final_params['Cycles (n)'] = real_cycles
        final_pred_dict['Thickness (nm)'] = (gpc * real_cycles) / 10.0
        
        # Physics Check
        phys_sc = ALDPhysics.calc_sc(x[1], x[0], x[2], user_in["Target AR"], user_in["Precursor"], user_in["CD (nm)"])
        
        return final_params, final_pred_dict, {"Physics SC (%)": f"{phys_sc:.2f}%", "Cost": f"{res.fun:.4f}"}

    def simulate(self, user_in, target, sweep_vals):
        rows = []
        for val in sweep_vals:
            temp_in = user_in.copy(); temp_in[target] = val
            rec, pred, val_data = self.optimize(temp_in)
            
            row = {target: val}
            row.update({k:v for k,v in rec.items() if isinstance(v, (int, float))})
            row.update(pred)
            row["Physics SC (%)"] = float(val_data["Physics SC (%)"].replace('%', ''))
            rows.append(row)
        return pd.DataFrame(rows)


# ==========================================
# 5. Main Execution Logic
# ==========================================
def main_cli():
    print("\n" + "="*60 + "\n  [CLI] Enterprise AI ALD Optimizer (XGBoost + Feature Eng.)\n" + "="*60)
    
    csv_file = "AI_ALD1.csv"
    if not os.path.exists(csv_file): print(f"❌ Error: {csv_file} not found."); return
    
    dm = ALDDataManager(csv_file, "cli")
    mm = ALDXGBoostModel(dm, "cli")
    opt = ALDOptimizer(dm, mm, "cli")
    
    print(f"\n📊 Model Performance (R2): {mm.metrics['R2']:.4f}")

    try:
        p_list = ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"]
        print("\n[Select Precursor]: " + ", ".join([f"{i+1}.{p}" for i, p in enumerate(p_list)]))
        p_idx = int(input("=> Input Number: ")) - 1
        sel_p = p_list[p_idx]
        th = float(input("=> Target Thickness (nm): "))
        ar = float(input("=> Target AR: "))
        cd = float(input("=> CD (nm): "))
    except: print("Input Error"); return

    u_in = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
    rec, pred, valid = opt.optimize(u_in)
    
    print("\n[💡 Optimized Recipe]\n", pd.Series(rec).to_string())
    print("\n[📈 Prediction]\n", pd.Series(pred).to_string())
    
    # Visualization
    print("\n📊 Generating Charts...")
    sweep_x = np.linspace(th*0.5, th*1.5, 10)
    df = opt.simulate(u_in, "Thickness (nm)", sweep_x)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    l1 = plt.plot(df["Thickness (nm)"], df["Temperature (c)"], 'r-o', label="Temp")[0]
    plt.xlabel("Target Thickness"); plt.ylabel("Temp (c)", color='r')
    plt.twinx().plot(df["Thickness (nm)"], df["GPC (A/cycle)"], 'b-s', label="GPC")
    plt.title("Optimization Trend")
    
    plt.subplot(1, 2, 2)
    plt.plot(df["Thickness (nm)"], df["Step Coverage (sc, %)"], 'g-^', label="AI SC")
    plt.plot(df["Thickness (nm)"], df["Physics SC (%)"], 'k--', label="Phys SC")
    plt.legend(); plt.title("SC Reliability")
    
    plt.tight_layout()
    try: plt.show()
    except: print("Plot failed, saved to result.png"); plt.savefig("result.png")


def main_gui():
    st.set_page_config(page_title="Enterprise ALD Optimizer", layout="wide")
    st.title("🚀 AI ALD Optimizer (Enterprise Edition)")

    @st.cache_resource
    def get_system():
        path = os.path.join(os.path.dirname(__file__), "AI_ALD1.csv")
        if not os.path.exists(path): path = "AI_ALD1.csv"
        dm = ALDDataManager(path, "gui")
        mm = ALDXGBoostModel(dm, "gui")
        return ALDOptimizer(dm, mm, "gui"), mm

    try: opt, mm = get_system()
    except Exception as e: st.error(f"System Error: {e}"); st.stop()

    st.sidebar.header("Configuration")
    sel_p = st.sidebar.selectbox("Precursor", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("Target Thickness (nm)", 1.0, 500.0, 15.0)
    ar = st.sidebar.number_input("Target AR", 1.0, 100.0, 10.0)
    cd = st.sidebar.number_input("CD (nm)", 1.0, 1000.0, 100.0)
    
    if 'done' not in st.session_state: st.session_state.done = False

    if st.sidebar.button("Optimize Process", type="primary"):
        u_in = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("Running AI Optimization..."):
            st.session_state.res = opt.optimize(u_in)
            st.session_state.u_in = u_in
            st.session_state.done = True

    if st.session_state.done:
        rec, pred, val = st.session_state.res
        
        tab1, tab2, tab3 = st.tabs(["Report", "Simulation", "Feature Analysis"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1: st.subheader("Recipe"); st.dataframe(pd.DataFrame([rec]).T)
            with c2: st.subheader("Prediction"); st.dataframe(pd.DataFrame([pred]).T); st.info(f"Phys SC: {val['Physics SC (%)']}")
            st.divider(); st.caption(f"Model R2: {mm.metrics['R2']:.4f} | Best Params: {mm.best_params}")

        with tab2:
            st.subheader("Target Sweep Simulation")
            c1, c2, c3 = st.columns(3)
            tgt = c1.selectbox("Target (X)", ["Thickness (nm)", "Target AR"])
            y1 = c2.selectbox("Left Axis (Y1)", ["Temperature (c)", "Pressure (torr)", "Cycles (n)"])
            y2 = c3.selectbox("Right Axis (Y2)", ["GPC (A/cycle)", "Step Coverage (sc, %)"])
            
            if st.button("Run Simulation"):
                curr = st.session_state.u_in[tgt]
                rng = np.linspace(curr*0.5, curr*1.5, 10)
                df = opt.simulate(st.session_state.u_in, tgt, rng)
                
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df[tgt], df[y1], 'r-o', label=y1)
                ax1.set_ylabel(y1, color='r')
                ax2 = ax1.twinx()
                ax2.plot(df[tgt], df[y2], 'b-s', label=y2)
                ax2.set_ylabel(y2, color='b')
                st.pyplot(fig)
        
        with tab3:
            st.subheader("🔍 Feature Importance (XGBoost)")
            st.info("Which process parameters affect the result most?")
            names, imps = mm.get_feature_importance()
            fig_f, ax_f = plt.subplots(figsize=(8, 5))
            ax_f.barh(names, imps, color='skyblue')
            ax_f.set_xlabel("Importance Score")
            ax_f.invert_yaxis()
            st.pyplot(fig_f)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(): main_gui()
        else: main_cli()
    except: main_cli()