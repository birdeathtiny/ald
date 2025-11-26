# ==============================================================================
#  [Enterprise-Grade] AI ALD Process Optimization System (End-Game Ver.)
# ==============================================================================
#  Architecture:
#  1. Config & Logger: Centralized configuration and logging mechanism.
#  2. ALDDataManager: ETL pipeline (Extract, Transform, Load) with Feature Eng.
#  3. ALDModelCore: XGBoost engine with Model Persistence (Save/Load logic).
#  4. ALDPhysics: Theoretical validation layer using Langmuir/Knudsen models.
#  5. ALDOptimizer: Inverse design engine using SLSQP constraints.
#  6. Interface: Dual-mode support (CLI & Streamlit GUI).
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import joblib  # For Model Persistence
import warnings
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
import xgboost as xgb
from scipy.optimize import minimize
from typing import Dict, Any, List, Optional

# Suppress minor warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. Configuration & Logging
# ==========================================
class Config:
    APP_NAME = "Enterprise ALD Optimizer"
    VERSION = "v33.0.0"
    MODEL_FILE = "ald_xgboost_model.pkl"
    SCALER_X_FILE = "ald_scaler_x.pkl"
    SCALER_Y_FILE = "ald_scaler_y.pkl"
    
    PRECURSOR_CONSTANTS = {
        "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005},
        "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001},
        "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005},
        "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008}
    }
    N_A = 6.022e23
    k_B = 1.38e-23

class Logger:
    def __init__(self, mode="cli"):
        self.mode = mode

    def info(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] ℹ️ {msg}"
        if self.mode == "cli": print(log_msg)
        elif self.mode == "gui": st.toast(msg)

    def error(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] ❌ {msg}"
        if self.mode == "cli": print(log_msg); sys.exit(1)
        else: st.error(msg); st.stop()

    def warning(self, msg):
        if self.mode == "cli": print(f"⚠️ {msg}")
        else: st.warning(msg)

# ==========================================
# 1. Data Management Layer
# ==========================================
class ALDDataManager:
    def __init__(self, file_path: str, logger: Logger):
        self.logger = logger
        self.file_path = file_path
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        
        self._load_data()
        self._preprocess_pipeline()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            # Fallback search
            curr = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(curr, os.path.basename(self.file_path))
            if os.path.exists(alt): self.file_path = alt
            else: self.logger.error(f"Data file not found: {self.file_path}")
            
        try: 
            self.df = pd.read_csv(self.file_path, encoding='CP949')
        except: 
            self.df = pd.read_csv(self.file_path)
        
        self.logger.info(f"Data Loaded: {len(self.df)} rows found.")

    def _preprocess_pipeline(self):
        # 1. Cleaning
        self.df.replace('-', np.nan, inplace=True)
        
        numeric_cols = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)',
            'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Thickness (nm)', 
            'Surface Roughness (RMS, nm)', 'Uniformity (%)', 'Step Coverage (sc, %)', 
            'Density (g/cm3)', 'GPC (A/cycle)', 'Aspect Ratio (AR)', 
            'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)'
        ]
        for c in numeric_cols:
            if c in self.df.columns: self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # 2. Categorical Processing
        if 'Co-reactant' in self.df.columns:
            self.df['Co-reactant'] = self.df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
            
        # 3. Feature Selection
        self.target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)', 'Step Coverage (sc, %)'
        ]
        self.ignore_cols = ['Aspect Ratio (AR)', 'Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', '순서']
        
        # Safe Drop
        self.df.drop(columns=[c for c in self.ignore_cols if c in self.df.columns], inplace=True)
        
        # One-Hot Encoding
        cat_cols = [c for c in ['Precursor', 'Co-reactant', 'Purge Gas'] if c in self.df.columns]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, dummy_na=False)

        # Final Feature Sets
        self.targets = [c for c in self.target_cols if c in self.df_encoded.columns]
        self.inputs = self.df_encoded.drop(columns=self.targets, errors='ignore').columns.tolist()

        X_raw = self.df_encoded[self.inputs].values
        Y_raw = self.df_encoded[self.targets].values
        
        # 4. Advanced Feature Engineering
        # Imputation
        imp = KNNImputer(n_neighbors=5)
        X_imp = imp.fit_transform(X_raw)
        Y_imp = imp.fit_transform(Y_raw)
        
        # Polynomial Features
        X_poly = self.poly.fit_transform(X_imp)
        self.feature_names = self.poly.get_feature_names_out(self.inputs)

        # 5. Splitting
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_poly, Y_imp, test_size=0.2, random_state=42
        )
        self.Y_test_raw = self.Y_scaler.inverse_transform(self.Y_scaler.fit_transform(self.Y_test)) # Hack for scaler fitting
        
        # Scale Fitting
        self.X_train_s = self.X_scaler.fit_transform(self.X_train)
        self.X_test_s = self.X_scaler.transform(self.X_test)
        self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)


# ==========================================
# 2. Model Management Layer (Persistence)
# ==========================================
class ALDXGBoostModel:
    def __init__(self, dm: ALDDataManager, logger: Logger, force_retrain=False):
        self.dm = dm
        self.logger = logger
        self.model = None
        self.metrics = {}
        
        if force_retrain or not os.path.exists(Config.MODEL_FILE):
            self._train_new_model()
        else:
            self._load_model()

    def _train_new_model(self):
        self.logger.info("🧠 Starting New Model Training (Smart Tuning)...")
        
        # Smart Tuning Logic (thickness based)
        xgb_base = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        param_dist = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.03, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        }
        
        search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_dist,
            n_iter=5, cv=2, verbose=0, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Tune on first target
        search.fit(self.dm.X_train_s, self.dm.Y_train_s[:, 0])
        best_params = search.best_params_
        self.logger.info(f"Optimized Hyperparams: {best_params}")
        
        # Train Final Model
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42))
        self.model.fit(self.dm.X_train_s, self.dm.Y_train_s)
        
        # Evaluate
        self._evaluate()
        
        # Save Model
        self._save_model()

    def _evaluate(self):
        pred_s = self.model.predict(self.dm.X_test_s)
        pred = self.dm.Y_scaler.inverse_transform(pred_s)
        r2 = r2_score(self.dm.Y_test_raw, pred)
        self.metrics['R2'] = r2
        self.logger.info(f"Model Evaluation Completed. R2: {r2:.4f}")

    def _save_model(self):
        try:
            joblib.dump(self.model, Config.MODEL_FILE)
            joblib.dump(self.dm.X_scaler, Config.SCALER_X_FILE)
            joblib.dump(self.dm.Y_scaler, Config.SCALER_Y_FILE)
            self.logger.info("💾 Model Saved Successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to save model: {e}")

    def _load_model(self):
        try:
            self.logger.info("📂 Loading pre-trained model...")
            self.model = joblib.load(Config.MODEL_FILE)
            # Overwrite data manager scalers with trained ones to ensure consistency
            self.dm.X_scaler = joblib.load(Config.SCALER_X_FILE)
            self.dm.Y_scaler = joblib.load(Config.SCALER_Y_FILE)
            self._evaluate()
        except Exception as e:
            self.logger.warning(f"Load failed ({e}). Retraining...")
            self._train_new_model()

    def predict(self, input_vector):
        x_poly = self.dm.poly.transform(input_vector)
        x_s = self.dm.X_scaler.transform(x_poly)
        y_s = self.model.predict(x_s)
        return self.dm.Y_scaler.inverse_transform(y_s)[0]

    def get_feature_importance(self):
        try:
            # Average importance across all estimators
            imps = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            idxs = np.argsort(imps)[::-1][:10]
            return [self.dm.feature_names[i] for i in idxs], imps[idxs]
        except: return [], []


# ==========================================
# 3. Physics Engine Layer
# ==========================================
class ALDPhysics:
    @staticmethod
    def calculate_sc(P, T, Pulse, AR, Precursor, CD_nm):
        """
        Calculates theoretical Step Coverage based on Langmuir & Knudsen diffusion models.
        """
        try:
            const = Config.PRECURSOR_CONSTANTS.get(Precursor, Config.PRECURSOR_CONSTANTS["TMA"])
            T_K = T + 273.15
            P_Pa = P * 133.322
            L = AR * (CD_nm * 1e-9)
            d = const["diameter_m"]
            m = const["mass_g_mol"] / 1000 / Config.N_A
            
            v_avg = np.sqrt(8 * Config.k_B * T_K / (np.pi * m))
            lambda_m = (Config.k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
            
            D_Kn = (1.0/3.0) * v_avg * (CD_nm * 1e-9)
            D_bulk = (1.0/3.0) * lambda_m * v_avg
            D_eff = 1.0 / (1.0/D_Kn + 1.0/D_bulk)
            
            lambda_pen = np.sqrt(D_eff * Pulse + 1e-15)
            phi = L / (lambda_pen + 1e-15)
            
            if phi < 1.0: return 100.0 / (1.0 + phi) # Reaction rate limited
            else: return np.exp(-phi) * 100.0        # Diffusion limited
        except: return 0.0


# ==========================================
# 4. Optimization Engine Layer
# ==========================================
class ALDOptimizer:
    def __init__(self, dm: ALDDataManager, mm: ALDXGBoostModel):
        self.dm = dm
        self.mm = mm

    def _build_input_vector(self, params, user_in):
        row = pd.DataFrame(0.0, index=[0], columns=self.dm.inputs)
        for k, v in params.items():
            if k in row.columns: row.at[0, k] = v
        
        # One-Hot Encoding Injection
        p_col = f"Precursor_{user_in['Precursor']}"
        if p_col in row.columns: row.at[0, p_col] = 1.0
        # Defaulting to H2O/N2 for simplicity in optimizer
        if "Co-reactant_H2O" in row.columns: row.at[0, "Co-reactant_H2O"] = 1.0
        if "Purge Gas_N2" in row.columns: row.at[0, "Purge Gas_N2"] = 1.0
        
        return row.values

    def optimize_recipe(self, user_in):
        init_cycles = max(10, int(user_in["Thickness (nm)"] * 10))
        
        def objective(x):
            # x: [Temp, Pressure, Pulse, PurgeT, PurgeF]
            p = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
                "Cycles (n)": init_cycles, "Co-reactant_Pulse Time (s)": x[2]
            }
            try:
                vec = self._build_input_vector(p, user_in)
                pred = self.mm.predict(vec)
                res = dict(zip(self.dm.targets, pred))
                
                # Cost: GPC Target Error + Surface Roughness
                target_gpc = (user_in["Thickness (nm)"] * 10) / init_cycles
                cost_gpc = (res.get('GPC (A/cycle)', 0.1) - target_gpc)**2
                cost_rough = res.get('Surface Roughness (RMS, nm)', 1.0)**2
                
                return 10000 * cost_gpc + 10 * cost_rough
            except: return 1e9

        # Optimized Bounds
        bounds = [(150, 450), (0.01, 2.0), (0.05, 3.0), (1.0, 20.0), (50, 1000)]
        x0 = [300, 0.5, 0.1, 5.0, 200]
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 30})
        
        x = res.x
        opt_params = {
            "Temperature (c)": round(x[0], 1), "Pressure (torr)": round(x[1], 3),
            "Pulse (s)": round(x[2], 2), "Purge (s)": round(x[3], 1), "Flow (sccm)": int(x[4])
        }
        
        # Recalculate Cycles & Prediction
        vec = self._build_input_vector(opt_params, user_in)
        final_pred = dict(zip(self.dm.targets, self.mm.predict(vec)))
        
        gpc = max(0.001, final_pred.get('GPC (A/cycle)', 0.1))
        final_cycles = int(round((user_in["Thickness (nm)"] * 10) / gpc))
        opt_params['Cycles (n)'] = final_cycles
        final_pred['Thickness (nm)'] = (gpc * final_cycles) / 10.0
        
        # Physics Validation
        phys_sc = ALDPhysics.calculate_sc(x[1], x[0], x[2], user_in["Target AR"], user_in["Precursor"], user_in["CD (nm)"])
        
        return opt_params, final_pred, {"Physics SC (%)": f"{phys_sc:.2f}%", "Cost": f"{res.fun:.4f}"}

    def run_simulation(self, user_in, target_col, sweep_range):
        data = []
        for val in sweep_range:
            u = user_in.copy(); u[target_col] = val
            r, p, v = self.optimize_recipe(u)
            
            row = {target_col: val}
            row.update(r); row.update(p)
            row['Phys SC'] = float(v['Physics SC (%)'].replace('%',''))
            data.append(row)
        return pd.DataFrame(data)


# ==============================================================================
#  5. User Interface Layer (CLI / GUI)
# ==============================================================================
def main_cli():
    print("\n" + "="*70)
    print(f"  🚀 {Config.APP_NAME} {Config.VERSION} (CLI Mode)")
    print("="*70)
    
    logger = Logger("cli")
    csv_file = "AI_ALD1.csv"
    
    dm = ALDDataManager(csv_file, "cli")
    mm = ALDXGBoostModel(dm, logger) # Will load or train model
    opt = ALDOptimizer(dm, mm, "cli")
    
    print("-" * 70)
    try:
        th = float(input(">> Enter Target Thickness (nm): "))
        ar = float(input(">> Enter Target AR: "))
        cd = float(input(">> Enter CD (nm): "))
    except: logger.error("Invalid Input"); return

    u_in = {"Precursor": "TMA", "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
    rec, pred, val = opt.optimize_recipe(u_in)
    
    print("\n[💡 Optimized Recipe]\n", pd.Series(rec).to_string())
    print("\n[📈 AI Prediction]\n", pd.Series(pred).to_string())
    print(f"\n[🔬 Physics Validation]: {val}")
    
    print("\n📊 Generating Simulation Charts...")
    sweep_x = np.linspace(th*0.5, th*1.5, 10)
    df = opt.run_simulation(u_in, "Thickness (nm)", sweep_x)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df["Thickness (nm)"], df["Temperature (c)"], 'r-o', label="Temp")
    plt.xlabel("Target Thickness"); plt.ylabel("Temp (c)", color='r')
    plt.twinx().plot(df["Thickness (nm)"], df["GPC (A/cycle)"], 'b-s', label="GPC")
    plt.title("Process Window Trend")
    
    plt.subplot(1, 2, 2)
    plt.plot(df["Thickness (nm)"], df["Step Coverage (sc, %)"], 'g-^', label="AI SC")
    plt.plot(df["Thickness (nm)"], df["Phys SC"], 'k--', label="Phys SC")
    plt.legend(); plt.title("SC Reliability Check")
    
    plt.tight_layout()
    try: plt.show()
    except: print("Plot failed. Saved to result.png"); plt.savefig("result.png")

def main_gui():
    st.set_page_config(page_title=Config.APP_NAME, layout="wide")
    st.title(f"🚀 {Config.APP_NAME}")
    
    # Session State for Persistence
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False

    @st.cache_resource
    def get_system():
        logger = Logger("gui")
        path = "AI_ALD1.csv"
        if not os.path.exists(path): path = os.path.join(os.path.dirname(__file__), "AI_ALD1.csv")
        
        dm = ALDDataManager(path, "gui")
        mm = ALDXGBoostModel(dm, logger)
        opt = ALDOptimizer(dm, mm, "gui")
        return opt, mm

    try: 
        opt, mm = get_system()
        st.session_state.system_ready = True
    except Exception as e: st.error(f"System Init Failed: {e}"); st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Process Targets")
    p = st.sidebar.selectbox("Precursor", ["TMA", "TDMAH", "TEMAHf", "Zr(NEt2)4"])
    th = st.sidebar.number_input("Thickness (nm)", 1.0, 1000.0, 15.0)
    ar = st.sidebar.number_input("Aspect Ratio (AR)", 1.0, 200.0, 10.0)
    cd = st.sidebar.number_input("Critical Dimension (nm)", 5.0, 5000.0, 100.0)
    
    # Force Retrain Option
    if st.sidebar.button("🔄 Force Retrain Model"):
        if os.path.exists(Config.MODEL_FILE): os.remove(Config.MODEL_FILE)
        st.cache_resource.clear()
        st.rerun()

    if 'done' not in st.session_state: st.session_state.done = False

    if st.sidebar.button("🔥 Run Optimization", type="primary"):
        u_in = {"Precursor": p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("Searching Optimal Recipe (Inverse Design)..."):
            st.session_state.res = opt.optimize_recipe(u_in)
            st.session_state.u_in = u_in
            st.session_state.done = True

    # Results View
    if st.session_state.done:
        rec, pred, val = st.session_state.res
        
        tab1, tab2, tab3 = st.tabs(["📄 Engineering Report", "📊 Sensitivity Analysis", "🧠 AI Insights"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1: st.subheader("✅ Optimized Recipe"); st.dataframe(pd.DataFrame([rec]).T, use_container_width=True)
            with c2: st.subheader("📈 Predicted Properties"); st.dataframe(pd.DataFrame([pred]).T, use_container_width=True)
            st.success(f"Physics Validation: Step Coverage = {val['Physics SC (%)']}")
        
        with tab2:
            st.subheader("Parameter Sweep Simulation")
            c1, c2, c3 = st.columns(3)
            tgt = c1.selectbox("Sweep Target (X-Axis)", ["Thickness (nm)", "Target AR"])
            y1 = c2.selectbox("Recipe Parameter (Left Y)", ["Temperature (c)", "Pressure (torr)", "Cycles (n)", "Pulse (s)"])
            y2 = c3.selectbox("Property Result (Right Y)", ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)"])
            
            if st.button("Run Simulation"):
                curr = st.session_state.u_in[tgt]
                rng = np.linspace(curr*0.5, curr*1.5, 10)
                df = opt.run_simulation(st.session_state.u_in, tgt, rng)
                
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df[tgt], df[y1], 'r-o', label=f"Recipe: {y1}")
                ax1.set_ylabel(y1, color='r'); ax1.tick_params(axis='y', labelcolor='r')
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                ax2 = ax1.twinx()
                ax2.plot(df[tgt], df[y2], 'b-s', label=f"Pred: {y2}")
                ax2.set_ylabel(y2, color='b'); ax2.tick_params(axis='y', labelcolor='b')
                
                lines = ax1.get_lines() + ax2.get_lines()
                ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')
                st.pyplot(fig)

        with tab3:
            st.subheader("Feature Importance (XGBoost)")
            names, imps = mm.get_feature_importance()
            if len(names) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(names, imps, color="#4A90E2")
                ax.invert_yaxis()
                ax.set_xlabel("Relative Importance Score")
                st.pyplot(fig)
            else:
                st.warning("Feature importance not available for this model state.")

if __name__ == "__main__":
    if "streamlit" in sys.modules: main_gui()
    else: main_cli()