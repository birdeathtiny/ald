# ==============================================================================
#  [Enterprise-Grade] AI ALD Process Optimization System (Pure Learning Edition)
# ==============================================================================
#  Architecture Overview:
#  1. System Core: Centralized Config & Advanced Logging System
#  2. Data Layer: Robust ETL, Smart Imputation, Polynomial Feature Engineering
#  3. Model Layer: Pure XGBoost Learning Engine (No Pre-fixed Params, Real-time Fit)
#  4. Physics Layer: Theoretical Step Coverage Validation (Langmuir-Knudsen)
#  5. Solver Layer: Inverse Design using Constrained Optimization (SLSQP)
#  6. Interface Layer: Dual-Mode Support (CLI Terminal & Streamlit Web GUI)
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
import matplotlib
from datetime import datetime

# Configure Backend for Headless Environments
if "streamlit" not in sys.modules:
    try: matplotlib.use('TkAgg')
    except: pass
import matplotlib.pyplot as plt

# Machine Learning & Optimization Libraries
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from scipy.optimize import minimize
from typing import Dict, Any, List, Optional, Tuple

# Suppress Warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
#  0. System Configuration & Logging
# ==============================================================================
class Config:
    APP_NAME = "Enterprise ALD Optimizer"
    VERSION = "v42.0 (Pure Learning)"
    DATA_FILE = "AI_ALD1.csv"
    
    # Physics Constants
    PRECURSOR_CONSTANTS = {
        "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005},
        "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001},
        "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005},
        "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008}
    }
    N_A = 6.022e23
    k_B = 1.38e-23

class Logger:
    """Centralized Logging System for Process Tracking"""
    def __init__(self, mode="cli"):
        self.mode = mode

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def info(self, msg: str):
        if self.mode == "cli": print(f"[{self._timestamp()}] ℹ️  {msg}")
        elif self.mode == "gui": st.toast(f"ℹ️ {msg}")

    def success(self, msg: str):
        if self.mode == "cli": print(f"[{self._timestamp()}] ✅ {msg}")
        elif self.mode == "gui": st.success(msg)

    def warning(self, msg: str):
        if self.mode == "cli": print(f"[{self._timestamp()}] ⚠️ {msg}")
        else: st.warning(msg)

    def error(self, msg: str):
        if self.mode == "cli": print(f"[{self._timestamp()}] ❌ {msg}"); sys.exit(1)
        else: st.error(msg); st.stop()


# ==============================================================================
#  1. Data Management Layer (ETL & Feature Engineering)
# ==============================================================================
class ALDDataManager:
    """
    Handles Data Extraction, Transformation, and Loading.
    Includes Feature Engineering (Polynomial Expansion) and Scaling.
    """
    def __init__(self, file_path: str, logger: Logger):
        self.logger = logger
        self.file_path = file_path
        
        # Transformers
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        # 💡 Polynomial Features: Expands feature space for non-linear learning
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        self._extract_data()
        self._transform_data()

    def _extract_data(self):
        """Robust Data Loading Logic"""
        if not os.path.exists(self.file_path):
            # Fallback: Check current directory
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(curr_dir, os.path.basename(self.file_path))
            if os.path.exists(alt_path):
                self.file_path = alt_path
            else:
                self.logger.error(f"Data file not found: {self.file_path}")
                
        try: 
            self.df = pd.read_csv(self.file_path, encoding='CP949')
        except: 
            self.df = pd.read_csv(self.file_path)
            
        self.logger.info(f"Data Successfully Loaded: {len(self.df)} records.")

    def _transform_data(self):
        """Preprocessing Pipeline: Cleaning -> Feature Selection -> Engineering -> Scaling"""
        # 1. Data Cleaning
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

        # 2. Categorical Handling
        if 'Co-reactant' in self.df.columns:
            self.df['Co-reactant'] = self.df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
            
        # 3. Feature Selection (Drop Irrelevant)
        drop_cols = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)', '순서']
        self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True, errors='ignore')
        
        # 4. One-Hot Encoding
        cat_cols = [c for c in ['Precursor', 'Co-reactant', 'Purge Gas'] if c in self.df.columns]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, dummy_na=False)

        # 5. Define Input/Output Schema
        self.target_cols = [
            'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
            'Density (g/cm3)', 'GPC (A/cycle)', 'Step Coverage (sc, %)'
        ]
        self.ignore_cols = ['Aspect Ratio (AR)', 'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)']
        
        self.targets = [c for c in self.target_cols if c in self.df_encoded.columns]
        self.inputs = self.df_encoded.drop(columns=self.targets + [c for c in self.ignore_cols if c in self.df_encoded.columns], errors='ignore').columns.tolist()

        X_raw = self.df_encoded[self.inputs].values
        Y_raw = self.df_encoded[self.targets].values
        
        # 6. Advanced Feature Engineering
        # Imputation (KNN)
        imp = KNNImputer(n_neighbors=5)
        X_imp = imp.fit_transform(X_raw)
        Y_imp = imp.fit_transform(Y_raw)
        
        # Polynomial Expansion (Interaction Terms)
        X_poly = self.poly.fit_transform(X_imp)
        self.feature_names = self.poly.get_feature_names_out(self.inputs)

        # 7. Scaling & Splitting
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_scaler.fit_transform(X_poly), 
            self.Y_scaler.fit_transform(Y_imp), 
            test_size=0.2, random_state=42
        )
        self.Y_test_raw = self.Y_scaler.inverse_transform(self.Y_test)
        
        if self.logger.mode == "cli":
            print(f"   -> Preprocessing: {X_raw.shape[1]} base features -> {X_poly.shape[1]} poly-features.")


# ==============================================================================
#  Class 2: Model Core Layer (Pure Learning Engine)
# ==============================================================================
class ALDXGBoostModel:
    """
    Manages the AI Model Lifecycle.
    Performs 'Pure Learning': No pre-fixed best parameters. 
    The model learns fresh from the data every time it runs.
    """
    def __init__(self, dm: ALDDataManager, logger: Logger):
        self.dm = dm
        self.logger = logger
        self.model = None
        self.metrics = {}
        self._train_pure_learning()

    def _train_pure_learning(self):
        """
        [Pure Learning Mode]
        Trains the model from scratch using robust default settings suitable for general regression.
        This ensures the model adapts to the current dataset without relying on hardcoded 'magic numbers'.
        """
        self.logger.info("🤖 Initializing AI Engine & Starting Real-Time Learning...")
        
        # 💡 Robust General Parameters (Not optimized for specific data, but good for learning)
        xgb_params = {
            'n_estimators': 500,       # Enough trees to learn patterns
            'learning_rate': 0.05,     # Standard learning rate
            'max_depth': 6,            # Standard depth for tabular data
            'subsample': 0.8,          # Prevent overfitting
            'colsample_bytree': 0.8,   # Feature randomness
            'n_jobs': -1,              # Use all cores
            'random_state': 42
        }
        
        # Train Multi-Output Regressor
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        self.model.fit(self.dm.X_train, self.dm.Y_train)
        
        # Evaluate Performance
        pred_s = self.model.predict(self.dm.X_test)
        pred = self.dm.Y_scaler.inverse_transform(pred_s)
        
        r2 = r2_score(self.dm.Y_test_raw, pred)
        rmse = np.sqrt(mean_squared_error(self.dm.Y_test_raw, pred))
        self.metrics = {'R2': r2, 'RMSE': rmse}
        
        self.logger.success(f"AI Training Complete. Accuracy (R2): {r2:.4f}")

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        """Predicts outputs for a given input vector (w/ Poly Features)"""
        x_p = self.dm.poly.transform(input_vector)
        x_s = self.dm.X_scaler.transform(x_p)
        y_s = self.model.predict(x_s)
        return self.dm.Y_scaler.inverse_transform(y_s)[0]

    def get_feature_importance(self) -> Tuple[List[str], List[float]]:
        """Extracts global feature importance"""
        try:
            # Average importance across all target estimators
            imps = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            idxs = np.argsort(imps)[::-1][:10] # Top 10
            return [self.dm.feature_names[i] for i in idxs], imps[idxs]
        except: return [], []


# ==============================================================================
#  Class 3: Physics Engine Layer
# ==============================================================================
class ALDPhysics:
    """Theoretical Validation Layer based on Physical Principles"""
    @staticmethod
    def calculate_sc(P, T, Pulse, AR, Precursor, CD_nm):
        try:
            const = Config.PRECURSOR_CONSTANTS.get(Precursor, Config.PRECURSOR_CONSTANTS["TMA"])
            T_K = T + 273.15
            P_Pa = P * 133.322
            L = AR * (CD_nm * 1e-9)
            
            # Kinetic Theory
            m = const["mass_g_mol"] / 1000.0 / Config.N_A
            v_avg = np.sqrt(8 * Config.k_B * T_K / (np.pi * m))
            
            d = const["diameter_m"]
            lam = (Config.k_B * T_K) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
            
            # Diffusion Coefficients
            D_Kn = (1.0/3.0) * v_avg * (CD_nm * 1e-9)
            D_bulk = (1.0/3.0) * lam * v_avg
            D_eff = 1.0 / (1.0/D_Kn + 1.0/D_bulk)
            
            # Penetration Depth & Thiele Modulus
            penetration = np.sqrt(D_eff * Pulse + 1e-15)
            phi = L / (penetration + 1e-15)
            
            # Step Coverage Calculation
            if phi < 1.0: return 100.0 / (1.0 + phi) # Reaction Limited
            else: return np.exp(-phi) * 100.0        # Diffusion Limited
        except: return 0.0


# ==============================================================================
#  Class 4: Optimization Engine (Inverse Design)
# ==============================================================================
class ALDOptimizer:
    """Handles Inverse Design Optimization using Scipy"""
    def __init__(self, dm: ALDDataManager, mm: ALDXGBoostModel):
        self.dm = dm
        self.mm = mm

    def _construct_input_vector(self, params: Dict, user_in: Dict):
        """Builds the input vector matching training data structure"""
        row = pd.DataFrame(0.0, index=[0], columns=self.dm.inputs)
        for k, v in params.items():
            if k in row.columns: row.at[0, k] = v
        
        # One-Hot Encoding Injection
        p_col = f"Precursor_{user_in['Precursor']}"
        if p_col in row.columns: row.at[0, p_col] = 1.0
        # Defaults
        if "Co-reactant_H2O" in row.columns: row.at[0, "Co-reactant_H2O"] = 1.0
        if "Purge Gas_N2" in row.columns: row.at[0, "Purge Gas_N2"] = 1.0
        
        return row.values

    def optimize_recipe(self, user_in: Dict) -> Tuple[Dict, Dict, Dict]:
        """Finds optimal process parameters"""
        
        init_cycles = max(10, int(user_in["Thickness (nm)"] * 10)) 
        
        def objective(x):
            # x: [Temp, Pressure, Pulse, PurgeT, PurgeF]
            params = {
                "Temperature (c)": x[0], "Pressure (torr)": x[1], "Precursor_Pulse Time (s)": x[2],
                "Purge Time (s)": x[3], "Purge Gas Flow Rate (cm3/min)": x[4],
                "Cycles (n)": init_cycles, "Co-reactant_Pulse Time (s)": x[2]
            }
            try:
                vec = self._construct_input_vector(params, user_in)
                pred = self.mm.predict(vec)
                res = dict(zip(self.dm.targets, pred))
                
                target_gpc = (user_in["Thickness (nm)"] * 10) / init_cycles
                gpc_loss = (res.get('GPC (A/cycle)', 0.1) - target_gpc)**2
                rough_loss = res.get('Surface Roughness (RMS, nm)', 1.0)**2
                
                return 10000.0 * gpc_loss + 10.0 * rough_loss
            except: return 1e9

        # Search Bounds
        bounds = [(150, 450), (0.01, 3.0), (0.05, 5.0), (1.0, 60.0), (50, 1000)]
        x0 = [300, 0.5, 0.1, 5.0, 200]
        
        # SLSQP Optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 30})
        
        x = res.x
        opt_params = {
            "Temperature (c)": round(x[0], 1), "Pressure (torr)": round(x[1], 3),
            "Pulse (s)": round(x[2], 2), "Purge (s)": round(x[3], 1), "Flow (sccm)": int(x[4])
        }
        
        # Recalculate with Optimized Params
        vec = self._construct_input_vector(opt_params, user_in)
        final_pred_vals = self.mm.predict(vec)
        final_pred_dict = dict(zip(self.dm.targets, final_pred_vals))
        
        # Cycle Correction
        gpc = max(0.001, final_pred_dict.get('GPC (A/cycle)', 0.1))
        final_cycles = int(round((user_in["Thickness (nm)"] * 10) / gpc))
        
        opt_params['Cycles (n)'] = final_cycles
        final_pred_dict['Thickness (nm)'] = (gpc * final_cycles) / 10.0
        
        # Physics Validation
        phys_sc = ALDPhysics.calculate_sc(x[1], x[0], x[2], user_in["Target AR"], user_in["Precursor"], user_in["CD (nm)"])
        
        return opt_params, final_pred_dict, {"Physics SC (%)": f"{phys_sc:.2f}%", "Cost": f"{res.fun:.4f}"}

    def run_simulation(self, user_in, target_col, sweep_range):
        """Simulates process trends"""
        data = []
        for val in sweep_range:
            u_temp = user_in.copy()
            u_temp[target_col] = val
            rec, pred, val_data = self.optimize_recipe(u_temp)
            
            row = {target_col: val}
            row.update(rec)
            row.update(pred)
            row['Phys SC'] = float(val_data['Physics SC (%)'].replace('%',''))
            data.append(row)
        return pd.DataFrame(data)


# ==============================================================================
#  5. User Interface Layer (CLI & GUI)
# ==============================================================================
def main_cli():
    print("\n" + "="*70)
    print(f"  🚀 {Config.APP_NAME} (CLI Mode)")
    print("="*70)
    
    logger = Logger("cli")
    csv_file = Config.DATA_FILE
    
    # System Init
    dm = ALDDataManager(csv_file, logger)
    mm = ALDXGBoostModel(dm, logger)
    opt = ALDOptimizer(dm, mm)
    
    print("-" * 70)
    try:
        th = float(input(">> Enter Target Thickness (nm): "))
        ar = float(input(">> Enter Target Aspect Ratio (AR): "))
        cd = float(input(">> Enter Critical Dimension (CD, nm): "))
    except: logger.error("Invalid Input Format"); return

    # Precursor Select
    precursors = list(Config.PRECURSOR_CONSTANTS.keys())
    print(f"\nSelect Precursor: {precursors}")
    try:
        p_idx = int(input(f">> Select (1-{len(precursors)}): ")) - 1
        sel_p = precursors[p_idx]
    except: sel_p = precursors[0]

    u_in = {"Precursor": sel_p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
    
    # Optimization
    rec, pred, val = opt.optimize_recipe(u_in)
    
    print("\n" + "="*30 + " OPTIMIZATION RESULTS " + "="*30)
    print(f"\n[💡 Optimized Recipe]\n{pd.Series(rec).to_string()}")
    print(f"\n[📈 AI Prediction]\n{pd.Series(pred).to_string()}")
    print(f"\n[🔬 Validation]\n{val}")
    
    # Simulation
    print("\n📊 Generating Simulation Charts...")
    sweep_x = np.linspace(th*0.5, th*1.5, 10)
    df = opt.run_simulation(u_in, "Thickness (nm)", sweep_x)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    

[Image of Matplotlib Line Chart]

    plt.plot(df["Thickness (nm)"], df["Temperature (c)"], 'r-o', label="Temp")
    plt.xlabel("Target Thickness"); plt.ylabel("Temp (c)", color='r')
    plt.twinx().plot(df["Thickness (nm)"], df["GPC (A/cycle)"], 'b-s', label="GPC")
    plt.title("Process Trend")
    
    plt.subplot(1, 2, 2)
    

[Image of Matplotlib Line Chart]

    plt.plot(df["Thickness (nm)"], df["Step Coverage (sc, %)"], 'g-^', label="AI Prediction")
    plt.plot(df["Thickness (nm)"], df["Phys SC"], 'k--', label="Physics Model")
    plt.xlabel("Target Thickness"); plt.ylabel("Step Coverage (%)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.title("AI vs Physics Validation")
    
    plt.tight_layout()
    try: plt.show()
    except: print("Plot display failed. Saved to 'result.png'."); plt.savefig("result.png")


def main_gui():
    st.set_page_config(page_title=Config.APP_NAME, layout="wide")
    st.title(f"🚀 {Config.APP_NAME}")
    st.caption(f"Version {Config.VERSION} | Real-Time Learning Engine")

    # Initialize System (Cached for Performance)
    @st.cache_resource
    def get_system():
        logger = Logger("gui")
        path = Config.DATA_FILE
        if not os.path.exists(path): 
            path = os.path.join(os.path.dirname(__file__), Config.DATA_FILE)
        
        dm = ALDDataManager(path, logger)
        mm = ALDXGBoostModel(dm, logger)
        return ALDOptimizer(dm, mm), mm

    try: 
        opt, mm = get_system()
    except Exception as e: 
        st.error(f"System Initialization Failed: {e}")
        st.stop()

    # Sidebar UI
    st.sidebar.header("⚙️ Process Targets")
    p = st.sidebar.selectbox("Precursor", list(Config.PRECURSOR_CONSTANTS.keys()))
    th = st.sidebar.number_input("Thickness (nm)", 1.0, 1000.0, 15.0)
    ar = st.sidebar.number_input("Aspect Ratio (AR)", 1.0, 200.0, 10.0)
    cd = st.sidebar.number_input("Critical Dimension (nm)", 5.0, 5000.0, 100.0)
    
    # Reset Logic
    if 'done' not in st.session_state: st.session_state.done = False

    if st.sidebar.button("🔥 Run Optimization", type="primary"):
        u_in = {"Precursor": p, "Thickness (nm)": th, "Target AR": ar, "CD (nm)": cd}
        with st.spinner("Searching Optimal Recipe..."):
            st.session_state.res = opt.optimize_recipe(u_in)
            st.session_state.u_in = u_in
            st.session_state.done = True

    # Results Dashboard
    if st.session_state.done:
        rec, pred, val = st.session_state.res
        
        tab1, tab2, tab3 = st.tabs(["📄 Engineering Report", "📊 Sensitivity Analysis", "🧠 AI Insights"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1: 
                st.subheader("✅ Optimized Recipe")
                st.dataframe(pd.DataFrame([rec]).T, use_container_width=True)
            with c2: 
                st.subheader("📈 Predicted Properties")
                st.dataframe(pd.DataFrame([pred]).T, use_container_width=True)
                st.success(f"Physics Validation (SC): {val['Physics SC (%)']}")
                st.info(f"Optimization Cost: {val['Cost']}")

        with tab2:
            st.subheader("Parameter Sweep Simulation")
            c1, c2, c3 = st.columns(3)
            tgt = c1.selectbox("Sweep Target (X-Axis)", ["Thickness (nm)", "Target AR"])
            y1 = c2.selectbox("Recipe Parameter (Left Y)", ["Temperature (c)", "Pressure (torr)", "Cycles (n)", "Pulse (s)"])
            y2 = c3.selectbox("Property Result (Right Y)", ["GPC (A/cycle)", "Step Coverage (sc, %)", "Surface Roughness (RMS, nm)"])
            
            if st.button("Run Simulation"):
                with st.spinner("Simulating..."):
                    curr = st.session_state.u_in[tgt]
                    rng = np.linspace(curr*0.5, curr*1.5, 10)
                    df = opt.run_simulation(st.session_state.u_in, tgt, rng)
                    
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    

[Image of Matplotlib Line Chart]

                    ax1.plot(df[tgt], df[y1], 'r-o', label=f"Recipe: {y1}")
                    ax1.set_ylabel(y1, color='r'); ax1.tick_params(axis='y', labelcolor='r')
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    
                    ax2 = ax1.twinx()
                    ax2.plot(df[tgt], df[y2], 'b-s', label=f"Pred: {y2}")
                    ax2.set_ylabel(y2, color='b'); ax2.tick_params(axis='y', labelcolor='b')
                    
                    lines = ax1.get_lines() + ax2.get_lines()
                    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')
                    st.pyplot(fig)
                    
                    st.divider()
                    st.subheader("⚖️ SC Validation (AI vs Physics)")
                    fig2, ax = plt.subplots(figsize=(10, 3))
                    

[Image of Matplotlib Line Chart]

                    ax.plot(df[tgt], df["Step Coverage (sc, %)"], 'g-', label="AI")
                    ax.plot(df[tgt], df["Phys SC"], 'k--', label="Physics")
                    ax.set_xlabel(tgt); ax.set_ylabel("Step Coverage (%)")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig2)

        with tab3:
            st.subheader("Feature Importance (XGBoost)")
            names, imps = mm.get_feature_importance()
            if len(names) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                

[Image of Matplotlib Bar Chart]

                ax.barh(names, imps, color="#4A90E2")
                ax.invert_yaxis()
                ax.set_xlabel("Importance Score")
                st.pyplot(fig)
            else:
                st.warning("Feature importance not available.")

if __name__ == "__main__":
    if "streamlit" in sys.modules: main_gui()
    else: main_cli()