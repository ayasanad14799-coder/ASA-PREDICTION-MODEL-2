import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import math

# =============================================================================
# 1. إعدادات الصفحة الأساسية
# =============================================================================
st.set_page_config(
    page_title="ASA-PREDICTION MODEL 2",
    page_icon="🏗️",
    layout="wide"
)

# =============================================================================
# 2. الهيدر الأكاديمي (باللوجوهات الرسمية)
# =============================================================================
def show_academic_header():
    col_left, col_mid, col_right = st.columns([1, 3, 1])
    
    with col_left:
        # لوجو الجامعة (تم تحويله للرابط المباشر Raw)
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL-2/main/LOGO.png", width=130)
        
    with col_mid:
        st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #1E3A8A; font-size: 42px; font-weight: bold; margin-bottom: 5px;'>
                    ASA-PREDICTION MODEL 2
                </h1>
                <h2 style='color: #D32F2F; font-size: 28px; font-weight: 600; margin-top: 0px; line-height: 1.3;'>
                    Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects
                </h2>
                <hr style='border: 0.5px solid #E5E7EB; width: 70%; margin: 20px auto;'>
                <span style='font-size: 20px; color: #4B5563;'>Prepared by:</span><br>
                <span style='font-size: 24px; font-weight: bold;'>Master's Researcher: Aya Mohammed Sanad Aboud</span><br><br>
                <span style='font-size: 22px; font-weight: bold; color: #4B5563;'>Under the Supervision of:</span><br>
                <span style='font-size: 24px; font-weight: 800; color: #111827;'>Prof. Ahmed Tahwia & Assoc. prof. Asser El-Sheikh</span>
            </div>
            """, unsafe_allow_html=True)
            
    with col_right:
        # لوجو الكلية (تم تحويله للرابط المباشر Raw)
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL-2/main/OIP.jfif", width=130)
        
    st.divider()

# =============================================================================
# 3. تحميل الموديلات
# =============================================================================
@st.cache_resource
def load_assets():
    try:
        models = joblib.load('concrete_model_multi.joblib')
        scaler = joblib.load('scaler_multi.joblib')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'concrete_model_multi.joblib' and 'scaler_multi.joblib' are uploaded.")
        return None, None

# =============================================================================
# 4. محرك التنبؤ والمعادلات الهندسية
# =============================================================================
def run_prediction_engine(inputs, prices):
    models, scaler = load_assets()
    if models is None or scaler is None: return None
    
    # الترتيب الصارم للمدخلات الـ 21 كما في كولاب
    feature_list = [
        inputs['Cement'], inputs['Water'], inputs['W_C'], inputs['NCA'], inputs['NFA'], 
        inputs['RCA_Weight'], inputs['RCA_P'], inputs['MRCA_P'], inputs['RFA_Weight'], 
        inputs['RFA_P'], inputs['Fly_Ash'], inputs['Silica_Fume'], inputs['Metakaolin'], 
        inputs['GGBFS'], inputs['RHA_P'], inputs['Nylon_Fiber'], inputs['Basalt_Fiber_Vol'], 
        inputs['Natural_Fiber'], inputs['SP'], inputs['Agg_Size'], inputs['Density']
    ]
    
    vector = np.array(feature_list).reshape(1, -1)
    vector_scaled = scaler.transform(vector)
    
    # 1. التنبؤات (AI)
    cs28 = models['CS_28'].predict(vector_scaled)[0]
    sts = models['STS'].predict(vector_scaled)[0]
    co2 = models['CO2'].predict(vector_scaled)[0]
    energy = models['Energy'].predict(vector_scaled)[0]
    
    # 2. المعادلات التقديرية (ACI Code)
    fs = 0.62 * math.sqrt(cs28) if cs28 > 0 else 0
    em = (4700 * math.sqrt(cs28)) / 1000 if cs28 > 0 else 0 # GPa
    cs7 = cs28 * 0.70
    cs90 = cs28 * 1.15
    
    # 3. حساب التكلفة الديناميكية
    total_cost = (
        (inputs['Cement'] * prices['Cement']) +
        (inputs['Water'] * prices['Water']) +
        (inputs['NCA'] * prices['NCA']) +
        (inputs['NFA'] * prices['NFA']) +
        (inputs['RCA_Weight'] * prices['RCA']) +
        (inputs['RFA_Weight'] * prices['RFA']) + 
        (inputs['SP'] * prices['SP'])
    )
    
    return {
        'CS28': cs28, 'STS': sts, 'CO2': co2, 'Energy': energy,
        'FS': fs, 'EM': em, 'CS7': cs7, 'CS90': cs90, 'Cost': total_cost
    }

# =============================================================================
# 5. الرادار الشامل
# =============================================================================
def show_radar_chart(results):
    # تطبيع القيم (Normalization) لرسم الرادار
    strength_score = min(results['CS28'] / 80, 1.0)
    eco_score = 1 - min(results['CO2'] / 600, 1.0)
    cost_score = 1 - min(results['Cost'] / 200, 1.0)
    energy_score = 1 - min(results['Energy'] / 3000, 1.0)

    categories = ['Structural Strength', 'CO2 Efficiency', 'Cost Efficiency', 'Energy Efficiency']
    scores = [strength_score, eco_score, cost_score, energy_score]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=categories, fill='toself',
        name='Mix Sustainability Profile', line_color='#1E3A8A', marker=dict(size=8)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".1%")),
        showlegend=False, title={'text': "<b>Comprehensive Sustainability Radar</b>", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 6. واجهة الإدخال والنتائج
# =============================================================================
def show_input_section():
    st.markdown("### 🏗️ Design Mix Inputs (21 Parameters)")
    
    with st.expander("💲 Dynamic Market Prices (USD/kg) - Update to calculate current mix cost"):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        prices = {
            'Cement': p_col1.number_input("Cement Price", value=0.15),
            'Water': p_col2.number_input("Water Price", value=0.002),
            'NCA': p_col3.number_input("NCA Price", value=0.02),
            'NFA': p_col4.number_input("NFA Price", value=0.015),
            'RCA': p_col1.number_input("RCA Price", value=0.01),
            'RFA': p_col2.number_input("RFA Price", value=0.008),
            'SP': p_col3.number_input("SP Price", value=2.5)
        }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("##### Binders & Water")
        cement = st.number_input("Cement (kg/m³)", value=350.0)
        water = st.number_input("Water (kg/m³)", value=175.0)
        w_c = st.number_input("W/C Ratio", value=0.5)
        sp = st.number_input("Superplasticizer (kg/m³)", value=2.0)
        density = st.number_input("Density (kg/m³)", value=2400.0)
        
    with c2:
        st.markdown("##### Natural Aggregates")
        nca = st.number_input("NCA (kg/m³)", value=1000.0)
        nfa = st.number_input("NFA (kg/m³)", value=700.0)
        agg_size = st.number_input("Max Agg Size (mm)", value=20.0)
        
    with c3:
        st.markdown("##### Recycled Aggregates")
        rca_w = st.number_input("RCA Weight (kg/m³)", value=0.0)
        rca_p = st.number_input("RCA (%)", value=0.0)
        mrca_p = st.number_input("MRCA (%)", value=0.0)
        rfa_w = st.number_input("RFA Weight (kg/m³)", value=0.0)
        rfa_p = st.number_input("RFA (%)", value=0.0)

    with c4:
        st.markdown("##### SCMs & Fibers")
        fly_ash = st.number_input("Fly Ash (kg/m³)", value=0.0)
        silica = st.number_input("Silica Fume (kg/m³)", value=0.0)
        mk = st.number_input("Metakaolin (kg/m³)", value=0.0)
        ggbfs = st.number_input("GGBFS (kg/m³)", value=0.0)
        rha_p = st.number_input("RHA (%)", value=0.0)
        nylon = st.number_input("Nylon Fiber (kg/m³)", value=0.0)
        basalt = st.number_input("Basalt Fiber Vol (%)", value=0.0)
        natural = st.number_input("Natural Fiber (kg/m³)", value=0.0)

    if st.button("🚀 Run Prediction & Analysis", use_container_width=True):
        inputs = {
            'Cement': cement, 'Water': water, 'W_C': w_c, 'NCA': nca, 'NFA': nfa,
            'RCA_Weight': rca_w, 'RCA_P': rca_p, 'MRCA_P': mrca_p, 'RFA_Weight': rfa_w,
            'RFA_P': rfa_p, 'Fly_Ash': fly_ash, 'Silica_Fume': silica, 'Metakaolin': mk,
            'GGBFS': ggbfs, 'RHA_P': rha_p, 'Nylon_Fiber': nylon, 'Basalt_Fiber_Vol': basalt,
            'Natural_Fiber': natural, 'SP': sp, 'Agg_Size': agg_size, 'Density': density
        }

        with st.spinner("Processing AI Models..."):
            res = run_prediction_engine(inputs, prices)
            if res:
                st.success("✅ Analysis Completed: Using Hybrid AI-Engineering Model")
                
                t_mech, t_env, t_eco = st.tabs(["🏗️ Mechanical", "🌱 Environmental", "💰 Economic"])
                
                with t_mech:
                    m1, m2 = st.columns(2)
                    m1.metric("CS 28-days (MPa) [AI Predicted]", f"{res['CS28']:.2f}")
                    m1.metric("Splitting Tensile (MPa) [AI Predicted]", f"{res['STS']:.2f}")
                    m2.metric("Flexural Strength (MPa) [ACI Estimated]", f"{res['FS']:.2f}")
                    m2.metric("Elastic Modulus (GPa) [ACI Estimated]", f"{res['EM']:.2f}")
                    m1.metric("CS 7-days (MPa) [Estimated]", f"{res['CS7']:.2f}")
                    m2.metric("CS 90-days (MPa) [Estimated]", f"{res['CS90']:.2f}")
                    
                with t_env:
                    e1, e2 = st.columns(2)
                    e1.metric("CO2 Footprint (kg/m³) [AI Predicted]", f"{res['CO2']:.2f}")
                    e2.metric("Energy Demand (MJ/m³) [AI Predicted]", f"{res['Energy']:.2f}")
                    
                with t_eco:
                    ec1, ec2 = st.columns(2)
                    ec1.metric("Total Material Cost (USD/m³) [Dynamic]", f"{res['Cost']:.2f}")
                    with ec2:
                        show_radar_chart(res)

# =============================================================================
# 7. المُحسّن (Optimizer)
# =============================================================================
def show_optimizer():
    st.header("⚖️ AI-Based Mix Optimizer")
    st.markdown("Searches the 1,701-mix database for optimal eco-efficient alternatives.")
    target_cs = st.number_input("Target Strength 28d (MPa)", value=40.0)
    tol = st.slider("Tolerance (± MPa)", 1.0, 10.0, 3.0)
    
    if st.button("Search Database"):
        try:
            df = pd.read_excel('Ready_For_AI_Training.xlsx')
            # الفلترة بناءً على المقاومة
            filtered = df[(df['CS_28'] >= target_cs - tol) & (df['CS_28'] <= target_cs + tol)]
            if not filtered.empty:
                # الترتيب حسب الأقل في انبعاثات الكربون والطاقة
                top = filtered.sort_values(by=['CO2', 'Energy'], ascending=[True, True]).head(5)
                cols = ['Mix_ID', 'Cement', 'W_C', 'CS_28', 'CO2', 'Energy']
                available = [c for c in cols if c in top.columns]
                st.dataframe(top[available].style.highlight_min(subset=['CO2', 'Energy'], color='#D1FAE5'))
            else:
                st.warning("No mixes found in this range. Try increasing the tolerance.")
        except Exception as e:
            st.error(f"Database file error: {e}")

# =============================================================================
# 8. صفحة الأداء (Performance)
# =============================================================================
def show_performance():
    st.header("📈 Model Performance & Metrics")
    
    target_choice = st.selectbox("Select Target Parameter for Analysis:", 
                                 ["Compressive Strength (CS_28)", "Tensile Strength (STS)", 
                                  "CO2 Emissions", "Energy Demand"])
    
    metrics_data = {
        "Compressive Strength (CS_28)": {"r2": "0.9706", "rmse": "2.20 MPa", "mae": "1.50 MPa", "cv": "0.8520", "prefix": "CS_28"},
        "Tensile Strength (STS)": {"r2": "0.9566", "rmse": "0.26 MPa", "mae": "0.15 MPa", "cv": "0.7730", "prefix": "STS"},
        "CO2 Emissions": {"r2": "0.9958", "rmse": "5.75 kg", "mae": "0.56 kg", "cv": "0.9152", "prefix": "CO2"},
        "Energy Demand": {"r2": "0.9850", "rmse": "139.44 MJ", "mae": "9.08 MJ", "cv": "0.8221", "prefix": "Energy"}
    }
    
    data = metrics_data[target_choice]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", data["r2"])
    c2.metric("RMSE", data["rmse"])
    c3.metric("MAE", data["mae"])
    c4.metric("Cross-Val Score (5-Fold)", data["cv"])
    
    st.divider()
    st.subheader(f"🔬 Visual Diagnostics: {target_choice}")
    
    p1, p2 = st.columns(2)
    with p1:
        img_path = f"Performance_Plots/{data['prefix']}_Actual_vs_Predicted.png"
        if os.path.exists(img_path): st.image(img_path, caption="Actual vs Predicted", use_container_width=True)
        else: st.warning(f"Image not found: {img_path}")
            
    with p2:
        img_path2 = f"Performance_Plots/{data['prefix']}_Residuals.png"
        if os.path.exists(img_path2): st.image(img_path2, caption="Residuals Distribution", use_container_width=True)
        else: st.warning(f"Image not found: {img_path2}")
            
    st.markdown("---")
    img_path3 = f"Performance_Plots/{data['prefix']}_Feature_Importance.png"
    if os.path.exists(img_path3): st.image(img_path3, caption="Feature Importance Analysis", use_container_width=True)

# =============================================================================
# 9. الدالة الرئيسية (Main)
# =============================================================================
def main():
    show_academic_header()
    tabs = st.tabs(["🏠 Home", "🚀 Predictor", "⚖️ Optimizer", "📈 Performance", "📝 Feedback", "📚 Docs"])
    
    with tabs[0]:
        st.markdown("### 🎯 Your AI-Powered Tool for Eco-Efficient Concrete Design")
        st.info("System optimized using 1,701 laboratory samples predicting mechanical and environmental outcomes.")
        st.markdown("""
        **Welcome to the ASA-PREDICTION MODEL 2!**  
        This dashboard integrates advanced Machine Learning (Multi-Output Random Forest) with 
        standard engineering equations (ACI) to provide a comprehensive, real-time evaluation 
        of concrete mixtures across three main pillars: Structural, Environmental, and Economic.
        """)
    
    with tabs[1]: show_input_section()
    with tabs[2]: show_optimizer()
    with tabs[3]: show_performance()
    with tabs[4]: st.write("Feedback form will be integrated here.")
    with tabs[5]: st.write("Methodology and standard ACI 318 calculations documented here.")

if __name__ == "__main__":
    main()
