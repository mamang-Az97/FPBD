import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis Pasar Kopi Tokopedia",
    page_icon="☕",
    layout="wide"
)

# =========================================================
# 2. HELPER FUNCTIONS & LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    """Membaca data CSV dan menghitung kolom Omzet secara otomatis."""
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    
    # Pastikan tipe data numerik
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce').fillna(0)
    df['Terjual'] = pd.to_numeric(df['Terjual'], errors='coerce').fillna(0)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    
    # Menambahkan variabel Omzet (Revenue = Harga x Terjual)
    df['Omzet'] = df['Harga'] * df['Terjual']
    return df

@st.cache_data
def convert_df_to_csv(df):
    """Mengonversi DataFrame ke CSV dengan Caching untuk optimasi performa."""
    return df.to_csv(index=False).encode('utf-8')

# Load Data
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file csv berada dalam folder yang sama dengan app.py!")
    st.stop()

# =========================================================
# 3. HEADER DASHBOARD
# =========================================================
st.title("☕ Dashboard Analisis Pasar Kopi Tokopedia")
st.markdown("Dashboard interaktif untuk menganalisis **Harga**, **Rating**, **Varian Kopi**, **Jumlah Terjual**, dan **Estimasi Omzet**.")

# =========================================================
# 4. SIDEBAR (FILTER DATA)
# =========================================================
st.sidebar.header("🔍 Filter Data")

# Filter Varian Kopi
varians = ['Semua Varian'] + sorted(list(df['Varian Kopi'].dropna().unique()))
selected_varian = st.sidebar.selectbox("Pilih Varian Kopi", varians)

# Filter Rentang Harga
min_harga, max_harga = int(df['Harga'].min()), int(df['Harga'].max())
selected_harga = st.sidebar.slider(
    "Rentang Harga (Rp)",
    min_value=min_harga,
    max_value=max_harga,
    value=(min_harga, min(max_harga, 500000)),
    step=5000
)

# Filter Rating
min_rating, max_rating = float(df['Rating'].min()), float(df['Rating'].max())
selected_rating = st.sidebar.slider(
    "Rentang Rating",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=0.1
)

# Penerapan Filter ke DataFrame
df_filtered = df[
    (df['Harga'] >= selected_harga[0]) & 
    (df['Harga'] <= selected_harga[1]) &
    (df['Rating'] >= selected_rating[0]) & 
    (df['Rating'] <= selected_rating[1])
]

if selected_varian != 'Semua Varian':
    df_filtered = df_filtered[df_filtered['Varian Kopi'] == selected_varian]

# =========================================================
# 5. RINGKASAN EKSEKUTIF (KPI CARDS)
# =========================================================
st.markdown("### 📊 Ringkasan Eksekutif")
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric("Total Produk", f"{len(df_filtered):,} item")
with col_kpi2:
    total_terjual = df_filtered['Terjual'].sum() if not df_filtered.empty else 0
    st.metric("Total Terjual", f"{total_terjual:,} unit")
with col_kpi3:
    total_omzet = df_filtered['Omzet'].sum() if not df_filtered.empty else 0
    st.metric("Estimasi Total Omzet", f"Rp {total_omzet:,.0f}")
with col_kpi4:
    mean_rating = df_filtered['Rating'].mean() if not df_filtered.empty else 0
    st.metric("Rata-Rata Rating", f"⭐ {mean_rating:.2f}")

st.markdown("---")

# =========================================================
# 6. TAB ANALISIS & VISUALISASI DATA
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Analisis Pasar & Korelasi", 
    "🏆 Top Performers & Market Share", 
    "🤖 Simulasi Regresi Linier"
])

# ---------------------------------------------------------
# TAB 1: ANALISIS PASAR & KORELASI
# ---------------------------------------------------------
with tab1:
    col_t1a, col_t1b = st.columns(2)
    
    with col_t1a:
        st.subheader("Distribusi Varian Kopi")
        df_varian = df_filtered['Varian Kopi'].value_counts().reset_index()
        df_varian.columns = ['Varian Kopi', 'Jumlah Produk']
        fig_varian = px.bar(
            df_varian.head(10), 
            x='Jumlah Produk', 
            y='Varian Kopi', 
            orientation='h',
            title="Top 10 Produk Terbanyak per Varian",
            color='Jumlah Produk',
            color_continuous_scale='Blues'
        )
        fig_varian.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_varian, use_container_width=True)

    with col_t1b:
        st.subheader("Analisis 4D: Harga vs Rating vs Terjual")
        fig_bubble = px.scatter(
            df_filtered, 
            x='Harga', 
            y='Rating',
            size='Terjual', 
            color='Varian Kopi',
            hover_name='Nama Produk',
            title="Bubble Size = Jumlah Terjual"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    col_t1c, col_t1d = st.columns(2)
    
    with col_t1c:
        st.subheader("Distribusi & Outlier Harga")
        fig_box = px.box(
            df_filtered, 
            x='Varian Kopi', 
            y='Harga', 
            color='Varian Kopi',
            title="Sebaran Harga per Varian Kopi"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_t1d:
        st.subheader("Matriks Korelasi (Harga vs Rating vs Terjual vs Omzet)")
        if len(df_filtered) > 1:
            corr = df_filtered[['Harga', 'Rating', 'Terjual', 'Omzet']].corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Korelasi Antar Variabel Numerik"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("⚠️ Data tidak mencukupi untuk menghitung matriks korelasi.")

# ---------------------------------------------------------
# TAB 2: TOP PERFORMERS & MARKET SHARE
# ---------------------------------------------------------
with tab2:
    col_t2a, col_t2b = st.columns(2)
    
    with col_t2a:
        st.subheader("Pangsa Pasar (Market Share) Omzet")
        fig_donut = px.pie(
            df_filtered, 
            names='Varian Kopi', 
            values='Omzet', 
            hole=0.4,
            title="Proporsi Omzet (Rupiah) per Varian",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_t2b:
        st.subheader("Top 10 Toko Berdasarkan Omzet")
        top_toko_omzet = df_filtered.groupby('Asal/Lokasi Toko')['Omzet'].sum().reset_index().nlargest(10, 'Omzet')
        fig_toko_omzet = px.bar(
            top_toko_omzet,
            x='Omzet',
            y='Asal/Lokasi Toko',
            orientation='h',
            title="Total Omzet (Rp) per Toko/Lokasi",
            color='Omzet',
            color_continuous_scale='Greens'
        )
        fig_toko_omzet.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_toko_omzet, use_container_width=True)

    st.subheader("Hierarki Pasar: Lokasi ➔ Varian Kopi")
    if not df_filtered.empty:
        fig_treemap = px.treemap(
            df_filtered, 
            path=['Asal/Lokasi Toko', 'Varian Kopi'], 
            values='Omzet',
            title="Hierarki Market Share Omzet Berdasarkan Lokasi & Varian"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SIMULASI REGRESI LINIER
# ---------------------------------------------------------
with tab3:
    st.subheader("Model Regresi Linier Berganda")
    st.markdown("Model ini menghitung estimasi **Jumlah Terjual (Y)** berdasarkan **Harga ($X_1$)** dan **Rating ($X_2$)**.")
    
    if len(df) > 2:
        X = df[['Harga', 'Rating']]
        Y = df['Terjual']
        model = LinearRegression()
        model.fit(X, Y)
        
        a = model.intercept_
        b1 = model.coef_[0]
        b2 = model.coef_[1]
        
        st.info("Persamaan Regresi Linier Berganda Terbentuk:")
        st.latex(rf"Y = {a:.2f} + ({b1:.6f}) \times \text{{Harga}} + ({b2:.2f}) \times \text{{Rating}}")
        
        st.markdown("#### 🔮 Simulasi Prediksi Penjualan & Omzet")
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            input_harga = st.number_input("Input Harga Produk (Rp)", min_value=1000, max_value=2000000, value=50000, step=5000)
        with col_in2:
            input_rating = st.slider("Input Rating Produk", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
            
        pred_terjual = model.predict([[input_harga, input_rating]])[0]
        pred_terjual_clean = max(0, int(np.round(pred_terjual)))
        pred_omzet = pred_terjual_clean * input_harga
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.success(f"📌 Estimasi Terjual: **± {pred_terjual_clean:,} unit**")
        with col_res2:
            st.success(f"💰 Estimasi Omzet: **± Rp {pred_omzet:,.0f}**")
    else:
        st.warning("Data tidak mencukupi untuk melatih model regresi.")

# =========================================================
# 7. TABEL DATA TERFILTER & DOWNLOAD BUTTON
# =========================================================
st.markdown("---")

col_header, col_download = st.columns([3, 1], vertical_alignment="bottom")

with col_header:
    st.subheader("📋 Seluruh Data Terfilter")
    st.caption(f"Menampilkan **{len(df_filtered):,}** baris data berdasarkan filter aktif.")

with col_download:
    csv_data = convert_df_to_csv(df_filtered)
    st.download_button(
        label="📥 Download Data CSV",
        data=csv_data,
        file_name="data_kopi_tokopedia_filtered.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

st.dataframe(
    df_filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Harga": st.column_config.NumberColumn("Harga", format="Rp %'d"),
        "Terjual": st.column_config.NumberColumn("Terjual", format="%d unit"),
        "Omzet": st.column_config.NumberColumn("Estimasi Omzet", format="Rp %'d"),
        "Rating": st.column_config.NumberColumn("Rating", format="⭐ %.1f")
    }
)
