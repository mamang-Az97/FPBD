import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Kopi Tokopedia",
    page_icon="☕",
    layout="wide"
)

# ---------------------------------------------------------
# FUNGSI HELPER
# ---------------------------------------------------------
@st.cache_data
def convert_df_to_csv(df_to_convert):
    """Mengubah DataFrame ke CSV dengan Caching agar hemat memori/proses."""
    return df_to_convert.to_csv(index=False).encode('utf-8')

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    df['Omzet'] = df['Harga'] * df['Terjual']
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file csv berada dalam folder yang sama!")
    st.stop()

# ---------------------------------------------------------
# HEADER DASHBOARD
# ---------------------------------------------------------
st.title("☕ Dashboard Analisis Pasar Kopi Tokopedia")
st.markdown("Dashboard interaktif untuk menganalisis hubungan antara **Harga**, **Rating**, **Varian Kopi**, dan tingkat **Penjualan (Terjual)**.")

# ---------------------------------------------------------
# SIDEBAR - FILTER DATA
# ---------------------------------------------------------
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
    step=100
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

# Penerapan Filter
df_filtered = df[
    (df['Harga'] >= selected_harga[0]) & 
    (df['Harga'] <= selected_harga[1]) &
    (df['Rating'] >= selected_rating[0]) & 
    (df['Rating'] <= selected_rating[1])
]

if selected_varian != 'Semua Varian':
    df_filtered = df_filtered[df_filtered['Varian Kopi'] == selected_varian]

# ---------------------------------------------------------
# METRIKS UTAMA (KPI)
# ---------------------------------------------------------
st.markdown("### 📊 Ringkasan Eksekutif")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Produk", f"{len(df_filtered):,} item")
with col2:
    total_terjual = df_filtered['Terjual'].sum() if not df_filtered.empty else 0
    st.metric("Total Terjual", f"{total_terjual:,} unit")
with col3:
    mean_rating = df_filtered['Rating'].mean() if not df_filtered.empty else 0
    st.metric("Rata-Rata Rating", f"⭐ {mean_rating:.2f}")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    mean_harga = df_filtered['Harga'].mean() if not df_filtered.empty else 0
    st.metric("Rata-Rata Harga", f"Rp {mean_harga:,.0f}")
with col2:
    total_omzet = df_filtered['Omzet'].sum() if not df_filtered.empty else 0
    st.metric("Total Estimasi Omzet", f"Rp {total_omzet:,.0f}")

st.markdown("---")

# ---------------------------------------------------------
# TAB ANALISIS & VISUALISASI
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Analisis & Korelasi", "🏆 Top Produk & Toko", "🤖 Simulasi Regresi Linier & Omset"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribusi Varian Kopi")
        df_varian = df_filtered['Varian Kopi'].value_counts().reset_index()
        df_varian.columns = ['Varian Kopi', 'Jumlah Produk']
        fig_varian = px.bar(
            df_varian.head(10), 
            x='Jumlah Produk', 
            y='Varian Kopi', 
            orientation='h',
            title="Top 10 Jumlah Produk per Varian",
            color='Jumlah Produk',
            color_continuous_scale='Blues'
        )
        fig_varian.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_varian, use_container_width=True)

    with col_right:
        st.subheader("Hubungan Harga vs Terjual")
        fig_scatter = px.scatter(
            df_filtered, 
            x='Harga', 
            y='Terjual', 
            color='Varian Kopi',
            hover_data=['Nama Produk', 'Rating'],
            title="Scatter Plot: Harga vs Terjual"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Matriks Korelasi (Harga, Rating, Terjual)")
    if len(df_filtered) > 1:
        corr = df_filtered[['Harga', 'Rating', 'Terjual']].corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Korelasi Antar Variabel Numerik"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("10 Produk Terlaris")
        top_produk = df_filtered.nlargest(10, 'Terjual')[['Nama Produk', 'Varian Kopi', 'Harga', 'Terjual']]
        st.dataframe(top_produk, hide_index=True, use_container_width=True)

    with col_t2:
        st.subheader("10 Toko / Lokasi Penjualan Tertinggi")
        top_toko = df_filtered.groupby('Asal/Lokasi Toko')['Terjual'].sum().reset_index().nlargest(10, 'Terjual')
        fig_toko = px.bar(
            top_toko,
            x='Terjual',
            y='Asal/Lokasi Toko',
            orientation='h',
            title="Total Terjual per Toko",
            color='Terjual',
            color_continuous_scale='Greens'
        )
        fig_toko.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_toko, use_container_width=True)
    
    st.markdown("---")
    
    # Layout sejajar untuk Judul + Informasi Jumlah Data dan Tombol Download
    col_header, col_download = st.columns([3, 1], vertical_alignment="bottom")
    
    with col_header:
        st.subheader("📋 Seluruh Data Terfilter")
        st.caption(f"Menampilkan **{len(df_filtered):,}** baris data berdasarkan filter aktif.")
    
    with col_download:
        csv_data = convert_df_to_csv(df_filtered)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="data_kopi_tokopedia_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    
    # Tampilan Tabel Interaktif
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga": st.column_config.NumberColumn("Harga", format="Rp %'d"),
            "Rating": st.column_config.NumberColumn("Rating", format="⭐ %.1f"),
            "Terjual": st.column_config.NumberColumn("Terjual", format="%d unit"),
            "Omzet": st.column_config.NumberColumn("Estimasi Omzet", format="Rp %'d"),
            "Link Produk": st.column_config.LinkColumn("Link Tokopedia", display_text="Lihat Produk")
        }
    )

with tab3:
    st.subheader("Model Regresi Linier Berganda")
    st.markdown("Model ini menghitung estimasi **Jumlah Terjual (Y)** berdasarkan **Harga ($X_1$)** dan **Rating ($X_2$)**.")
    
    # Training Model Regresi
    X = df[['Harga', 'Rating']]
    Y = df['Terjual']
    model = LinearRegression()
    model.fit(X, Y)
    
    a = model.intercept_
    b1 = model.coef_[0]
    b2 = model.coef_[1]
    
    st.info("Persamaan Regresi Linier Berganda Terbentuk:")
    st.latex(rf"Y = {a:.2f} + ({b1:.6f}) \times \text{{Harga}} + ({b2:.2f}) \times \text{{Rating}}")
    
    st.markdown("#### 🔮 Simulasi Prediksi Penjualan")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        input_harga = st.number_input("Input Harga Produk (Rp)", min_value=400, max_value=2000000, value=50000, step=5000)
    with col_input2:
        input_rating = st.slider("Input Rating Produk", float(df['Rating'].min()), float(df['Rating'].max()), value=5.0)

    pred_terjual = model.predict([[input_harga, input_rating]])[0]
    pred_terjual_clean = max(0, int(np.round(pred_terjual)))

    st.success(f"📌 Estimasikan Penjualan: **± {pred_terjual_clean} unit**") 

    # ---------------------------------------------------------
    # VISUALISASI GARIS TREN PREDIKSI VS DATA AKTUAL
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Evaluasi Garis Tren Prediksi Model vs Data Aktual")
    
    if len(df_filtered) > 2:
        min_x = df_filtered['Harga'].min()
        max_x = df_filtered['Harga'].max()
        avg_rating = df_filtered['Rating'].mean()
    
        x_line = np.linspace(min_x, max_x, 100)
        df_line = pd.DataFrame({'Harga': x_line, 'Rating': avg_rating})
        
        y_line = model.predict(df_line)
        y_line_clean = np.maximum(0, y_line)
    
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_filtered['Harga'],
            y=df_filtered['Terjual'],
            mode='markers',
            name='Data Aktual (Tokopedia)',
            marker=dict(color='#3366CC', opacity=0.6, size=7),
            hovertemplate="<b>Harga:</b> Rp %{x:,.0f}<br><b>Terjual:</b> %{y:,} unit<extra></extra>"
        ))
    
        fig_trend.add_trace(go.Scatter(
            x=x_line,
            y=y_line_clean,
            mode='lines',
            name='Garis Tren Prediksi Model',
            line=dict(color='red', width=3),
            hovertemplate="<b>Harga Prediksi:</b> Rp %{x:,.0f}<br><b>Estimasi Terjual:</b> %{y:,.1f} unit<extra></extra>"
        ))
    
        fig_trend.update_layout(
            xaxis_title="Harga Produk (Rupiah)",
            yaxis_title="Jumlah Produk Terjual (Unit)",
            legend=dict(
                x=0.70, y=0.95,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="LightGray",
                borderwidth=1
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=30, b=40)
        )
    
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Data terfilter terlalu sedikit untuk menampilkan grafik tren regresi.")

    st.markdown("---")
    
    # Visualisasi Pangsa Pasar Omzet
    st.subheader("📊 Analisis Pasar Omzet")
    st.latex(rf"\text{{Omzet}} = \text{{Harga}} \times \text{{Terjual}}")
    
    col_omzet1, col_omzet2 = st.columns(2)
    with col_omzet1:
        fig_donut_omzet = px.pie(
            df_filtered, 
            names='Varian Kopi', 
            values='Omzet', 
            hole=0.4,
            title="Market Share Omzet (Rupiah) per Varian Kopi",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut_omzet.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_donut_omzet, use_container_width=True)
        
    with col_omzet2:
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
