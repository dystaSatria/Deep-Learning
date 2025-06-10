import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Veri Analizi Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık ve açıklama
st.title("📊 Veri Analizi Dashboard")
st.markdown("Bu uygulama ile veri analizi ve görselleştirme işlemlerini kolayca gerçekleştirebilirsiniz.")

# Sidebar - Navigasyon
st.sidebar.title("Navigasyon")
sayfa = st.sidebar.selectbox(
    "Bir sayfa seçin:",
    ["Ana Sayfa", "Veri Yükleme", "Veri Analizi", "Görselleştirme", "Makine Öğrenmesi"]
)

# Örnek veri oluşturma fonksiyonu
@st.cache_data
def ornek_veri_olustur():
    np.random.seed(42)
    tarihler = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    veri = pd.DataFrame({
        'Tarih': tarihler,
        'Satış': np.random.normal(1000, 200, len(tarihler)),
        'Müşteri_Sayısı': np.random.poisson(50, len(tarihler)),
        'Kategori': np.random.choice(['Elektronik', 'Giyim', 'Ev_Eşyası', 'Kitap'], len(tarihler)),
        'Şehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya'], len(tarihler))
    })
    veri['Gelir'] = veri['Satış'] * np.random.uniform(0.8, 1.2, len(veri))
    return veri

# Ana Sayfa
if sayfa == "Ana Sayfa":
    st.header("Hoş Geldiniz! 🎉")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Toplam Satış",
            value="₺2.5M",
            delta="15.2%"
        )
    
    with col2:
        st.metric(
            label="Aktif Müşteri",
            value="1,250",
            delta="-5.1%"
        )
    
    with col3:
        st.metric(
            label="Ortalama Sipariş",
            value="₺950",
            delta="8.7%"
        )
    
    st.markdown("---")
    
    # Hızlı İstatistikler
    st.subheader("📈 Hızlı İstatistikler")
    
    # Örnek veri ile grafik
    df = ornek_veri_olustur()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df.head(100), x='Tarih', y='Satış', title='Son 100 Günlük Satış Trendi')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        kategori_satış = df.groupby('Kategori')['Satış'].sum().reset_index()
        fig = px.pie(kategori_satış, values='Satış', names='Kategori', title='Kategori Bazında Satış Dağılımı')
        st.plotly_chart(fig, use_container_width=True)

# Veri Yükleme Sayfası
elif sayfa == "Veri Yükleme":
    st.header("📁 Veri Yükleme")
    
    # Dosya yükleme
    yuklenebilecek_dosya = st.file_uploader(
        "CSV dosyası yükleyin:",
        type=['csv'],
        help="Lütfen geçerli bir CSV dosyası seçin"
    )
    
    if yuklenebilecek_dosya is not None:
        try:
            df = pd.read_csv(yuklenebilecek_dosya)
            st.success("Dosya başarıyla yüklendi!")
            st.dataframe(df.head())
            
            # Veri bilgileri
            st.subheader("Veri Bilgileri")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Satır sayısı:** {df.shape[0]}")
                st.write(f"**Sütun sayısı:** {df.shape[1]}")
            
            with col2:
                st.write("**Sütun tipleri:**")
                st.write(df.dtypes)
            
        except Exception as e:
            st.error(f"Dosya yüklenirken hata oluştu: {e}")
    
    else:
        st.info("Örnek veri ile devam etmek için 'Veri Analizi' sayfasına geçin.")

# Veri Analizi Sayfası
elif sayfa == "Veri Analizi":
    st.header("🔍 Veri Analizi")
    
    df = ornek_veri_olustur()
    
    # Veri önizleme
    st.subheader("Veri Önizleme")
    st.dataframe(df.head())
    
    # Temel istatistikler
    st.subheader("📊 Temel İstatistikler")
    st.write(df.describe())
    
    # Filtreleme seçenekleri
    st.subheader("🔧 Filtreleme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kategori_filtre = st.multiselect(
            "Kategori seçin:",
            options=df['Kategori'].unique(),
            default=df['Kategori'].unique()
        )
    
    with col2:
        şehir_filtre = st.multiselect(
            "Şehir seçin:",
            options=df['Şehir'].unique(),
            default=df['Şehir'].unique()
        )
    
    # Tarih aralığı
    tarih_aralığı = st.date_input(
        "Tarih aralığı seçin:",
        value=(df['Tarih'].min(), df['Tarih'].max()),
        min_value=df['Tarih'].min(),
        max_value=df['Tarih'].max()
    )
    
    # Filtrelenmiş veri
    filtrelenmiş_df = df[
        (df['Kategori'].isin(kategori_filtre)) &
        (df['Şehir'].isin(şehir_filtre)) &
        (df['Tarih'] >= pd.to_datetime(tarih_aralığı[0])) &
        (df['Tarih'] <= pd.to_datetime(tarih_aralığı[1]))
    ]
    
    st.subheader("Filtrelenmiş Veri")
    st.write(f"Toplam {len(filtrelenmiş_df)} kayıt")
    st.dataframe(filtrelenmiş_df)

# Görselleştirme Sayfası
elif sayfa == "Görselleştirme":
    st.header("📈 Görselleştirme")
    
    df = ornek_veri_olustur()
    
    # Grafik türü seçimi
    grafik_türü = st.selectbox(
        "Grafik türü seçin:",
        ["Çizgi Grafik", "Bar Grafik", "Scatter Plot", "Histogram", "Heatmap"]
    )
    
    if grafik_türü == "Çizgi Grafik":
        fig = px.line(df, x='Tarih', y='Satış', color='Kategori', title='Zaman Serisi Satış Analizi')
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafik_türü == "Bar Grafik":
        şehir_satış = df.groupby('Şehir')['Satış'].sum().reset_index()
        fig = px.bar(şehir_satış, x='Şehir', y='Satış', title='Şehir Bazında Toplam Satış')
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafik_türü == "Scatter Plot":
        fig = px.scatter(df, x='Müşteri_Sayısı', y='Satış', color='Kategori', 
                        title='Müşteri Sayısı vs Satış İlişkisi')
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafik_türü == "Histogram":
        fig = px.histogram(df, x='Satış', nbins=50, title='Satış Dağılımı')
        st.plotly_chart(fig, use_container_width=True)
    
    elif grafik_türü == "Heatmap":
        # Korelasyon matrisi
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title='Korelasyon Matrisi')
        st.plotly_chart(fig, use_container_width=True)

# Makine Öğrenmesi Sayfası
elif sayfa == "Makine Öğrenmesi":
    st.header("🤖 Makine Öğrenmesi")
    
    st.info("Bu bölümde basit makine öğrenmesi modelleri oluşturabilirsiniz.")
    
    df = ornek_veri_olustur()
    
    # Model türü seçimi
    model_türü = st.selectbox(
        "Model türü seçin:",
        ["Lineer Regresyon", "Satış Tahmini", "Kategori Sınıflandırma"]
    )
    
    if model_türü == "Satış Tahmini":
        st.subheader("📊 Gelecek Satış Tahmini")
        
        # Basit trend analizi
        günlük_satış = df.groupby('Tarih')['Satış'].sum().reset_index()
        günlük_satış['Gün_Sayısı'] = range(len(günlük_satış))
        
        # Basit lineer trend
        trend = np.polyfit(günlük_satış['Gün_Sayısı'], günlük_satış['Satış'], 1)
        
        # Gelecek tahminleri
        gelecek_günler = 30
        gelecek_x = range(len(günlük_satış), len(günlük_satış) + gelecek_günler)
        gelecek_tahmin = np.polyval(trend, gelecek_x)
        
        # Grafik
        fig = go.Figure()
        
        # Mevcut veri
        fig.add_trace(go.Scatter(
            x=günlük_satış['Tarih'],
            y=günlük_satış['Satış'],
            mode='lines',
            name='Gerçek Satış',
            line=dict(color='blue')
        ))
        
        # Tahmin
        gelecek_tarihler = pd.date_range(
            start=günlük_satış['Tarih'].max() + timedelta(days=1),
            periods=gelecek_günler,
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=gelecek_tarihler,
            y=gelecek_tahmin,
            mode='lines',
            name='Tahmin',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(title='Satış Tahmin Modeli', xaxis_title='Tarih', yaxis_title='Satış')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Önümüzdeki 30 gün için ortalama günlük satış tahmini: ₺{gelecek_tahmin.mean():.0f}")

# Footer
st.markdown("---")
st.markdown("Streamlit ile oluşturulmuş demo uygulama 🚀")
