import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import json
import time
import platform
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
import os
import urllib.request

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RSEI计算系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 强力中文字体解决方案 ====================
@st.cache_resource
# ==================== 终极强力中文字体解决方案 v2 ====================
@st.cache_resource
def download_and_setup_chinese_font():
    """
    下载并设置中文字体。优先使用国内CDN，增加备用链接，确保成功。
    """
    font_dir = Path("./fonts")
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / "SimHei.ttf"

    if not font_path.exists():
        st.info("首次运行，正在配置中文字体...")

        # 优先使用国内CDN，备用GitHub
        font_urls = [
            # "http://www.fonts.net.cn/Public/Uploads/2019-01-02/5c2c77f02f5a1.ttf",  # SimHei from fonts.net.cn
            "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"  # GitHub backup
        ]

        download_success = False
        for i, url in enumerate(font_urls, 1):
            try:
                with st.spinner(f"正在尝试从源 {i} 下载字体..."):
                    urllib.request.urlretrieve(url, font_path)

                if font_path.exists() and font_path.stat().st_size > 1000:  # 检查文件是否有效
                    st.success(f"✅ 字体下载成功！")
                    download_success = True
                    break
            except Exception as e:
                st.warning(f"⚠️ 从源 {i} 下载失败: {e}")
                if font_path.exists():
                    font_path.unlink()  # 删除不完整的文件

        if not download_success:
            st.error("❌ 所有字体下载源均失败。中文将无法正常显示。请检查网络连接或防火墙设置。")
            return None

    # 加载字体
    try:
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        font_name = font_prop.get_name()

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        return f"✅ 字体 '{font_name}' 加载成功"
    except Exception as e:
        st.error(f"❌ 字体文件加载失败: {e}")
        return None


# 初始化字体
font_status = download_and_setup_chinese_font()


# =============================
# 核心类（与Tkinter版本相同）
# =============================
@dataclass
class BandConfig:
    blue: int = 1
    green: int = 2
    red: int = 3
    nir: int = 4
    swir1: int = 5
    swir2: int = 6
    tir: int = 7

    @classmethod
    def landsat8_stacked(cls):
        return cls(blue=1, green=2, red=3, nir=4, swir1=5, swir2=6, tir=7)

    @classmethod
    def sentinel2_stacked(cls):
        return cls(blue=1, green=2, red=3, nir=4, swir1=5, swir2=6, tir=None)


@dataclass
class RSEIConfig:
    satellite: str = 'Landsat8'
    band_config: BandConfig = None
    use_pca: bool = True
    export_indices: bool = True
    export_geotiff: bool = True
    mask_water: bool = True
    water_index: str = 'MNDWI'
    water_threshold: Optional[float] = None
    use_otsu: bool = True

    def __post_init__(self):
        if self.band_config is None:
            if self.satellite.lower() == 'landsat8':
                self.band_config = BandConfig.landsat8_stacked()
            elif self.satellite.lower() == 'sentinel2':
                self.band_config = BandConfig.sentinel2_stacked()
            else:
                self.band_config = BandConfig()


class MultiSpectralImageReader:
    def __init__(self, config: RSEIConfig):
        self.config = config
        self.metadata = None
        self.bands_data = {}

    def read_multiband_tif(self, tif_path: str) -> Dict[str, np.ndarray]:
        st.write(f"📡 读取: {Path(tif_path).name}")

        with rasterio.open(tif_path) as src:
            st.write(f"  尺寸: {src.width} x {src.height}, 波段: {src.count}")

            self.metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height,
                'dtype': 'float32',
                'count': 1,
                'driver': 'GTiff',
                'nodata': np.nan
            }

            required_bands = self._get_required_bands()
            max_band_idx = max([idx for idx in required_bands.values() if idx is not None])

            if src.count < max_band_idx:
                raise ValueError(f"波段不足: 需要{max_band_idx}个")

            bands = {}
            for band_name, band_idx in required_bands.items():
                if band_idx is None:
                    continue

                band_data = src.read(band_idx).astype(float)
                if src.nodata is not None:
                    band_data[band_data == src.nodata] = np.nan
                band_data[band_data < -9999] = np.nan
                band_data[band_data > 50000] = np.nan
                bands[band_name] = band_data

        st.success(f"✅ 读取{len(bands)}个波段")
        self.bands_data = bands
        return bands

    def _get_required_bands(self) -> Dict[str, int]:
        bc = self.config.band_config
        return {
            'blue': bc.blue, 'green': bc.green, 'red': bc.red,
            'nir': bc.nir, 'swir1': bc.swir1, 'swir2': bc.swir2,
            'tir': bc.tir
        }

    def apply_scale_factor(self, bands: Dict[str, np.ndarray],
                           scale_factor: float = 0.0001) -> Dict[str, np.ndarray]:
        st.write(f"🔧 缩放因子: {scale_factor}")
        scaled_bands = {}
        optical_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

        for band_name, band_data in bands.items():
            if band_name in optical_bands:
                scaled_bands[band_name] = band_data * scale_factor
            else:
                scaled_bands[band_name] = band_data
        return scaled_bands


class OTSUThreshold:
    @staticmethod
    def calculate_otsu_threshold(data: np.ndarray, bins: int = 256) -> Tuple[float, Dict]:
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            raise ValueError("数据全为NaN")

        hist, bin_edges = np.histogram(valid_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist = hist.astype(float)
        prob = hist / hist.sum()

        max_variance = 0
        optimal_threshold = 0

        w0 = np.cumsum(prob)
        w1 = 1 - w0
        mu = np.cumsum(prob * bin_centers)
        mu_total = mu[-1]

        for i in range(1, bins):
            if w0[i] == 0 or w1[i] == 0:
                continue

            mu0 = mu[i] / w0[i] if w0[i] > 0 else 0
            mu1 = (mu_total - mu[i]) / w1[i] if w1[i] > 0 else 0
            variance = w0[i] * w1[i] * (mu0 - mu1) ** 2

            if variance > max_variance:
                max_variance = variance
                optimal_threshold = bin_centers[i]

        threshold_idx = np.argmin(np.abs(bin_centers - optimal_threshold))

        metrics = {
            'threshold': optimal_threshold,
            'max_variance': max_variance,
            'histogram': hist,
            'bin_centers': bin_centers,
            'threshold_idx': threshold_idx,
            'background_prob': w0[threshold_idx],
            'foreground_prob': w1[threshold_idx]
        }

        return optimal_threshold, metrics


class WaterMaskGenerator:
    @staticmethod
    def calculate_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            mndwi = (green - swir1) / (green + swir1)
        return mndwi

    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
        return ndwi

    @staticmethod
    def calculate_aweish(blue: np.ndarray, green: np.ndarray,
                         nir: np.ndarray, swir1: np.ndarray,
                         swir2: np.ndarray) -> np.ndarray:
        aweish = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
        return aweish

    @staticmethod
    def create_water_mask(bands: Dict[str, np.ndarray],
                          method: str = 'MNDWI',
                          threshold: Optional[float] = None,
                          use_otsu: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        st.write(f"💧 水体掩膜 ({method})")

        if method.upper() == 'NDWI':
            water_index = WaterMaskGenerator.calculate_ndwi(bands['green'], bands['nir'])
        elif method.upper() == 'MNDWI':
            water_index = WaterMaskGenerator.calculate_mndwi(bands['green'], bands['swir1'])
        elif method.upper() == 'AWEISH':
            water_index = WaterMaskGenerator.calculate_aweish(
                bands['blue'], bands['green'], bands['nir'],
                bands['swir1'], bands['swir2']
            )
        else:
            raise ValueError(f"不支持: {method}")

        if use_otsu and threshold is None:
            st.write("  🔍 OTSU计算中...")
            try:
                otsu_threshold, metrics = OTSUThreshold.calculate_otsu_threshold(water_index, bins=256)
                final_threshold = otsu_threshold
                st.write(f"    阈值: {final_threshold:.4f}")
                st.write(f"    方差: {metrics['max_variance']:.6f}")
            except Exception as e:
                st.warning(f"OTSU失败: {e}")
                final_threshold = 0.0
        elif threshold is not None:
            final_threshold = threshold
        else:
            final_threshold = 0.0

        water_mask = water_index > final_threshold

        total_pixels = np.sum(~np.isnan(water_index))
        water_pixels = np.sum(water_mask)
        water_ratio = water_pixels / total_pixels * 100 if total_pixels > 0 else 0

        st.write(f"  水域: {water_pixels:,} ({water_ratio:.2f}%)")

        return water_index, water_mask, final_threshold


class RemoteSensingIndices:
    @staticmethod
    def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
        return np.clip(ndvi, -1, 1)

    @staticmethod
    def calculate_ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ndbi = (swir1 - nir) / (swir1 + nir)
        return ndbi

    @staticmethod
    def calculate_ndbsi(bands: Dict[str, np.ndarray]) -> np.ndarray:
        swir1, red, nir, blue = bands['swir1'], bands['red'], bands['nir'], bands['blue']
        with np.errstate(divide='ignore', invalid='ignore'):
            si = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
            ndbi = RemoteSensingIndices.calculate_ndbi(swir1, nir)
            ndbsi = (si + ndbi) / 2
        return ndbsi

    @staticmethod
    def calculate_wet(bands: Dict[str, np.ndarray], sensor: str = 'Landsat8') -> np.ndarray:
        coeffs = {
            'blue': 0.1511, 'green': 0.1973, 'red': 0.3283,
            'nir': 0.3407, 'swir1': -0.7117, 'swir2': -0.4559
        }
        wet = np.zeros_like(bands['blue'])
        for band_name, coeff in coeffs.items():
            if band_name in bands:
                wet += coeff * bands[band_name]
        return wet

    @staticmethod
    def calculate_lst_simple(tir: np.ndarray) -> np.ndarray:
        if np.nanmax(tir) > 400:
            ML = 3.3420E-04
            AL = 0.10000
            K1 = 774.8853
            K2 = 1321.0789
            L_lambda = ML * tir + AL
            with np.errstate(divide='ignore', invalid='ignore'):
                T_b = K2 / np.log(K1 / L_lambda + 1)
        else:
            T_b = tir
        return T_b - 273.15


class RSEICalculator:
    def __init__(self, config: RSEIConfig = None):
        self.config = config or RSEIConfig()
        self.indices = {}
        self.rsei = None
        self.rsei_components = None

    def normalize(self, array: np.ndarray, inverse: bool = False) -> np.ndarray:
        valid_mask = ~np.isnan(array)
        if not valid_mask.any():
            return array

        min_val = np.nanmin(array)
        max_val = np.nanmax(array)

        if max_val == min_val:
            return np.ones_like(array) * 0.5

        normalized = (array - min_val) / (max_val - min_val)
        if inverse:
            normalized = 1 - normalized
        return normalized

    def apply_water_mask(self, array: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        masked_array = array.copy()
        masked_array[water_mask] = np.nan
        return masked_array

    def calculate_rsei_pca(self, greenness: np.ndarray, wetness: np.ndarray,
                           dryness: np.ndarray, heat: np.ndarray,
                           water_mask: Optional[np.ndarray] = None) -> np.ndarray:
        st.write("🔬 PCA计算中...")

        if water_mask is not None:
            greenness = self.apply_water_mask(greenness, water_mask)
            wetness = self.apply_water_mask(wetness, water_mask)
            dryness = self.apply_water_mask(dryness, water_mask)
            heat = self.apply_water_mask(heat, water_mask)

        green_norm = self.normalize(greenness, inverse=False)
        wet_norm = self.normalize(wetness, inverse=False)
        dry_norm = self.normalize(dryness, inverse=True)
        heat_norm = self.normalize(heat, inverse=True)

        mask = ~(np.isnan(green_norm) | np.isnan(wet_norm) |
                 np.isnan(dry_norm) | np.isnan(heat_norm))

        n_valid = mask.sum()
        st.write(f"  有效像素: {n_valid:,}")

        if n_valid < 100:
            raise ValueError("有效像素太少")

        data_matrix = np.column_stack([
            green_norm[mask], wet_norm[mask],
            dry_norm[mask], heat_norm[mask]
        ])

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)

        pca = PCA(n_components=4)
        pc_all = pca.fit_transform(data_scaled)
        pc1 = pc_all[:, 0]

        rsei_raw = np.full(greenness.shape, np.nan)
        rsei_raw[mask] = pc1.flatten()
        rsei = self.normalize(rsei_raw, inverse=False)

        st.write(f"  PC1贡献: {pca.explained_variance_ratio_[0] * 100:.2f}%")

        self.rsei_components = {
            'greenness': green_norm,
            'wetness': wet_norm,
            'dryness': dry_norm,
            'heat': heat_norm,
            'pca': {
                'variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist()
            }
        }

        return rsei

    def classify_rsei(self, rsei: np.ndarray) -> np.ndarray:
        classified = np.full_like(rsei, np.nan)
        classified[rsei < 0.2] = 1
        classified[(rsei >= 0.2) & (rsei < 0.4)] = 2
        classified[(rsei >= 0.4) & (rsei < 0.6)] = 3
        classified[(rsei >= 0.6) & (rsei < 0.8)] = 4
        classified[rsei >= 0.8] = 5
        return classified


# =============================
# Streamlit主程序
# =============================
def main():
    st.title("🌿 RSEI计算系统 v3.2")
    st.markdown("**Remote Sensing Ecological Index Calculator with OTSU Auto-Threshold**")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 参数配置")

        # 显示字体状态
        st.info(font_status)

        st.subheader("📁 文件")
        uploaded_file = st.file_uploader("上传多波段TIF", type=['tif', 'tiff'])
        output_dir = st.text_input("输出目录", value="./rsei_results")

        st.markdown("---")
        st.subheader("🛰️ 卫星")
        satellite = st.radio("类型", ["Landsat8", "Sentinel2"])

        st.markdown("---")
        st.subheader("🔬 方法")
        use_pca = st.checkbox("PCA方法", value=True)

        st.markdown("---")
        st.subheader("💧 水体掩膜")
        mask_water = st.checkbox("去除水域", value=True)

        if mask_water:
            water_index = st.selectbox("水体指数", ["MNDWI", "NDWI", "AWEIsh"])
            use_otsu = st.checkbox("OTSU自动阈值（推荐）", value=True)

            if not use_otsu:
                water_threshold = st.slider("手动阈值", -1.0, 1.0, 0.0, 0.1)
            else:
                water_threshold = None
        else:
            water_index = "MNDWI"
            water_threshold = None
            use_otsu = False

        st.markdown("---")
        st.subheader("💾 导出")
        export_geotiff = st.checkbox("GeoTIFF", value=True)
        export_indices = st.checkbox("所有指数", value=True)

        st.markdown("---")
        run_button = st.button("▶️ 开始计算", type="primary", use_container_width=True)

    # 主界面
    tab1, tab2, tab3, tab4 = st.tabs(["📊 结果", "ℹ️ 说明", "📖 关于", "🔍 字体测试"])

    with tab2:
        st.markdown("""
        ### 🌟 OTSU自动阈值功能

        **优势**:
        - 自动计算最优水体分割阈值
        - 无需人工干预
        - 适用于水陆分界明显的区域

        **使用方法**:
        1. 上传影像
        2. 勾选"去除水域"和"OTSU自动阈值"
        3. 点击"开始计算"
        4. 系统自动分析并应用最优阈值

        **注意**:
        - 推荐使用 MNDWI + OTSU 组合
        - 水域占比极端情况可能需手动调整
        - 计算完成后可查看OTSU分析图
        """)

    with tab3:
        st.markdown("""
        ### 📚 关于RSEI

        **Remote Sensing Ecological Index**

        由四个指标构成:
        - 🌱 绿度 (NDVI)
        - 💧 湿度 (WET)
        - 🏜️ 干度 (NDBSI)
        - 🌡️ 热度 (LST)

        **OTSU算法**:

        大津算法通过最大化类间方差自动确定最优阈值，
        广泛应用于图像分割领域。

        ---

        **版本**: v3.2  
        **更新**: 中文字体完美支持  
        **支持**: Landsat 8, Sentinel-2
        """)

    with tab4:
        st.markdown("### 🔍 字体测试")
        st.write(f"**当前字体配置:** {plt.rcParams['font.sans-serif']}")
        st.write(f"**负号显示:** {plt.rcParams['axes.unicode_minus']}")

        # 绘制测试图
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            test_data = [0.65, 0.72, 0.58, 0.81]
            test_labels = ['绿度', '湿度', '干度', '热度']
            colors_test = ['#9ACD32', '#4169E1', '#D2691E', '#FF6347']

            bars = ax.bar(test_labels, test_data, color=colors_test, alpha=0.8, edgecolor='black')
            ax.set_xlabel('RSEI指标', fontsize=12, fontweight='bold')
            ax.set_ylabel('归一化值', fontsize=12, fontweight='bold')
            ax.set_title('中文字体测试 - 柱状图', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            test_pie_data = [15, 25, 35, 20, 5]
            test_pie_labels = ['差', '较差', '中等', '良好', '优秀']
            colors_pie = ['#8B0000', '#FF4500', '#FFD700', '#9ACD32', '#006400']

            ax.pie(test_pie_data, labels=test_pie_labels, colors=colors_pie,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
            ax.set_title('中文字体测试 - 饼图', fontsize=14, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if font_status:
            st.success(f"✅ 字体配置成功！当前使用: **{font_status}**")
            st.info("如果上方图表中文显示正常，说明字体配置完美！")
        else:
            st.warning("⚠️ 未找到理想的中文字体，使用系统默认字体")
            st.info("建议在 Streamlit Cloud 部署时添加 packages.txt 文件")

    # 执行计算
    if run_button:
        if not uploaded_file:
            st.error("⚠️ 请上传影像！")
        else:
            with tab1:
                try:
                    config = RSEIConfig(
                        satellite=satellite,
                        use_pca=use_pca,
                        export_indices=export_indices,
                        export_geotiff=export_geotiff,
                        mask_water=mask_water,
                        water_index=water_index,
                        water_threshold=water_threshold,
                        use_otsu=use_otsu
                    )

                    temp_file = Path("temp_input.tif")
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.read())

                    with st.spinner("🔄 计算中..."):
                        result = execute_rsei(str(temp_file), output_dir, config)

                    display_results(result)

                    if temp_file.exists():
                        temp_file.unlink()

                    st.balloons()

                except Exception as e:
                    st.error(f"❌ 失败: {str(e)}")
                    with st.expander("详细错误"):
                        import traceback
                        st.code(traceback.format_exc())


def execute_rsei(input_file, output_dir, config):
    """执行RSEI计算"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 读取
    status_text.text("📁 读取影像...")
    progress_bar.progress(10)

    reader = MultiSpectralImageReader(config)
    bands = reader.read_multiband_tif(input_file)

    # 预处理
    status_text.text("⚙️ 预处理...")
    progress_bar.progress(20)

    max_val = np.nanmax(bands['red'])
    if max_val > 1.0:
        bands = reader.apply_scale_factor(bands, 0.0001)

    # 水体
    water_index = None
    water_mask = None
    water_threshold_used = None

    if config.mask_water:
        status_text.text("💧 水体识别...")
        progress_bar.progress(30)
        water_index, water_mask, water_threshold_used = WaterMaskGenerator.create_water_mask(
            bands, config.water_index, config.water_threshold, config.use_otsu
        )

    # 指数
    status_text.text("🔬 计算指数...")
    progress_bar.progress(50)

    calc = RemoteSensingIndices()
    ndvi = calc.calculate_ndvi(bands['red'], bands['nir'])
    wet = calc.calculate_wet(bands, config.satellite)
    ndbsi = calc.calculate_ndbsi(bands)

    if 'tir' in bands and bands['tir'] is not None:
        lst = calc.calculate_lst_simple(bands['tir'])
    else:
        lst = ndbsi

    indices = {'ndvi': ndvi, 'wet': wet, 'ndbsi': ndbsi, 'lst': lst}

    # RSEI
    status_text.text("🌍 计算RSEI...")
    progress_bar.progress(70)

    rsei_calc = RSEICalculator(config)
    rsei = rsei_calc.calculate_rsei_pca(ndvi, wet, ndbsi, lst, water_mask)

    # 分类
    status_text.text("📊 分类...")
    progress_bar.progress(85)

    rsei_class = rsei_calc.classify_rsei(rsei)

    # 导出
    status_text.text("💾 导出...")
    progress_bar.progress(95)

    if config.export_geotiff and reader.metadata:
        try:
            with rasterio.open(output_path / 'RSEI.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei.astype('float32'), 1)

            with rasterio.open(output_path / 'RSEI_classified.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei_class.astype('float32'), 1)
        except Exception as e:
            st.warning(f"⚠️ GeoTIFF导出失败: {e}")

    # 统计
    class_names = ['差', '较差', '中等', '良好', '优秀']
    total_valid = np.sum(~np.isnan(rsei_class))
    class_stats = {
        name: int(np.sum(rsei_class == i))
        for i, name in enumerate(class_names, 1)
    }

    progress_bar.progress(100)
    status_text.text("✅ 完成")

    return {
        'indices': indices,
        'rsei': rsei,
        'rsei_class': rsei_class,
        'water_index': water_index,
        'water_mask': water_mask,
        'water_threshold': water_threshold_used,
        'class_stats': class_stats,
        'metadata': reader.metadata,
        'output_path': output_path
    }


def display_results(result):
    """显示结果"""

    st.header("📊 RSEI分析结果")

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)

    rsei = result['rsei']
    with col1:
        st.metric("均值", f"{np.nanmean(rsei):.4f}")
    with col2:
        st.metric("中位数", f"{np.nanmedian(rsei):.4f}")
    with col3:
        st.metric("标准差", f"{np.nanstd(rsei):.4f}")
    with col4:
        mean_val = np.nanmean(rsei)
        quality = "优秀⭐⭐⭐⭐⭐" if mean_val >= 0.8 else "良好⭐⭐⭐⭐" if mean_val >= 0.6 else "中等⭐⭐⭐"
        st.metric("等级", quality)

    st.markdown("---")

    # 等级分布
    st.subheader("📈 等级分布")

    class_df = pd.DataFrame({
        '等级': list(result['class_stats'].keys()),
        '像素数': list(result['class_stats'].values())
    })
    class_df['百分比'] = (class_df['像素数'] / class_df['像素数'].sum() * 100).round(2)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(class_df, use_container_width=True)

    with col2:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#8B0000', '#FF4500', '#FFD700', '#9ACD32', '#006400']
            wedges, texts, autotexts = ax.pie(
                class_df['像素数'],
                labels=class_df['等级'],
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11}
            )
            # 设置百分比文字为白色加粗
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.set_title('RSEI等级分布', fontsize=13, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 饼图绘制失败: {e}")

    st.markdown("---")

    # 空间分布
    st.subheader("🗺️ 空间分布")

    tab_ndvi, tab_wet, tab_dry, tab_heat, tab_water, tab_rsei, tab_class = st.tabs([
        "🌱 NDVI", "💧 WET", "🏜️ NDBSI", "🌡️ LST", "💦 水体", "🌿 RSEI", "📊 等级"
    ])

    with tab_ndvi:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result['indices']['ndvi'], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            ax.set_title('NDVI (绿度)', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI值', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 绘图失败: {e}")

    with tab_wet:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result['indices']['wet'], cmap='Blues')
            ax.set_title('WET (湿度)', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('湿度值', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 绘图失败: {e}")

    with tab_dry:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result['indices']['ndbsi'], cmap='YlOrBr')
            ax.set_title('NDBSI (干度)', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('干度值', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 绘图失败: {e}")

    with tab_heat:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result['indices']['lst'], cmap='hot')
            ax.set_title('LST (热度)', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('温度(°C)', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 绘图失败: {e}")

    with tab_water:
        if result['water_index'] is not None:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # 水体指数
                im1 = ax1.imshow(result['water_index'], cmap='RdYlBu', vmin=-0.5, vmax=0.5)
                threshold = result['water_threshold'] if result['water_threshold'] else 0.0
                ax1.set_title(f'水体指数 (OTSU阈值={threshold:.4f})',
                              fontsize=14, fontweight='bold', pad=15)
                ax1.axis('off')
                cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                cbar1.set_label('指数值', fontsize=11)
                cbar1.ax.axhline(threshold, color='red', linestyle='--', linewidth=2)

                # 水体掩膜
                water_display = np.where(result['water_mask'], 1, 0).astype(float)
                water_display[np.isnan(result['indices']['ndvi'])] = np.nan
                cmap_water = ListedColormap(['#8B4513', '#4169E1'])
                im2 = ax2.imshow(water_display, cmap=cmap_water, vmin=0, vmax=1)
                ax2.set_title('水体掩膜', fontsize=14, fontweight='bold', pad=15)
                ax2.axis('off')

                legend_elements = [
                    Patch(facecolor='#8B4513', label='陆地'),
                    Patch(facecolor='#4169E1', label='水域')
                ]
                ax2.legend(handles=legend_elements, loc='upper right',
                           fontsize=11, framealpha=0.9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"❌ 水体图绘制失败: {e}")
        else:
            st.info("ℹ️ 未启用水体掩膜")

    with tab_rsei:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_rsei = ['#8B0000', '#FF4500', '#FFD700', '#9ACD32', '#006400']
            cmap_rsei = LinearSegmentedColormap.from_list('RSEI', colors_rsei, N=256)
            im = ax.imshow(result['rsei'], cmap=cmap_rsei, vmin=0, vmax=1)
            ax.set_title('RSEI 生态指数', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('生态指数', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ RSEI图绘制失败: {e}")

    with tab_class:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_class = ['#8B0000', '#FF4500', '#FFD700', '#9ACD32', '#006400']
            cmap_class = LinearSegmentedColormap.from_list('RSEI_class', colors_class, N=5)
            im = ax.imshow(result['rsei_class'], cmap=cmap_class, vmin=1, vmax=5)
            ax.set_title('RSEI等级分类', fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[1, 2, 3, 4, 5])
            cbar.ax.set_yticklabels(['差', '较差', '中等', '良好', '优秀'], fontsize=10)
            cbar.set_label('生态等级', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ 等级图绘制失败: {e}")

    st.markdown("---")

    # 下载
    st.subheader("📥 结果文件")
    st.success(f"✅ 已保存到: `{result['output_path'].absolute()}`")

    # 显示文件列表
    if result['output_path'].exists():
        files = list(result['output_path'].glob('*.tif'))
        if files:
            st.write("**生成的文件:**")
            for f in files:
                st.write(f"  - {f.name}")

#     下载：
    # 下载区域（新增/修改这部分）
    st.subheader("📥 结果文件下载")
    st.success(f"文件存储路径: `{result['output_path'].absolute()}`")

    # 获取目录下的所有文件
    output_dir = result['output_path']
    if output_dir.exists():
        # 过滤出常见的结果文件（可根据实际生成的文件类型调整）
        result_files = list(output_dir.glob('*.tif')) + list(output_dir.glob('*.txt')) + list(output_dir.glob('*.csv'))

        if result_files:
            st.write("可下载的文件:")
            for file in result_files:
                # 读取文件内容（二进制模式，适用于所有文件类型）
                with open(file, "rb") as f:
                    file_content = f.read()

                # 创建下载按钮（文件名作为按钮文字，方便用户识别）
                st.download_button(
                    label=f"下载 {file.name}",
                    data=file_content,
                    file_name=file.name,  # 保存到本地的文件名
                    mime="application/octet-stream",  # 通用二进制文件类型
                    use_container_width=True
                )
        else:
            st.info("⚠️ 该目录下暂无结果文件，请先运行计算生成结果。")
    else:
        st.error(f"❌ 目录不存在: {output_dir}")


if __name__ == "__main__":
    main()

## 运行方式

# ```bash
# streamlit run RSEI_APP.py
# ```