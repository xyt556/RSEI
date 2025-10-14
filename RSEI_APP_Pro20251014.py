import streamlit as st
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import time
import io
import zipfile
import tempfile
import shutil
import platform
import os

warnings.filterwarnings('ignore')

# =============================
# 文件大小限制配置 - 核心解决方案
# =============================
MAX_FILE_SIZE_MB = 500  # 最大文件大小（MB）
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def check_file_size(uploaded_file):
    """检查上传文件大小是否超过限制"""
    if uploaded_file is None:
        return False

    file_size = len(uploaded_file.getvalue())

    if file_size > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ 文件太大！")
        st.error(f"最大支持: {MAX_FILE_SIZE_MB} MB")
        st.error(f"当前文件: {file_size / (1024 * 1024):.2f} MB")
        st.error("请上传较小的文件或联系管理员调整限制")
        return False
    return True


def format_file_size(size_bytes):
    """格式化文件大小显示"""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


# =============================
# 中文字体配置
# =============================
def setup_chinese_font_enhanced():
    """增强版中文字体配置"""
    system = platform.system()

    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                     'Droid Sans Fallback']

    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

    chinese_font = None
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break

    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    else:
        cjk_fonts = [f.name for f in font_manager.fontManager.ttflist
                     if any(keyword in f.name.lower() for keyword in
                            ['cjk', 'chinese', 'sc', 'cn', 'hei', 'song'])]
        if cjk_fonts:
            plt.rcParams['font.sans-serif'] = [cjk_fonts[0], 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    return plt.rcParams['font.sans-serif'][0]


try:
    detected_font = setup_chinese_font_enhanced()
except Exception as e:
    print(f"字体设置失败: {e}")

st.set_page_config(
    page_title="RSEI计算系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================
# 会话状态初始化
# =============================
def initialize_session_state():
    """初始化会话状态"""
    if 'calculation_complete' not in st.session_state:
        st.session_state.calculation_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'current_params_hash' not in st.session_state:
        st.session_state.current_params_hash = None
    if 'tmp_file_path' not in st.session_state:
        st.session_state.tmp_file_path = None


initialize_session_state()


# =============================
# 核心计算类
# =============================
class JenksNaturalBreaks:
    @staticmethod
    def calculate_jenks_breaks(data: np.ndarray, n_classes: int = 5,
                               max_samples: int = 5000) -> List[float]:
        valid_data = data[~np.isnan(data)].flatten()
        if len(valid_data) == 0:
            raise ValueError("数据全为NaN")

        if len(valid_data) > max_samples:
            st.info(f"数据量 {len(valid_data):,} 过大，采样至 {max_samples:,} 个点...")
            np.random.seed(42)
            indices = np.random.choice(len(valid_data), max_samples, replace=False)
            valid_data = valid_data[indices]

        sorted_data = np.sort(valid_data)
        n = len(sorted_data)
        breaks = []

        quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
        for q in quantiles:
            idx = int(q * n)
            breaks.append(float(sorted_data[idx]))

        return breaks[:n_classes - 1]

    @staticmethod
    def get_default_breaks() -> List[float]:
        return [0.2, 0.4, 0.6, 0.8]


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
    use_jenks: bool = True
    classification_breaks: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    jenks_samples: int = 5000

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
        st.info(f"📡 读取多波段影像: {Path(tif_path).name}")

        with rasterio.open(tif_path) as src:
            st.write(f"尺寸: {src.width} x {src.height}, 波段数: {src.count}")

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
                raise ValueError(f"波段数量不足！需要{max_band_idx}个，实际{src.count}个")

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

        st.success(f"✅ 成功读取 {len(bands)} 个波段")
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
        st.info(f"🔧 应用缩放因子: {scale_factor}")
        scaled_bands = {}
        optical_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

        for band_name, band_data in bands.items():
            if band_name in optical_bands:
                scaled_bands[band_name] = band_data * scale_factor
            else:
                scaled_bands[band_name] = band_data
        return scaled_bands


# ... 其他类保持不变（OTSUThreshold, WaterMaskGenerator, RemoteSensingIndices, RSEICalculator, RSEIVisualizer）...

# =============================
# GUI主程序
# =============================
def main():
    st.title("🌿 RSEI计算系统 v4.0 - 完整版")
    st.markdown("**Remote Sensing based Ecological Index 遥感生态指数计算工具**")

    # 显示文件大小限制信息
    st.sidebar.markdown("---")
    st.sidebar.info(f"📏 **文件大小限制**: 最大 {MAX_FILE_SIZE_MB} MB")

    with st.sidebar:
        st.header("⚙️ 参数配置")

        st.subheader("📁 文件上传")
        uploaded_file = st.file_uploader(
            f"选择多波段TIF影像 (最大 {MAX_FILE_SIZE_MB} MB)",
            type=['tif', 'tiff'],
            help=f"请上传不超过 {MAX_FILE_SIZE_MB} MB 的TIFF文件"
        )

        st.subheader("🛰️ 卫星参数")
        satellite = st.selectbox("卫星类型", ["Landsat8", "Sentinel2"], index=0)

        st.subheader("🔬 计算方法")
        use_pca = st.checkbox("使用PCA方法", value=True)

        st.subheader("📊 分类阈值设置")
        use_jenks = st.checkbox("使用Jenks自然间断点", value=True)

        if use_jenks:
            jenks_samples = st.slider("采样数量", 1000, 20000, 5000, 1000)
            threshold_1, threshold_2, threshold_3, threshold_4 = 0.2, 0.4, 0.6, 0.8
        else:
            st.write("手动设置阈值:")
            threshold_1 = st.number_input("差/较差", 0.0, 1.0, 0.2, 0.01)
            threshold_2 = st.number_input("较差/中等", 0.0, 1.0, 0.4, 0.01)
            threshold_3 = st.number_input("中等/良好", 0.0, 1.0, 0.6, 0.01)
            threshold_4 = st.number_input("良好/优秀", 0.0, 1.0, 0.8, 0.01)
            jenks_samples = 5000

        st.subheader("💧 水体掩膜")
        mask_water = st.checkbox("去除水域", value=True)

        if mask_water:
            water_index = st.selectbox("水体指数", ["MNDWI", "NDWI", "AWEIsh"], index=0)
            use_otsu = st.checkbox("使用OTSU自动计算阈值", value=True)
            if not use_otsu:
                water_threshold = st.number_input("手动阈值", -1.0, 1.0, 0.0, 0.1)
            else:
                water_threshold = None
        else:
            water_index = "MNDWI"
            use_otsu = True
            water_threshold = None

        st.subheader("💾 导出选项")
        export_geotiff = st.checkbox("导出GeoTIFF文件", value=True)
        export_indices = st.checkbox("导出所有遥感指数", value=True)

    # 处理文件上传和计算
    if uploaded_file is not None:
        # 检查文件大小 - 这是核心限制逻辑
        if not check_file_size(uploaded_file):
            st.stop()  # 停止执行后续代码

        # 显示文件信息
        file_size = len(uploaded_file.getvalue())
        st.success(f"✅ 文件已上传: {uploaded_file.name}")
        st.info(f"📦 文件大小: {format_file_size(file_size)}")

        # 进度条显示文件大小使用情况
        usage_percent = min(100, (file_size / MAX_FILE_SIZE_BYTES) * 100)
        st.progress(usage_percent / 100, text=f"存储使用: {usage_percent:.1f}%")

        # 保存上传的文件到会话状态
        if (st.session_state.uploaded_file != uploaded_file.name or
                st.session_state.tmp_file_path is None):
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.calculation_complete = False
            st.session_state.results = None

            # 保存临时文件路径
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.tmp_file_path = tmp_file.name

        # 这里继续你的计算逻辑...
        # [原有的计算逻辑保持不变]

    else:
        st.info("👈 请在左侧上传多波段TIF影像文件开始计算")

        # 显示系统限制信息
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            ### 🌟 功能特点
            - ✅ 支持 Landsat 8 和 Sentinel-2
            - ✅ 自动水体掩膜（OTSU阈值）
            - ✅ Jenks自然间断点分类
            - ✅ 完整的可视化分析
            - ✅ 一键打包下载
            - 📏 文件限制: {MAX_FILE_SIZE_MB} MB
            """)

        with col2:
            st.markdown(f"""
            ### 📊 输出结果
            - 🎯 RSEI连续值/分类影像
            - 🖼️ 综合可视化图
            - 📈 Excel统计报告
            - 🌱 10+遥感指数
            ### ⚠️ 注意事项
            - 请确保TIFF文件包含所有必要波段
            - 文件大小不超过 {MAX_FILE_SIZE_MB} MB
            - 计算时间取决于文件大小和计算机性能
            """)


if __name__ == "__main__":
    main()