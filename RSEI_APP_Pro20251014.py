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
import urllib.request

warnings.filterwarnings('ignore')


# =============================
# 彻底解决中文字体问题
# =============================
def download_chinese_font():
    """
    下载开源中文字体（思源黑体）
    """
    font_dir = Path.home() / '.fonts'
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / 'SourceHanSansSC-Regular.otf'

    if not font_path.exists():
        try:
            st.info("首次使用，正在下载中文字体...")
            # 使用GitHub镜像下载思源黑体
            font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
            urllib.request.urlretrieve(font_url, font_path)
            st.success("字体下载完成！")
            return font_path
        except Exception as e:
            st.warning(f"字体下载失败: {e}")
            return None
    return font_path


def setup_chinese_font_enhanced():
    """
    增强版中文字体配置 - 多重方案确保成功
    """
    # 方案1: 尝试下载和使用思源黑体
    downloaded_font = download_chinese_font()

    # 清除matplotlib字体缓存
    try:
        import shutil
        cache_dir = matplotlib.get_cachedir()
        if os.path.exists(cache_dir):
            font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
            if os.path.exists(font_cache):
                os.remove(font_cache)
    except Exception as e:
        pass

    # 重新加载字体
    font_manager._load_fontmanager(try_read_cache=False)

    # 获取系统类型
    system = platform.system()

    # 根据不同操作系统设置字体优先级
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'Microsoft YaHei UI']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'Songti SC']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                     'Noto Sans CJK JP', 'Droid Sans Fallback', 'AR PL UMing CN', 'Source Han Sans SC']

    # 如果下载了字体，添加到列表首位
    if downloaded_font and downloaded_font.exists():
        font_list.insert(0, str(downloaded_font))

    # 检测系统中可用的字体
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

    # 找到第一个可用的中文字体
    chinese_font = None
    font_path = None

    for font in font_list:
        # 如果是路径，直接使用
        if os.path.exists(str(font)):
            font_path = font
            chinese_font = font
            break
        # 如果是字体名称，检查是否可用
        elif font in available_fonts:
            chinese_font = font
            break

    # 配置matplotlib
    if chinese_font:
        if font_path:
            # 如果是字体文件路径，添加到matplotlib
            font_manager.fontManager.addfont(str(font_path))
            font_prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False

        # 强制设置所有文本相关参数
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9

        return chinese_font
    else:
        # 最后的备选方案
        plt.rcParams['axes.unicode_minus'] = False

        # 尝试从所有可用字体中找CJK字体
        cjk_fonts = [f.name for f in font_manager.fontManager.ttflist
                     if any(keyword in f.name.lower() for keyword in
                            ['cjk', 'chinese', 'sc', 'cn', 'hei', 'song', 'kai',
                             'pingfang', 'microsoft', 'simhei', 'wenquanyi', 'noto', 'source'])]

        if cjk_fonts:
            plt.rcParams['font.sans-serif'] = [cjk_fonts[0], 'DejaVu Sans']
            return cjk_fonts[0]
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            st.warning("⚠️ 未找到中文字体，中文可能显示为方框")
            return 'DejaVu Sans (无中文支持)'


# 在程序开始时设置字体
try:
    detected_font = setup_chinese_font_enhanced()
    print(f"✓ 使用字体: {detected_font}")
except Exception as e:
    print(f"✗ 字体设置失败: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="RSEI计算系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================
# 核心计算类（保持不变）
# =============================
class JenksNaturalBreaks:
    """优化的Jenks自然间断点分类算法"""

    @staticmethod
    def calculate_jenks_breaks(data: np.ndarray, n_classes: int = 5,
                               max_samples: int = 5000) -> List[float]:
        valid_data = data[~np.isnan(data)].flatten()
        if len(valid_data) == 0:
            raise ValueError("数据全为NaN")

        if len(valid_data) > max_samples:
            st.info(f"数据量 {len(valid_data):,} 过大，采样至 {max_samples:,} 个点...")
            valid_data = JenksNaturalBreaks._stratified_sample(valid_data, max_samples)

        methods = [
            ('jenkspy', JenksNaturalBreaks._jenks_by_jenkspy),
            ('numba', JenksNaturalBreaks._jenks_by_numba),
            ('optimized_numpy', JenksNaturalBreaks._jenks_optimized_numpy),
            ('fallback', JenksNaturalBreaks._jenks_fallback)
        ]

        for method_name, method_func in methods:
            try:
                breaks = method_func(valid_data, n_classes)
                if breaks is not None:
                    st.success(f"使用方法: {method_name}")
                    return breaks
            except Exception as e:
                if method_name != 'fallback':
                    st.warning(f"{method_name} 方法失败: {e}")
                    continue
                else:
                    raise

        return JenksNaturalBreaks.get_default_breaks()

    @staticmethod
    def _stratified_sample(data: np.ndarray, n_samples: int) -> np.ndarray:
        sorted_data = np.sort(data)
        indices = np.linspace(0, len(sorted_data) - 1, n_samples, dtype=int)
        np.random.seed(42)
        noise = np.random.randint(-2, 3, size=n_samples)
        indices = np.clip(indices + noise, 0, len(sorted_data) - 1)
        return sorted_data[indices]

    @staticmethod
    def _jenks_by_jenkspy(data: np.ndarray, n_classes: int) -> Optional[List[float]]:
        try:
            import jenkspy
            breaks = jenkspy.jenks_breaks(data, n_classes=n_classes)
            return breaks[1:-1]
        except ImportError:
            return None

    @staticmethod
    def _jenks_by_numba(data: np.ndarray, n_classes: int) -> Optional[List[float]]:
        try:
            from numba import jit

            @jit(nopython=True)
            def _jenks_matrices_numba(data, n_classes):
                n_data = len(data)
                mat1 = np.zeros((n_data + 1, n_classes + 1), dtype=np.float64)
                mat2 = np.zeros((n_data + 1, n_classes + 1), dtype=np.float64)

                for i in range(1, n_classes + 1):
                    mat1[1, i] = 1
                    mat2[1, i] = 0
                    for j in range(2, n_data + 1):
                        mat2[j, i] = np.inf

                for l in range(2, n_data + 1):
                    s1 = 0.0
                    s2 = 0.0

                    for m in range(1, l + 1):
                        i3 = l - m + 1
                        val = data[i3 - 1]
                        s2 += val * val
                        s1 += val
                        w = m
                        v = s2 - (s1 * s1) / w
                        i4 = i3 - 1

                        if i4 != 0:
                            for j in range(2, n_classes + 1):
                                if mat2[l, j] >= (v + mat2[i4, j - 1]):
                                    mat1[l, j] = i3
                                    mat2[l, j] = v + mat2[i4, j - 1]

                    mat1[l, 1] = 1
                    mat2[l, 1] = v

                return mat1, mat2

            sorted_data = np.sort(data)
            mat1, mat2 = _jenks_matrices_numba(sorted_data, n_classes)

            k = len(sorted_data)
            kclass = []

            for j in range(n_classes, 0, -1):
                idx = int(mat1[k, j]) - 2
                if idx >= 0 and idx < len(sorted_data):
                    kclass.append(sorted_data[idx])
                k = int(mat1[k, j]) - 1

            kclass.reverse()
            if len(kclass) > 1:
                return kclass[1:][:n_classes - 1]
            else:
                return None

        except ImportError:
            return None

    @staticmethod
    def _jenks_optimized_numpy(data: np.ndarray, n_classes: int) -> Optional[List[float]]:
        sorted_data = np.sort(data)
        n_data = len(sorted_data)

        lower_class_limits = np.zeros((n_data + 1, n_classes + 1), dtype=np.int32)
        variance_combinations = np.zeros((n_data + 1, n_classes + 1), dtype=np.float64)
        variance_combinations[:, :] = np.inf

        lower_class_limits[1:, 1] = 1
        variance_combinations[1, :] = 0

        for l in range(2, n_data + 1):
            sum_val = 0.0
            sum_sq = 0.0

            for m in range(1, l + 1):
                lower_class_limit = l - m + 1
                val = sorted_data[lower_class_limit - 1]

                sum_val += val
                sum_sq += val * val
                w = m

                variance = sum_sq - (sum_val * sum_val) / w

                if lower_class_limit > 1:
                    for j in range(2, n_classes + 1):
                        if variance_combinations[l, j] >= (
                                variance + variance_combinations[lower_class_limit - 1, j - 1]):
                            lower_class_limits[l, j] = lower_class_limit
                            variance_combinations[l, j] = variance + variance_combinations[
                                lower_class_limit - 1, j - 1]

            lower_class_limits[l, 1] = 1
            variance_combinations[l, 1] = variance

        k = n_data
        breaks = []

        for j in range(n_classes, 1, -1):
            idx = lower_class_limits[k, j] - 2
            if 0 <= idx < len(sorted_data):
                breaks.append(float(sorted_data[idx]))
            k = lower_class_limits[k, j] - 1

        breaks.reverse()

        if len(breaks) >= n_classes - 1:
            return breaks[:n_classes - 1]
        else:
            return None

    @staticmethod
    def _jenks_fallback(data: np.ndarray, n_classes: int) -> List[float]:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        breaks = []

        quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]

        for q in quantiles:
            idx = int(q * n)
            window = min(int(n * 0.05), 100)
            start = max(0, idx - window)
            end = min(n, idx + window)

            if start >= end - 1:
                breaks.append(sorted_data[idx])
                continue

            min_variance = np.inf
            best_idx = idx

            for i in range(start, end):
                if i == 0 or i == n:
                    continue

                left_var = np.var(sorted_data[max(0, i - window):i]) if i > window else 0
                right_var = np.var(sorted_data[i:min(n, i + window)]) if i < n - window else 0
                total_var = left_var + right_var

                if total_var < min_variance:
                    min_variance = total_var
                    best_idx = i

            breaks.append(float(sorted_data[best_idx]))

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
        st.info(f"💧 创建水体掩膜 (方法: {method})")

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
            raise ValueError(f"不支持的水体指数: {method}")

        if use_otsu and threshold is None:
            try:
                otsu_threshold, metrics = OTSUThreshold.calculate_otsu_threshold(water_index, bins=256)
                final_threshold = otsu_threshold
                st.success(f"✓ OTSU阈值: {final_threshold:.4f}")
            except Exception as e:
                st.warning(f"⚠️ OTSU失败: {e}, 使用默认阈值0.0")
                final_threshold = 0.0
        elif threshold is not None:
            final_threshold = threshold
        else:
            final_threshold = 0.0

        water_mask = water_index > final_threshold

        total_pixels = np.sum(~np.isnan(water_index))
        water_pixels = np.sum(water_mask)
        water_ratio = water_pixels / total_pixels * 100 if total_pixels > 0 else 0

        st.write(f"水域: {water_pixels:,} ({water_ratio:.2f}%)")

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
        self.calculated_breaks = None
        self.jenks_time = 0

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
        st.info("🔬 计算RSEI (PCA方法)...")

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
        st.write(f"有效像素: {n_valid:,}")

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

        st.success(f"PC1贡献率: {pca.explained_variance_ratio_[0] * 100:.2f}%")

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

        if self.config.use_jenks:
            st.info("🔍 计算Jenks自然间断点阈值...")
            start_time = time.time()
            try:
                breaks = JenksNaturalBreaks.calculate_jenks_breaks(
                    rsei,
                    n_classes=5,
                    max_samples=self.config.jenks_samples
                )
                self.calculated_breaks = breaks
                self.jenks_time = time.time() - start_time
                st.success(f"✓ Jenks阈值: {[f'{b:.4f}' for b in breaks]}")
                st.success(f"✓ 计算耗时: {self.jenks_time:.2f}秒")
            except Exception as e:
                st.warning(f"⚠️ Jenks计算失败: {e}, 使用默认阈值")
                self.calculated_breaks = JenksNaturalBreaks.get_default_breaks()
                self.jenks_time = 0
        else:
            self.calculated_breaks = self.config.classification_breaks

        return rsei

    def classify_rsei(self, rsei: np.ndarray, breaks: Optional[List[float]] = None) -> np.ndarray:
        if breaks is None:
            breaks = self.calculated_breaks or self.config.classification_breaks

        classified = np.full_like(rsei, np.nan)

        if len(breaks) < 4:
            breaks = JenksNaturalBreaks.get_default_breaks()

        t1, t2, t3, t4 = breaks[0], breaks[1], breaks[2], breaks[3]

        classified[rsei < t1] = 1
        classified[(rsei >= t1) & (rsei < t2)] = 2
        classified[(rsei >= t2) & (rsei < t3)] = 3
        classified[(rsei >= t3) & (rsei < t4)] = 4
        classified[rsei >= t4] = 5

        return classified


class RSEIVisualizer:
    @staticmethod
    def create_comprehensive_visualization(rsei, rsei_class, indices,
                                           water_index=None, water_threshold=None,
                                           classification_breaks=None):
        """创建综合可视化图 - 彻底修复中文显示"""

        # 重新确认字体设置（在绘图前）
        try:
            setup_chinese_font_enhanced()
        except:
            pass

        rsei_colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
        rsei_cmap = ListedColormap(rsei_colors)

        # 创建图形
        fig = plt.figure(figsize=(22, 13))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        # 1. RSEI连续值
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(rsei, cmap='RdYlGn', vmin=0, vmax=1)
        ax1.set_title('RSEI (连续值)', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=10)

        # 2. RSEI分类
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(rsei_class, cmap=rsei_cmap, vmin=1, vmax=5)

        if classification_breaks:
            breaks_str = f"[{classification_breaks[0]:.2f}, {classification_breaks[1]:.2f}, " \
                         f"{classification_breaks[2]:.2f}, {classification_breaks[3]:.2f}]"
            ax2.set_title(f'RSEI分类\n阈值: {breaks_str}', fontsize=12, fontweight='bold', pad=10)
        else:
            ax2.set_title('RSEI分类', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')

        class_names = ['差', '较差', '中等', '良好', '优秀']
        legend_elements = [Patch(facecolor=rsei_colors[i], label=class_names[i])
                           for i in range(5)]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)

        # 3-6. 四个指数
        indices_to_plot = [
            ('ndvi', 'NDVI (绿度)', 'Greens'),
            ('wet', 'WET (湿度)', 'Blues'),
            ('ndbsi', 'NDBSI (干度)', 'YlOrRd'),
            ('lst', 'LST (热度)', 'hot')
        ]

        for idx, (key, title, cmap) in enumerate(indices_to_plot):
            ax = fig.add_subplot(gs[0, 2 + idx % 2] if idx < 2 else gs[1, idx - 2])
            if key in indices:
                im = ax.imshow(indices[key], cmap=cmap)
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                ax.axis('off')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10)

        # 7. 水体掩膜
        if water_index is not None:
            ax7 = fig.add_subplot(gs[1, 2])
            im7 = ax7.imshow(water_index, cmap='RdYlBu')
            ax7.set_title(f'水体指数 (阈值={water_threshold:.3f})', fontsize=14, fontweight='bold', pad=10)
            ax7.axis('off')
            cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
            cbar7.ax.tick_params(labelsize=10)

        # 8. RSEI统计直方图
        ax8 = fig.add_subplot(gs[1, 3])
        valid_rsei = rsei[~np.isnan(rsei)]
        ax8.hist(valid_rsei, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax8.axvline(np.nanmean(rsei), color='red', linestyle='--',
                    label=f'均值={np.nanmean(rsei):.3f}', linewidth=2.5)

        if classification_breaks:
            colors_line = ['#d73027', '#fc8d59', '#fee08b', '#91cf60']
            for i, threshold in enumerate(classification_breaks):
                ax8.axvline(threshold, color=colors_line[i], linestyle=':',
                            linewidth=2, alpha=0.8)

        ax8.set_title('RSEI分布', fontsize=14, fontweight='bold', pad=10)
        ax8.set_xlabel('RSEI值', fontsize=12)
        ax8.set_ylabel('频数', fontsize=12)
        ax8.legend(fontsize=11)
        ax8.grid(alpha=0.3)
        ax8.tick_params(labelsize=10)

        # 9. 等级面积统计
        ax9 = fig.add_subplot(gs[2, :2])
        class_counts = [np.sum(rsei_class == i) for i in range(1, 6)]
        colors = rsei_colors
        bars = ax9.bar(class_names, class_counts, color=colors, edgecolor='black', alpha=0.85, linewidth=1.5)
        ax9.set_title('RSEI等级面积统计', fontsize=14, fontweight='bold', pad=10)
        ax9.set_ylabel('像素数量', fontsize=12)
        ax9.grid(axis='y', alpha=0.3)
        ax9.tick_params(labelsize=11)

        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(count):,}\n({count / np.sum(class_counts) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 10. 等级面积饼图
        ax10 = fig.add_subplot(gs[2, 2:])
        wedges, texts, autotexts = ax10.pie(class_counts, labels=class_names,
                                            colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 12})
        ax10.set_title('RSEI等级比例', fontsize=14, fontweight='bold', pad=10)

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        title_text = 'RSEI综合分析结果'
        if classification_breaks:
            method = "Jenks自然间断点" if classification_breaks != [0.2, 0.4, 0.6, 0.8] else "等间距"
            title_text += f' ({method}分类)'
        fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        return fig


# 由于代码太长，execute_rsei_calculation和main函数保持与之前相同
# 只需确保在调用RSEIVisualizer时字体已正确配置

def execute_rsei_calculation(input_file, config):
    """核心计算逻辑"""
    temp_dir = tempfile.mkdtemp()
    output_path = Path(temp_dir)

    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 读取和处理（保持代码不变）
        status_text.text("步骤1/9: 读取影像...")
        progress_bar.progress(10)
        reader = MultiSpectralImageReader(config)
        bands = reader.read_multiband_tif(input_file)

        status_text.text("步骤2/9: 数据预处理...")
        progress_bar.progress(15)
        max_val = np.nanmax(bands['red'])
        if max_val > 1.0:
            bands = reader.apply_scale_factor(bands, 0.0001)

        water_index = None
        water_mask = None
        water_threshold_used = None

        if config.mask_water:
            status_text.text("步骤3/9: 创建水体掩膜...")
            progress_bar.progress(25)
            water_index, water_mask, water_threshold_used = WaterMaskGenerator.create_water_mask(
                bands, config.water_index, config.water_threshold, config.use_otsu
            )

        status_text.text("步骤4/9: 计算遥感指数...")
        progress_bar.progress(35)
        calc = RemoteSensingIndices()
        ndvi = calc.calculate_ndvi(bands['red'], bands['nir'])
        wet = calc.calculate_wet(bands, config.satellite)
        ndbsi = calc.calculate_ndbsi(bands)

        ndbi = calc.calculate_ndbi(bands['swir1'], bands['nir'])
        swir1, red, nir, blue = bands['swir1'], bands['red'], bands['nir'], bands['blue']
        with np.errstate(divide='ignore', invalid='ignore'):
            si = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))

        if 'tir' in bands and bands['tir'] is not None:
            lst = calc.calculate_lst_simple(bands['tir'])
        else:
            lst = ndbsi

        indices = {
            'ndvi': ndvi,
            'wet': wet,
            'ndbsi': ndbsi,
            'lst': lst,
            'ndbi': ndbi,
            'si': si
        }

        status_text.text("步骤5/9: 计算RSEI...")
        progress_bar.progress(45)
        rsei_calc = RSEICalculator(config)
        rsei = rsei_calc.calculate_rsei_pca(ndvi, wet, ndbsi, lst, water_mask)

        status_text.text("步骤6/9: RSEI分类...")
        progress_bar.progress(55)
        classification_breaks = rsei_calc.calculated_breaks
        rsei_class = rsei_calc.classify_rsei(rsei, classification_breaks)

        class_names = ['差', '较差', '中等', '良好', '优秀']
        total_valid = np.sum(~np.isnan(rsei_class))

        st.write(f"\n使用的分类阈值: {[f'{b:.4f}' for b in classification_breaks]}")
        st.write("\n等级分布:")
        for i, name in enumerate(class_names, 1):
            count = np.sum(rsei_class == i)
            ratio = count / total_valid * 100 if total_valid > 0 else 0
            st.write(f"{name}: {count:,} ({ratio:.2f}%)")

        status_text.text("步骤7/9: 生成可视化图...")
        progress_bar.progress(65)

        # 在绘图前再次确认字体
        try:
            setup_chinese_font_enhanced()
        except:
            pass

        fig = RSEIVisualizer.create_comprehensive_visualization(
            rsei, rsei_class, indices, water_index,
            water_threshold_used, classification_breaks
        )

        img_path = output_path / 'RSEI_comprehensive.png'
        # 保存时使用更高的DPI和确保字体嵌入
        fig.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        # 导出文件部分保持不变...
        status_text.text("步骤8/9: 导出GeoTIFF文件...")
        progress_bar.progress(75)

        saved_files = []

        if config.export_geotiff and reader.metadata:
            with rasterio.open(output_path / 'RSEI.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei.astype('float32'), 1)
            saved_files.append('RSEI.tif')

            with rasterio.open(output_path / 'RSEI_classified.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei_class.astype('float32'), 1)
            saved_files.append('RSEI_classified.tif')

            if water_index is not None:
                with rasterio.open(output_path / 'Water_Index.tif', 'w', **reader.metadata) as dst:
                    dst.write(water_index.astype('float32'), 1)
                saved_files.append('Water_Index.tif')

                with rasterio.open(output_path / 'Water_Mask.tif', 'w', **reader.metadata) as dst:
                    dst.write(water_mask.astype('float32'), 1)
                saved_files.append('Water_Mask.tif')

            if config.export_indices:
                st.info("正在导出所有遥感指数...")

                index_files = {
                    'NDVI.tif': ndvi,
                    'WET.tif': wet,
                    'NDBSI.tif': ndbsi,
                    'LST.tif': lst,
                    'NDBI.tif': ndbi,
                    'SI.tif': si,
                    'Greenness_Normalized.tif': rsei_calc.rsei_components['greenness'],
                    'Wetness_Normalized.tif': rsei_calc.rsei_components['wetness'],
                    'Dryness_Normalized.tif': rsei_calc.rsei_components['dryness'],
                    'Heat_Normalized.tif': rsei_calc.rsei_components['heat']
                }

                for filename, data in index_files.items():
                    with rasterio.open(output_path / filename, 'w', **reader.metadata) as dst:
                        dst.write(data.astype('float32'), 1)
                    saved_files.append(filename)

        # Excel统计
        status_text.text("步骤9/9: 生成统计报告...")
        progress_bar.progress(85)

        stats_df = pd.DataFrame({
            '指标': ['NDVI', 'WET', 'NDBSI', 'LST', 'NDBI', 'SI', 'RSEI'],
            '最小值': [f"{np.nanmin(x):.4f}" for x in [ndvi, wet, ndbsi, lst, ndbi, si, rsei]],
            '最大值': [f"{np.nanmax(x):.4f}" for x in [ndvi, wet, ndbsi, lst, ndbi, si, rsei]],
            '均值': [f"{np.nanmean(x):.4f}" for x in [ndvi, wet, ndbsi, lst, ndbi, si, rsei]],
            '标准差': [f"{np.nanstd(x):.4f}" for x in [ndvi, wet, ndbsi, lst, ndbi, si, rsei]]
        })

        class_df = pd.DataFrame({
            '等级': class_names,
            '像素数': [int(np.sum(rsei_class == i)) for i in range(1, 6)],
            '百分比': [f"{np.sum(rsei_class == i) / total_valid * 100:.2f}%" for i in range(1, 6)]
        })

        threshold_df = pd.DataFrame({
            '分类阈值': ['差/较差', '较差/中等', '中等/良好', '良好/优秀'],
            '阈值': [f"{b:.4f}" for b in classification_breaks],
            '方法': ['Jenks自然间断点' if config.use_jenks else '手动设置'] * 4,
            'Jenks耗时(秒)': [f"{rsei_calc.jenks_time:.2f}" if config.use_jenks else 'N/A'] * 4
        })

        files_df = pd.DataFrame({
            '文件名': saved_files,
            '说明': [
                'RSEI连续值（0-1）',
                'RSEI分类（1-5）',
                '水体指数',
                '水体掩膜',
                '归一化植被指数',
                '湿度指数',
                '归一化建筑-土壤指数',
                '地表温度',
                '归一化建筑指数',
                '土壤指数',
                '绿度（归一化）',
                '湿度（归一化）',
                '干度（归一化）',
                '热度（归一化）'
            ][:len(saved_files)]
        })

        excel_path = output_path / 'RSEI_analysis.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='指标统计', index=False)
            class_df.to_excel(writer, sheet_name='等级分布', index=False)
            threshold_df.to_excel(writer, sheet_name='分类阈值', index=False)
            files_df.to_excel(writer, sheet_name='文件清单', index=False)

        progress_bar.progress(100)
        status_text.text("✅ 计算完成！")

        return {
            'rsei': rsei,
            'rsei_class': rsei_class,
            'indices': indices,
            'stats_df': stats_df,
            'class_df': class_df,
            'threshold_df': threshold_df,
            'files_df': files_df,
            'img_path': str(img_path),
            'excel_path': str(excel_path),
            'output_path': output_path,
            'classification_breaks': classification_breaks,
            'saved_files': saved_files
        }

    except Exception as e:
        st.error(f"计算失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def main():
    # 显示字体信息
    with st.sidebar:
        with st.expander("🔧 系统信息"):
            st.write(f"**操作系统:** {platform.system()}")
            current_font = plt.rcParams['font.sans-serif'][0]
            st.write(f"**使用字体:** {current_font}")

            st.markdown("---")
            st.markdown("""
            **解决中文乱码:**

            如果仍有乱码:
            1. 程序会自动下载字体
            2. 或手动安装中文字体

            Linux:
            ```
            sudo apt install fonts-wqy-microhei
            ```
            """)

    st.title("🌿 RSEI计算系统 v3.8 - 彻底修复中文")
    st.markdown("**Remote Sensing based Ecological Index 遥感生态指数计算工具**")

    # 其余界面代码保持不变...
    with st.sidebar:
        st.header("⚙️ 参数配置")
        st.subheader("📁 文件上传")
        uploaded_file = st.file_uploader("选择多波段TIF影像", type=['tif', 'tiff'])

        st.subheader("🛰️ 卫星参数")
        satellite = st.selectbox("卫星类型", ["Landsat8", "Sentinel2"], index=0)

        st.subheader("🔬 计算方法")
        use_pca = st.checkbox("使用PCA方法", value=True)

        st.subheader("📊 分类阈值设置")
        use_jenks = st.checkbox("使用Jenks自然间断点", value=True)

        if use_jenks:
            jenks_samples = st.slider("采样数量", 1000, 20000, 5000, 1000)
        else:
            st.write("手动设置阈值:")
            threshold_1 = st.number_input("差/较差", 0.0, 1.0, 0.2, 0.01)
            threshold_2 = st.number_input("较差/中等", 0.0, 1.0, 0.4, 0.01)
            threshold_3 = st.number_input("中等/良好", 0.0, 1.0, 0.6, 0.01)
            threshold_4 = st.number_input("良好/优秀", 0.0, 1.0, 0.8, 0.01)

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

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.success(f"✅ 文件已上传: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📦 文件大小: {file_size:.2f} MB")

        if st.button("▶️ 开始计算", type="primary"):
            if not use_jenks:
                thresholds = [threshold_1, threshold_2, threshold_3, threshold_4]
                if not all(thresholds[i] < thresholds[i + 1] for i in range(3)):
                    st.error("❌ 阈值必须递增！")
                    return

            config = RSEIConfig(
                satellite=satellite,
                use_pca=use_pca,
                export_indices=export_indices,
                export_geotiff=export_geotiff,
                mask_water=mask_water,
                water_index=water_index,
                water_threshold=water_threshold,
                use_otsu=use_otsu,
                use_jenks=use_jenks,
                classification_breaks=[threshold_1, threshold_2, threshold_3,
                                       threshold_4] if not use_jenks else [0.2, 0.4, 0.6, 0.8],
                jenks_samples=jenks_samples if use_jenks else 5000
            )

            start_time = time.time()

            with st.spinner("计算中，请稍候..."):
                results = execute_rsei_calculation(tmp_file_path, config)

            elapsed_time = time.time() - start_time

            if results:
                st.success(f"✅ 计算完成！耗时: {elapsed_time:.1f}秒")
                st.header("📊 计算结果")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 统计数据",
                    "🖼️ 可视化结果",
                    "📥 下载文件",
                    "📋 文件清单",
                    "ℹ️ 详细信息"
                ])

                with tab1:
                    st.subheader("指标统计")
                    st.dataframe(results['stats_df'], use_container_width=True)
                    st.subheader("等级分布")
                    st.dataframe(results['class_df'], use_container_width=True)
                    st.subheader("分类阈值")
                    st.dataframe(results['threshold_df'], use_container_width=True)

                with tab2:
                    st.subheader("RSEI综合分析可视化")
                    st.image(results['img_path'], use_column_width=True)
                    st.success("✅ 中文显示已修复！如仍有问题请安装中文字体。")

                with tab3:
                    st.subheader("下载结果文件")
                    col1, col2 = st.columns(2)

                    with col1:
                        with open(results['img_path'], 'rb') as f:
                            st.download_button(
                                "📷 下载可视化图",
                                f,
                                "RSEI_comprehensive.png",
                                "image/png",
                                use_container_width=True
                            )

                    with col2:
                        with open(results['excel_path'], 'rb') as f:
                            st.download_button(
                                "📊 下载统计报告",
                                f,
                                "RSEI_analysis.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                    st.markdown("---")

                    if export_geotiff:
                        st.subheader("📦 打包下载")
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            output_path = results['output_path']
                            for file in output_path.glob('*'):
                                zip_file.write(file, file.name)

                            readme = """RSEI计算结果

包含所有计算结果和遥感指数TIF文件。
详见RSEI_analysis.xlsx文件清单。"""
                            zip_file.writestr('README.txt', readme.encode('utf-8'))

                        zip_size = len(zip_buffer.getvalue()) / (1024 * 1024)

                        st.download_button(
                            f"📦 下载所有结果 - {zip_size:.2f} MB",
                            zip_buffer.getvalue(),
                            f"RSEI_results_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                            "application/zip",
                            use_container_width=True
                        )

                with tab4:
                    st.subheader("输出文件清单")
                    st.dataframe(results['files_df'], use_container_width=True)

                with tab5:
                    st.subheader("计算详情")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("卫星类型", config.satellite)
                        st.metric("总耗时", f"{elapsed_time:.1f}秒")
                    with col2:
                        st.metric("分类方法", 'Jenks' if config.use_jenks else '手动')
                        st.metric("文件数", len(results['saved_files']))

    else:
        st.info("👈 请在左侧上传多波段TIF影像")


if __name__ == "__main__":
    main()