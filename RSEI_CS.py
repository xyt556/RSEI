import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import warnings
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================
# 核心计算类（与之前相同）
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
        print(f"📡 读取多波段影像: {Path(tif_path).name}")
        
        with rasterio.open(tif_path) as src:
            print(f"  尺寸: {src.width} x {src.height}, 波段数: {src.count}")
            
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
                
                print(f"  {band_name}: [{np.nanmin(band_data):.1f}, {np.nanmax(band_data):.1f}]")
        
        print(f"✅ 成功读取 {len(bands)} 个波段")
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
        print(f"🔧 应用缩放因子: {scale_factor}")
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
        print(f"💧 创建水体掩膜 (方法: {method})")
        
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
        
        print(f"  水体指数范围: [{np.nanmin(water_index):.4f}, {np.nanmax(water_index):.4f}]")
        
        if use_otsu and threshold is None:
            print("  🔍 使用OTSU算法计算阈值...")
            try:
                otsu_threshold, metrics = OTSUThreshold.calculate_otsu_threshold(water_index, bins=256)
                final_threshold = otsu_threshold
                print(f"    ✓ OTSU阈值: {final_threshold:.4f}")
                print(f"    ✓ 类间方差: {metrics['max_variance']:.6f}")
            except Exception as e:
                print(f"    ⚠️  OTSU失败: {e}, 使用默认阈值0.0")
                final_threshold = 0.0
        elif threshold is not None:
            final_threshold = threshold
            print(f"  阈值: {final_threshold:.4f} (手动)")
        else:
            final_threshold = 0.0
            print(f"  阈值: {final_threshold:.4f} (默认)")
        
        water_mask = water_index > final_threshold
        
        total_pixels = np.sum(~np.isnan(water_index))
        water_pixels = np.sum(water_mask)
        water_ratio = water_pixels / total_pixels * 100 if total_pixels > 0 else 0
        
        print(f"  水域: {water_pixels:,} ({water_ratio:.2f}%)")
        
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
        print("🔬 计算RSEI (PCA方法)...")
        
        if water_mask is not None:
            print("  应用水体掩膜...")
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
        print(f"  有效像素: {n_valid:,}")
        
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
        
        print(f"  PC1贡献率: {pca.explained_variance_ratio_[0]*100:.2f}%")
        
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
# GUI主程序
# =============================
class RSEIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 RSEI计算系统 v3.1 - OTSU自动阈值版")
        self.root.geometry("1100x850")
        
        # 变量
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="./rsei_results")
        self.satellite = tk.StringVar(value="Landsat8")
        self.use_pca = tk.BooleanVar(value=True)
        self.export_indices = tk.BooleanVar(value=True)
        self.export_geotiff = tk.BooleanVar(value=True)
        self.mask_water = tk.BooleanVar(value=True)
        self.water_index = tk.StringVar(value="MNDWI")
        self.use_otsu = tk.BooleanVar(value=True)
        self.water_threshold = tk.DoubleVar(value=0.0)
        
        self.log_queue = queue.Queue()
        self.is_running = False
        
        self.create_widgets()
        sys.stdout = TextRedirector(self.log_queue)
        self.update_log()
        
    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 配置标签页
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="⚙️ 参数配置")
        self.create_config_tab(config_frame)
        
        # 日志标签页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📋 运行日志")
        self.create_log_tab(log_frame)
        
        # 结果标签页
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="📊 结果查看")
        self.create_result_tab(result_frame)
        
        # 控制栏
        self.create_control_bar()
        
    def create_config_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 文件选择
        file_group = ttk.LabelFrame(scrollable_frame, text="📁 文件选择", padding=10)
        file_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_group, text="输入影像:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_group, textvariable=self.input_file, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(file_group, text="浏览...", command=self.browse_input).grid(row=0, column=2)
        
        ttk.Label(file_group, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_group, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(file_group, text="浏览...", command=self.browse_output).grid(row=1, column=2)
        
        # 卫星参数
        satellite_group = ttk.LabelFrame(scrollable_frame, text="🛰️ 卫星参数", padding=10)
        satellite_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(satellite_group, text="卫星类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(satellite_group, text="Landsat 8", variable=self.satellite, 
                       value="Landsat8").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(satellite_group, text="Sentinel-2", variable=self.satellite, 
                       value="Sentinel2").grid(row=0, column=2, sticky=tk.W)
        
        # 计算方法
        method_group = ttk.LabelFrame(scrollable_frame, text="🔬 计算方法", padding=10)
        method_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(method_group, text="使用PCA方法（推荐）", 
                       variable=self.use_pca).pack(anchor=tk.W, pady=2)
        
        # 水体掩膜
        water_group = ttk.LabelFrame(scrollable_frame, text="💧 水体掩膜（OTSU自动阈值）", padding=10)
        water_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(water_group, text="去除水域", 
                       variable=self.mask_water,
                       command=self.toggle_water_options).grid(row=0, column=0, 
                                                               columnspan=3, sticky=tk.W, pady=2)
        
        ttk.Label(water_group, text="水体指数:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=(20, 0))
        self.water_index_combo = ttk.Combobox(water_group, textvariable=self.water_index, 
                                             values=["MNDWI", "NDWI", "AWEIsh"], 
                                             state="readonly", width=15)
        self.water_index_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # OTSU选项
        self.otsu_check = ttk.Checkbutton(water_group, text="使用OTSU自动计算阈值（推荐）", 
                                         variable=self.use_otsu,
                                         command=self.toggle_threshold_input)
        self.otsu_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5, padx=(20, 0))
        
        ttk.Label(water_group, text="手动阈值:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=(20, 0))
        self.threshold_spin = ttk.Spinbox(water_group, from_=-1.0, to=1.0, 
                                         increment=0.1, textvariable=self.water_threshold, 
                                         width=15, state='disabled')
        self.threshold_spin.grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Label(water_group, text="(仅在不使用OTSU时生效)", 
                 foreground="gray").grid(row=3, column=2, sticky=tk.W)
        
        # 导出选项
        export_group = ttk.LabelFrame(scrollable_frame, text="💾 导出选项", padding=10)
        export_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(export_group, text="导出GeoTIFF文件", 
                       variable=self.export_geotiff).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(export_group, text="导出所有遥感指数", 
                       variable=self.export_indices).pack(anchor=tk.W, pady=2)
        
        # 说明
        info_group = ttk.LabelFrame(scrollable_frame, text="ℹ️ 使用说明", padding=10)
        info_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        info_text = scrolledtext.ScrolledText(info_group, height=10, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)
        info_text.insert(tk.END, """
🌟 OTSU自动阈值功能说明:

1. OTSU算法会自动计算最优的水体分割阈值
2. 推荐使用MNDWI + OTSU组合，效果最佳
3. 程序会自动生成OTSU分析图，显示阈值计算过程
4. 如果OTSU结果不理想，可取消勾选，手动设置阈值

使用步骤:
1. 选择多波段TIF影像文件
2. 选择卫星类型
3. 勾选"去除水域"和"使用OTSU自动计算阈值"
4. 点击"开始计算"
5. 查看运行日志和结果

注意事项:
- OTSU算法适用于水陆分界明显的区域
- 对于水域占比很小或很大的区域，可能需要手动调整
- 计算完成后会生成OTSU分析图（直方图+类间方差曲线）
        """)
        info_text.config(state=tk.DISABLED)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_log_tab(self, parent):
        self.log_text = scrolledtext.ScrolledText(parent, height=30, wrap=tk.WORD, 
                                                  font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_control = ttk.Frame(parent)
        log_control.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_control, text="清空日志", 
                  command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control, text="保存日志", 
                  command=self.save_log).pack(side=tk.LEFT, padx=5)
        
    def create_result_tab(self, parent):
        result_label = ttk.Label(parent, text="计算完成后，结果将显示在这里", 
                                font=("Arial", 12))
        result_label.pack(pady=20)
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="📂 打开输出目录", 
                  command=self.open_output_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 查看统计报告", 
                  command=self.view_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🖼️ 查看可视化结果", 
                  command=self.view_visualization).pack(side=tk.LEFT, padx=5)
        
        self.result_frame = ttk.Frame(parent)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_control_bar(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="就绪", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.run_button = ttk.Button(control_frame, text="▶️ 开始计算", 
                                     command=self.run_analysis)
        self.run_button.pack(side=tk.RIGHT, padx=5)
        
    def toggle_water_options(self):
        state = 'readonly' if self.mask_water.get() else 'disabled'
        self.water_index_combo.config(state=state)
        state = 'normal' if self.mask_water.get() else 'disabled'
        self.otsu_check.config(state=state)
        self.toggle_threshold_input()
        
    def toggle_threshold_input(self):
        if self.mask_water.get() and not self.use_otsu.get():
            self.threshold_spin.config(state='normal')
        else:
            self.threshold_spin.config(state='disabled')
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="选择多波段TIF影像",
            filetypes=[("TIF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            
    def browse_output(self):
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir.set(dirname)
            
    def update_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.update_log)
        
    def save_log(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("成功", f"日志已保存")
            
    def open_output_dir(self):
        import os
        import platform
        output_path = Path(self.output_dir.get())
        if output_path.exists():
            if platform.system() == 'Windows':
                os.startfile(output_path)
            elif platform.system() == 'Darwin':
                os.system(f'open "{output_path}"')
            else:
                os.system(f'xdg-open "{output_path}"')
        else:
            messagebox.showwarning("警告", "输出目录不存在！")
            
    def view_statistics(self):
        excel_file = Path(self.output_dir.get()) / 'RSEI_analysis.xlsx'
        if excel_file.exists():
            import os
            os.startfile(excel_file)
        else:
            messagebox.showwarning("警告", "统计文件不存在！")
            
    def view_visualization(self):
        img_file = Path(self.output_dir.get()) / 'RSEI_comprehensive.png'
        if img_file.exists():
            from PIL import Image, ImageTk
            img_window = tk.Toplevel(self.root)
            img_window.title("可视化结果")
            
            img = Image.open(img_file)
            img.thumbnail((1200, 900), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(img_window, image=photo)
            label.image = photo
            label.pack()
        else:
            messagebox.showwarning("警告", "可视化文件不存在！")
            
    def run_analysis(self):
        if not self.input_file.get():
            messagebox.showerror("错误", "请选择输入影像文件！")
            return
        
        if self.is_running:
            messagebox.showwarning("警告", "已有任务在运行中！")
            return
        
        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text="运行中...", foreground="orange")
        self.progress.start()
        self.is_running = True
        
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
        
    def _run_analysis_thread(self):
        try:
            config = RSEIConfig(
                satellite=self.satellite.get(),
                use_pca=self.use_pca.get(),
                export_indices=self.export_indices.get(),
                export_geotiff=self.export_geotiff.get(),
                mask_water=self.mask_water.get(),
                water_index=self.water_index.get(),
                water_threshold=None if self.use_otsu.get() else self.water_threshold.get(),
                use_otsu=self.use_otsu.get()
            )
            
            self.execute_rsei_calculation(
                self.input_file.get(),
                self.output_dir.get(),
                config
            )
            
            self.root.after(0, self._on_analysis_complete, True, "计算完成！")
            
        except Exception as e:
            self.root.after(0, self._on_analysis_complete, False, str(e))
            
    def _on_analysis_complete(self, success, message):
        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        self.is_running = False
        
        if success:
            self.status_label.config(text="完成", foreground="green")
            messagebox.showinfo("成功", message)
        else:
            self.status_label.config(text="错误", foreground="red")
            messagebox.showerror("错误", f"计算失败: {message}")
            
    def execute_rsei_calculation(self, input_file, output_dir, config):
        """核心计算逻辑"""
        print("="*80)
        print("🌿 开始RSEI计算...")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. 读取影像
        print("\n📁 读取影像...")
        reader = MultiSpectralImageReader(config)
        bands = reader.read_multiband_tif(input_file)
        
        # 2. 预处理
        print("\n⚙️ 预处理...")
        max_val = np.nanmax(bands['red'])
        if max_val > 1.0:
            bands = reader.apply_scale_factor(bands, 0.0001)
        
        # 3. 水体掩膜
        water_index = None
        water_mask = None
        water_threshold_used = None
        
        if config.mask_water:
            print("\n💧 创建水体掩膜...")
            water_index, water_mask, water_threshold_used = WaterMaskGenerator.create_water_mask(
                bands, config.water_index, config.water_threshold, config.use_otsu
            )
        
        # 4. 计算指数
        print("\n🔬 计算遥感指数...")
        calc = RemoteSensingIndices()
        ndvi = calc.calculate_ndvi(bands['red'], bands['nir'])
        wet = calc.calculate_wet(bands, config.satellite)
        ndbsi = calc.calculate_ndbsi(bands)
        
        if 'tir' in bands and bands['tir'] is not None:
            lst = calc.calculate_lst_simple(bands['tir'])
        else:
            lst = ndbsi
        
        indices = {'ndvi': ndvi, 'wet': wet, 'ndbsi': ndbsi, 'lst': lst}
        
        # 5. 计算RSEI
        print("\n🌍 计算RSEI...")
        rsei_calc = RSEICalculator(config)
        rsei = rsei_calc.calculate_rsei_pca(ndvi, wet, ndbsi, lst, water_mask)
        
        # 6. 分类
        print("\n📊 分类...")
        rsei_class = rsei_calc.classify_rsei(rsei)
        
        class_names = ['差', '较差', '中等', '良好', '优秀']
        total_valid = np.sum(~np.isnan(rsei_class))
        
        for i, name in enumerate(class_names, 1):
            count = np.sum(rsei_class == i)
            ratio = count / total_valid * 100 if total_valid > 0 else 0
            print(f"  {name}: {count:,} ({ratio:.2f}%)")
        
        # 7. 导出
        print("\n💾 导出结果...")
        
        if config.export_geotiff and reader.metadata:
            with rasterio.open(output_path / 'RSEI.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei.astype('float32'), 1)
            
            with rasterio.open(output_path / 'RSEI_classified.tif', 'w', **reader.metadata) as dst:
                dst.write(rsei_class.astype('float32'), 1)
            
            if water_index is not None:
                with rasterio.open(output_path / 'Water_Index.tif', 'w', **reader.metadata) as dst:
                    dst.write(water_index.astype('float32'), 1)
                with rasterio.open(output_path / 'Water_Mask.tif', 'w', **reader.metadata) as dst:
                    dst.write(water_mask.astype('float32'), 1)
        
        # Excel
        stats_df = pd.DataFrame({
            '指标': ['NDVI', 'WET', 'NDBSI', 'LST', 'RSEI'],
            '最小值': [np.nanmin(x) for x in [ndvi, wet, ndbsi, lst, rsei]],
            '最大值': [np.nanmax(x) for x in [ndvi, wet, ndbsi, lst, rsei]],
            '均值': [np.nanmean(x) for x in [ndvi, wet, ndbsi, lst, rsei]],
            '标准差': [np.nanstd(x) for x in [ndvi, wet, ndbsi, lst, rsei]]
        })
        
        class_df = pd.DataFrame({
            '等级': class_names,
            '像素数': [int(np.sum(rsei_class == i)) for i in range(1, 6)],
            '百分比': [f"{np.sum(rsei_class == i)/total_valid*100:.2f}%" for i in range(1, 6)]
        })
        
        with pd.ExcelWriter(output_path / 'RSEI_analysis.xlsx', engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='指标统计', index=False)
            class_df.to_excel(writer, sheet_name='等级分布', index=False)
        
        print("\n✅ 完成！")
        print(f"📂 结果: {output_path.absolute()}")


class TextRedirector:
    def __init__(self, queue):
        self.queue = queue
        
    def write(self, string):
        self.queue.put(string)
        
    def flush(self):
        pass


def main():
    root = tk.Tk()
    app = RSEIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()