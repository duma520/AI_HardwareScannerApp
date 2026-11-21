import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform
import psutil
import GPUtil
import cpuinfo
import cv2
import torch
import json
from datetime import datetime

class HardwareScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI加速能力检测工具 v1.0.3")
        self.root.geometry("800x600")
        
        # 全局样式设置
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Highlight.TLabel', foreground='blue')
        self.style.configure('Warning.TLabel', foreground='red')
        
        # 主界面布局
        self.create_widgets()
        
        # 检测结果存储
        self.results = {
            "system": {},
            "cpu": {},
            "gpu": {},
            "ram": {},
            "ai_frameworks": {},
            "recommendations": []
        }

    def create_widgets(self):
        """创建GUI界面组件"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题区域
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=10)
        ttk.Label(title_frame, text="AI加速能力全面检测", style='Title.TLabel').pack()
        
        # 检测按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="开始全面检测", command=self.run_full_scan).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="快速检测", command=self.run_quick_scan).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存报告", command=self.save_report).pack(side=tk.RIGHT, padx=5)
        
        # 结果显示区域
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个标签页
        self.create_system_tab()
        self.create_cpu_tab()
        self.create_gpu_tab()
        self.create_ai_tab()
        self.create_recommendation_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5,0))
        self.status_var.set("就绪 - 点击开始检测按钮")

    def create_system_tab(self):
        """系统信息标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="系统信息")
        
        # 使用网格布局
        for i in range(6): tab.grid_columnconfigure(i, weight=1)
        
        ttk.Label(tab, text="操作系统:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.os_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.os_label.grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="Python版本:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.py_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.py_label.grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="OpenCV版本:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.cv_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cv_label.grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="PyTorch版本:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.torch_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.torch_label.grid(row=3, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="CUDA可用性:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.cuda_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cuda_label.grid(row=4, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="Matplotlib版本:").grid(row=5, column=0, sticky="e", padx=5, pady=2)
        self.matplotlib_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.matplotlib_label.grid(row=5, column=1, sticky="w", pady=2)

    def create_cpu_tab(self):
        """CPU信息标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="CPU信息")
        
        # CPU基本信息
        ttk.Label(tab, text="处理器型号:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.cpu_model_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cpu_model_label.grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="核心数量:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.cpu_cores_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cpu_cores_label.grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="线程数量:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.cpu_threads_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cpu_threads_label.grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="基准频率:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.cpu_freq_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cpu_freq_label.grid(row=3, column=1, sticky="w", pady=2)
        
        # CPU能力检测
        ttk.Label(tab, text="AVX指令集:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.avx_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.avx_label.grid(row=4, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="SSE4.2指令集:").grid(row=5, column=0, sticky="e", padx=5, pady=2)
        self.sse_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.sse_label.grid(row=5, column=1, sticky="w", pady=2)

    def create_gpu_tab(self):
        """GPU信息标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="GPU信息")
        
        # GPU基本信息
        ttk.Label(tab, text="显卡型号:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.gpu_model_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.gpu_model_label.grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="显存容量:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.gpu_mem_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.gpu_mem_label.grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="CUDA核心数:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.cuda_cores_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cuda_cores_label.grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="驱动版本:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.driver_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.driver_label.grid(row=3, column=1, sticky="w", pady=2)
        
        # OpenCV GPU支持
        ttk.Label(tab, text="OpenCV CUDA:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.cv_cuda_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.cv_cuda_label.grid(row=4, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="OpenCL支持:").grid(row=5, column=0, sticky="e", padx=5, pady=2)
        self.opencl_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.opencl_label.grid(row=5, column=1, sticky="w", pady=2)

    def create_ai_tab(self):
        """AI框架支持标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="AI支持")
        
        # PyTorch支持
        ttk.Label(tab, text="PyTorch GPU加速:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.torch_gpu_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.torch_gpu_label.grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="TensorRT可用性:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.tensorrt_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.tensorrt_label.grid(row=1, column=1, sticky="w", pady=2)
        
        # ONNX Runtime支持
        ttk.Label(tab, text="ONNX Runtime:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.onnx_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.onnx_label.grid(row=2, column=1, sticky="w", pady=2)
        
        # 模型支持检测
        ttk.Label(tab, text="YOLOv8支持:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.yolo_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.yolo_label.grid(row=3, column=1, sticky="w", pady=2)
        
        ttk.Label(tab, text="MobileNet支持:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.mobilenet_label = ttk.Label(tab, text="", style='Highlight.TLabel')
        self.mobilenet_label.grid(row=4, column=1, sticky="w", pady=2)

    def create_recommendation_tab(self):
        """推荐方案标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="优化建议")
        
        self.recommendation_text = tk.Text(tab, wrap=tk.WORD, height=15, font=('Arial', 10))
        self.recommendation_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.recommendation_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.recommendation_text.config(yscrollcommand=scrollbar.set)
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

    def run_full_scan(self):
        """执行全面硬件检测"""
        self.status_var.set("正在检测系统信息...")
        self.root.update()
        
        try:
            # 1. 检测系统信息
            self.scan_system_info()
            
            # 2. 检测CPU信息
            self.status_var.set("正在检测CPU信息...")
            self.root.update()
            self.scan_cpu_info()
            
            # 3. 检测GPU信息
            self.status_var.set("正在检测GPU信息...")
            self.root.update()
            self.scan_gpu_info()
            
            # 4. 检测AI框架支持
            self.status_var.set("正在检测AI框架支持...")
            self.root.update()
            self.scan_ai_support()
            
            # 5. 生成优化建议
            self.status_var.set("正在生成优化建议...")
            self.root.update()
            self.generate_recommendations()
            
            self.status_var.set("检测完成！")
            messagebox.showinfo("完成", "硬件检测已完成，请查看各标签页详情")
            
        except Exception as e:
            self.status_var.set(f"检测出错: {str(e)}")
            messagebox.showerror("错误", f"检测过程中发生错误:\n{str(e)}")

    def run_quick_scan(self):
        """执行快速检测（仅关键信息）"""
        self.status_var.set("正在快速检测...")
        self.root.update()
        
        try:
            self.scan_system_info()
            self.scan_cpu_info()
            self.scan_gpu_info()
            self.generate_recommendations()
            
            self.status_var.set("快速检测完成！")
            messagebox.showinfo("完成", "快速检测已完成")
        except Exception as e:
            self.status_var.set(f"快速检测出错: {str(e)}")
            messagebox.showerror("错误", f"快速检测出错:\n{str(e)}")

    def scan_system_info(self):
        """扫描系统基本信息"""
        # 操作系统信息
        system_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        }
        
        self.os_label.config(text=f"{system_info['system']} {system_info['release']}")
        self.py_label.config(text=system_info['python_version'])
        
        # OpenCV信息
        cv_info = {
            "version": cv2.__version__,
            "cuda": cv2.cuda.getCudaEnabledDeviceCount() > 0,
            "opencl": cv2.ocl.haveOpenCL()
        }
        
        self.cv_label.config(text=cv_info['version'])
        self.cv_cuda_label.config(text="支持" if cv_info['cuda'] else "不支持")
        self.opencl_label.config(text="支持" if cv_info['opencl'] else "不支持")
        
        # PyTorch信息
        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "无"
        }
        
        self.torch_label.config(text=torch_info['version'])
        self.cuda_label.config(text="可用" if torch_info['cuda_available'] else "不可用")
        self.torch_gpu_label.config(text="可用" if torch_info['cuda_available'] else "不可用")
        
        # Matplotlib信息
        try:
            import matplotlib
            matplotlib_info = {
                "version": matplotlib.__version__,
                "font_cache": "已构建" if matplotlib.get_cachedir() else "未构建",
                "available": True
            }
            self.matplotlib_label.config(text=f"{matplotlib_info['version']} ({matplotlib_info['font_cache']})")
        except ImportError:
            matplotlib_info = {"available": False}
            self.matplotlib_label.config(text="未安装")
        
        # 存储结果
        self.results['system'] = system_info
        self.results['system']['opencv'] = cv_info
        self.results['system']['pytorch'] = torch_info
        self.results['system']['matplotlib'] = matplotlib_info

    def scan_cpu_info(self):
        """扫描CPU信息"""
        # 使用cpuinfo获取详细信息
        cpu_info = cpuinfo.get_cpu_info()
        
        # 基本信息
        self.cpu_model_label.config(text=cpu_info.get('brand_raw', '未知'))
        self.cpu_cores_label.config(text=psutil.cpu_count(logical=False))
        self.cpu_threads_label.config(text=psutil.cpu_count(logical=True))
        
        # CPU频率
        freq = psutil.cpu_freq()
        self.cpu_freq_label.config(text=f"{freq.current:.2f} MHz (最大 {freq.max:.2f} MHz)")
        
        # 指令集支持
        flags = cpu_info.get('flags', [])
        self.avx_label.config(text="支持" if 'avx' in flags else "不支持")
        self.sse_label.config(text="支持" if 'sse4_2' in flags else "不支持")
        
        # 存储结果
        self.results['cpu'] = {
            "model": cpu_info.get('brand_raw', '未知'),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency": {
                "current": freq.current,
                "max": freq.max
            },
            "features": {
                "avx": 'avx' in flags,
                "sse4_2": 'sse4_2' in flags,
                "avx2": 'avx2' in flags,
                "fma": 'fma' in flags
            },
            "usage": psutil.cpu_percent(interval=1)
        }

    def scan_gpu_info(self):
        """扫描GPU信息"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 取第一个GPU
                
                # 基本信息
                self.gpu_model_label.config(text=gpu.name)
                self.gpu_mem_label.config(text=f"{gpu.memoryTotal}MB")
                self.driver_label.config(text=gpu.driver)
                
                # CUDA核心数（估算）
                cuda_cores = self.estimate_cuda_cores(gpu.name)
                self.cuda_cores_label.config(text=str(cuda_cores) if cuda_cores else "未知")
                
                # 存储结果
                self.results['gpu'] = {
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "driver": gpu.driver,
                    "cuda_cores": cuda_cores,
                    "load": gpu.load * 100,
                    "detected": True
                }
            else:
                self.gpu_model_label.config(text="未检测到独立GPU")
                self.results['gpu'] = {"detected": False}
                
        except Exception as e:
            self.gpu_model_label.config(text=f"检测出错: {str(e)}")
            self.results['gpu'] = {"error": str(e), "detected": False}

    def estimate_cuda_cores(self, gpu_name):
        """估算CUDA核心数（基于常见GPU型号）"""
        gpu_name = gpu_name.lower()
        
        # NVIDIA GPU CUDA核心数估算
        if 'rtx 3090' in gpu_name: return 10496
        if 'rtx 3080' in gpu_name: return 8704
        if 'rtx 3070' in gpu_name: return 5888
        if 'rtx 3060' in gpu_name: return 3584
        if 'gtx 1660' in gpu_name: return 1408
        if 'gtx 1080' in gpu_name: return 2560
        
        # 无法识别时返回None
        return None

    def scan_ai_support(self):
        """检测AI框架和模型支持情况"""
        ai_support = {}
        
        # 检测ONNX Runtime
        try:
            import onnxruntime
            onnx_ver = getattr(onnxruntime, '__version__', '未知')
            providers = getattr(onnxruntime, 'get_available_providers', lambda: [])()
            ai_support['onnxruntime'] = {
                "version": onnx_ver,
                "gpu": 'CUDAExecutionProvider' in providers,
                "available": True
            }
            self.onnx_label.config(text=f"支持 (GPU: {'是' if ai_support['onnxruntime']['gpu'] else '否'})")
        except Exception as e:
            ai_support['onnxruntime'] = {
                "available": False,
                "error": str(e)
            }
            self.onnx_label.config(text=f"错误: {str(e)}")
        
        # 检测TensorRT
        try:
            import tensorrt
            ai_support['tensorrt'] = {
                "version": getattr(tensorrt, '__version__', '未知'),
                "available": True
            }
            self.tensorrt_label.config(text=f"支持 (v{ai_support['tensorrt']['version']})")
        except Exception as e:
            ai_support['tensorrt'] = {
                "available": False,
                "error": str(e)
            }
            self.tensorrt_label.config(text=f"错误: {str(e)}")
        
        # 检测YOLOv8支持
        try:
            from ultralytics import YOLO
            ai_support['yolov8'] = {"available": True}
            self.yolo_label.config(text="支持")
        except ImportError as e:
            ai_support['yolov8'] = {
                "available": False,
                "error": str(e)
            }
            self.yolo_label.config(text="未安装")
        
        # 检测MobileNet支持
        try:
            import tensorflow as tf
            tf.keras.applications.MobileNetV2()
            ai_support['mobilenet'] = {"available": True}
            self.mobilenet_label.config(text="支持")
        except Exception as e:
            ai_support['mobilenet'] = {
                "available": False,
                "error": str(e)
            }
            self.mobilenet_label.config(text="未安装/不支持")
        
        self.results['ai_frameworks'] = ai_support

    def generate_recommendations(self):
        """生成优化建议"""
        recommendations = []
        
        # CPU相关建议
        cpu = self.results['cpu']
        if cpu['features']['avx']:
            recommendations.append("✅ 您的CPU支持AVX指令集，适合运行优化后的OpenCV和NumPy")
        else:
            recommendations.append("⚠️ 您的CPU不支持AVX指令集，部分AI加速功能可能受限")
        
        if cpu['cores'] >= 4:
            recommendations.append(f"✅ 您的CPU有{cpu['cores']}个物理核心，建议启用多线程处理")
        else:
            recommendations.append(f"⚠️ 您的CPU只有{cpu['cores']}个物理核心，建议减少并行任务数量")
        
        # GPU相关建议
        gpu = self.results['gpu']
        if 'detected' in gpu and not gpu['detected']:
            recommendations.append("⚠️ 未检测到独立GPU，将无法使用CUDA加速")
        elif 'name' in gpu:
            recommendations.append(f"✅ 检测到GPU: {gpu['name']} (显存: {gpu['memory_total']}MB)")
            
            if gpu['memory_total'] < 2000:
                recommendations.append("⚠️ GPU显存较小，建议使用轻量级模型如YOLOv8n或MobileNet")
            else:
                recommendations.append("✅ 显存充足，可以运行较大的模型如YOLOv8x")
        
        # AI框架建议
        ai = self.results['ai_frameworks']
        if ai['onnxruntime'].get('available', False) and ai['onnxruntime'].get('gpu', False):
            recommendations.append("✅ ONNX Runtime已安装并支持GPU加速，建议使用ONNX格式模型")
        
        if ai['tensorrt'].get('available', False):
            recommendations.append("✅ TensorRT已安装，可对模型进行极致优化")
        
        # 模型选择建议
        if ai['yolov8'].get('available', False):
            recommendations.append("✅ 已安装YOLOv8，推荐使用YOLOv8n进行实时目标检测")
        
        if ai['mobilenet'].get('available', False):
            recommendations.append("✅ MobileNet可用，适合轻量级图像分类任务")
        
        # 根据硬件组合推荐最佳方案
        if gpu.get('memory_total', 0) > 4000 and ai['tensorrt'].get('available', False):
            recommendations.append("🚀 推荐方案: YOLOv8 + TensorRT加速 (最佳性能)")
        elif gpu.get('memory_total', 0) > 2000:
            recommendations.append("🏆 推荐方案: YOLOv8s + ONNX Runtime (平衡性能)")
        else:
            recommendations.append("🛠️ 推荐方案: MobileNetV3 + OpenCV DNN (兼容模式)")
        
        # 显示建议
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(tk.END, "硬件优化建议:\n\n")
        
        for rec in recommendations:
            color = 'red' if '⚠️' in rec else 'green' if '✅' in rec else 'blue'
            self.recommendation_text.insert(tk.END, rec + "\n", color)
            self.recommendation_text.tag_config(color, foreground=color)
        
        self.results['recommendations'] = recommendations

    def save_report(self):
        """保存检测报告到文件"""
        if not self.results.get('system'):
            messagebox.showwarning("警告", "请先执行检测再保存报告")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="保存检测报告"
        )
        
        if file_path:
            try:
                # 添加生成时间
                self.results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 根据文件类型保存
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.results, f, indent=4)
                else:
                    with open(file_path, 'w') as f:
                        self.write_text_report(f)
                
                messagebox.showinfo("成功", f"报告已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错:\n{str(e)}")

    def write_text_report(self, file_handle):
        """生成文本格式报告"""
        file_handle.write("="*50 + "\n")
        file_handle.write("AI加速能力检测报告\n")
        file_handle.write(f"生成时间: {self.results['timestamp']}\n")
        file_handle.write("="*50 + "\n\n")
        
        # 系统信息
        file_handle.write("[系统信息]\n")
        sys_info = self.results['system']
        file_handle.write(f"操作系统: {sys_info['system']} {sys_info['release']}\n")
        file_handle.write(f"Python版本: {sys_info['python_version']}\n")
        file_handle.write(f"OpenCV版本: {sys_info['opencv']['version']} (CUDA: {'是' if sys_info['opencv']['cuda'] else '否'})\n")
        file_handle.write(f"PyTorch版本: {sys_info['pytorch']['version']} (CUDA: {'是' if sys_info['pytorch']['cuda_available'] else '否'})\n")
        if 'matplotlib' in sys_info:
            file_handle.write(f"Matplotlib版本: {sys_info['matplotlib'].get('version', '未知')} (字体缓存: {sys_info['matplotlib'].get('font_cache', '未知')})\n")
        file_handle.write("\n")
        
        # CPU信息
        file_handle.write("[CPU信息]\n")
        cpu = self.results['cpu']
        file_handle.write(f"型号: {cpu['model']}\n")
        file_handle.write(f"核心/线程: {cpu['cores']}/{cpu['threads']}\n")
        file_handle.write(f"当前频率: {cpu['frequency']['current']} MHz\n")
        file_handle.write(f"指令集: AVX({cpu['features']['avx']}) SSE4.2({cpu['features']['sse4_2']})\n\n")
        
        # GPU信息
        file_handle.write("[GPU信息]\n")
        gpu = self.results['gpu']
        if 'detected' in gpu and not gpu['detected']:
            file_handle.write("未检测到独立GPU\n\n")
        else:
            file_handle.write(f"型号: {gpu['name']}\n")
            file_handle.write(f"显存: {gpu['memory_total']} MB\n")
            file_handle.write(f"驱动版本: {gpu['driver']}\n")
            file_handle.write(f"CUDA核心: {gpu.get('cuda_cores', '未知')}\n\n")
        
        # AI框架支持
        file_handle.write("[AI框架支持]\n")
        ai = self.results['ai_frameworks']
        file_handle.write(f"ONNX Runtime: {'是' if ai['onnxruntime'].get('available', False) else '否'} (GPU: {'是' if ai['onnxruntime'].get('gpu', False) else '否'})\n")
        file_handle.write(f"TensorRT: {'是' if ai['tensorrt'].get('available', False) else '否'}\n")
        file_handle.write(f"YOLOv8: {'是' if ai['yolov8'].get('available', False) else '否'}\n")
        file_handle.write(f"MobileNet: {'是' if ai['mobilenet'].get('available', False) else '否'}\n\n")
        
        # 优化建议
        file_handle.write("[优化建议]\n")
        for rec in self.results['recommendations']:
            file_handle.write(f"- {rec.replace('✅', '[推荐]').replace('⚠️', '[注意]')}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = HardwareScannerApp(root)
    root.mainloop()
