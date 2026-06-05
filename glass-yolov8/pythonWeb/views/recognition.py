from flask import Blueprint, render_template, send_file, jsonify, request, Response, session
# from MvCameraControl_class import *
# from CameraParams_header import *
from ctypes import *
import time
import cv2
import numpy as np
import os
import sys
import uuid
from datetime import datetime
import threading
import atexit

try:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from utils.oss_client import upload_captured_image, get_image_url, check_image_exists
    from utils.local_cache import LocalCacheManager

    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False
    print("OSS模块未安装，将使用本地存储")

# 导入在线功能的本地缓存管理器（统一使用online_processing模块的实例）
from .online_processing import online_cache_manager


# CameraManager class
class CameraManager:
    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.device_status = False
        self.stOutFrame = None
        self.frame_info = None
        self.frame_acquired = False
        self.strModeName = "Unknown Device"
        self.last_error = ""
        self.lock = threading.RLock()  # 添加可重入锁保护并发访问（允许同一线程多次获取）
        self.streaming_paused = False  # 视频流暂停标志（拍照时暂停视频流）

#该函数是整个相机系统的核心初始化方法，确保相机正确连接并准备好进行图像采集。
    def data_camera(self):
        with self.lock:  # 保护相机初始化过程
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
                    deviceList = MV_CC_DEVICE_INFO_LIST()
                    self.cam = MvCamera()
                    ret = self.cam.MV_CC_EnumDevices(tlayerType, deviceList)
                    if ret != 0:
                        self.last_error = f"枚举设备失败，错误码: 0x{ret:x}"
                        print(self.last_error)
                        continue
                    if deviceList.nDeviceNum == 0:
                        self.last_error = "未检测到任何相机设备！"
                        print(self.last_error)
                        continue
                    print(f"检测到 {deviceList.nDeviceNum} 个设备")
                    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
                    ret = self.cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
                    if ret != 0:
                        self.last_error = f"创建句柄失败，错误码: 0x{ret:x}"
                        print(self.last_error)
                        continue
                    try:
                        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                            self.strModeName = ctypes.string_at(stDeviceList.SpecialInfo.stGigEInfo.chModelName).decode(
                                'utf-8', 'ignore')
                        elif stDeviceList.nTLayerType == MV_USB_DEVICE:
                            self.strModeName = ctypes.string_at(stDeviceList.SpecialInfo.stUsb3VInfo.chModelName).decode(
                                'utf-8', 'ignore')
                        else:
                            self.strModeName = "Unknown Device Type"
                    except Exception as e:
                        self.last_error = f"获取设备名称时出错: {e}"
                        print(self.last_error)
                        self.strModeName = "Unknown Device"
                    print(f"设备名称: {self.strModeName}")
                    ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                    if ret != 0:
                        self.last_error = f"打开相机失败，错误码: 0x{ret:x} (0x80000203 表示相机未打开，可能是权限或资源冲突)"
                        print(self.last_error)
                        if self.cam:
                            self.cam.MV_CC_DestroyHandle()
                        continue
                    print("相机已成功打开")
                    try:
                        enum_entry = MVCC_ENUMVALUE()
                        ret = self.cam.MV_CC_GetEnumValue("TriggerMode", enum_entry)
                        if ret == 0:
                            print(f"当前触发模式: {enum_entry.nCurValue}, 支持的模式数量: {enum_entry.nSupportedNum}")
                        else:
                            self.last_error = f"无法获取触发模式信息: 0x{ret:x}"
                            print(self.last_error)
                        print("尝试设置连续模式...")
                        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", 0)  # 强制设置为连续模式
                        if ret != 0:
                            self.last_error = f"设置连续模式失败，错误码: 0x{ret:x}"
                            print(self.last_error)
                            continue
                        print("成功设置连续模式")
                    except Exception as e:
                        self.last_error = f"设置触发模式时发生异常: {e}"
                        print(self.last_error)
                        continue
                    ret = self.cam.MV_CC_StartGrabbing()
                    if ret != 0:
                        self.last_error = f"开始取流失败，错误码: 0x{ret:x}"
                        print(self.last_error)
                        continue
                    print("开始取流成功")
                    self.device_status = True
                    self.last_error = ""  # Reset error on success
                    return True
                except Exception as e:
                    self.last_error = f"相机初始化失败: {e}"
                    print(self.last_error)
                print(f"初始化尝试 {attempt + 1}/{max_attempts} 失败，重试...")
                time.sleep(1)  # Wait before retry
                if self.cam:
                    self.cam.MV_CC_DestroyHandle()
                    self.cam = None
            return False

#获取单帧图像数据
    def get_image(self, MV_E_TIMEOUT=None, blocking=False):
        """
        获取图像数据
        
        Args:
            MV_E_TIMEOUT: 超时时间（暂未使用）
            blocking: 是否阻塞等待锁（True=拍照专用，False=视频流专用）
        """
        # 尝试获取锁
        if not self.lock.acquire(blocking=blocking):
            # 相机正在被其他线程使用，跳过这一帧
            return None
        
        try:
            if not self.device_status:
                # 不打印日志，避免视频流产生大量重复输出
                self.last_error = "相机未连接，无法获取图像！"
                return None

            # 检查相机对象是否有效
            if self.cam is None:
                self.last_error = "相机对象无效"
                print(self.last_error)
                self.device_status = False
                return None
            self.stOutFrame = MV_FRAME_OUT()
            self.frame_acquired = False
            start_time = time.time()

            # 增加超时时间到 8000ms，给相机更多时间
            ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 8000)

            if ret != 0:
                # 根据错误类型进行不同处理
                if ret == 0x80000003:
                    # MV_E_CALLORDER: 函数调用顺序错误（通常是并发冲突）
                    self.last_error = "错误: 函数调用顺序错误 (0x80000003)"
                    # 静默返回，不打印日志（避免大量重复日志）
                    return None
                elif ret == 0x8000000D:
                    self.last_error = "错误: 功能未实现 (0x8000000D)"
                    print(self.last_error)
                    print("尝试重启取流...")
                    self.restart_streaming()
                elif ret == MV_E_TIMEOUT:
                    self.last_error = "错误: 获取图像超时，可能是相机未发送数据"
                    print(self.last_error)
                    print("检测到超时，尝试重启取流...")
                    self.restart_streaming()
                elif ret == MV_E_NODATA:
                    self.last_error = "错误: 没有可用的图像数据"
                elif ret == MV_E_GC_GENERIC:
                    self.last_error = "错误: 通用错误"
                elif ret == 0x80000203:
                    self.last_error = "错误: 相机未打开或权限不足"
                    self.device_status = False
                    print(self.last_error)
                else:
                    self.last_error = f"获取图像失败，错误码: 0x{ret:x}"
                    print(self.last_error)

                return None

            # 检查帧信息
            if not hasattr(self.stOutFrame, 'stFrameInfo') or self.stOutFrame.stFrameInfo is None:
                self.last_error = "错误: 帧信息无效"
                print(self.last_error)
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return None

            self.frame_info = self.stOutFrame.stFrameInfo

            # 验证帧信息
            if (not hasattr(self.frame_info, 'nWidth') or not hasattr(self.frame_info, 'nHeight') or
                    not hasattr(self.frame_info, 'nFrameLen')):
                self.last_error = "错误: 帧信息不完整"
                print(self.last_error)
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return None

            # 检查图像数据
            if self.stOutFrame.pBufAddr is None or self.frame_info.nFrameLen == 0:
                self.last_error = "错误：获取的图像数据为空！"
                print(self.last_error)
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return None

            # 验证图像尺寸
            if self.frame_info.nWidth <= 0 or self.frame_info.nHeight <= 0:
                self.last_error = f"错误: 无效的图像尺寸 {self.frame_info.nWidth}x{self.frame_info.nHeight}"
                print(self.last_error)
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return None

            # 复制图像数据
            nPayloadSize = self.frame_info.nFrameLen
            pData = self.stOutFrame.pBufAddr

            # 验证数据指针
            if pData is None:
                self.last_error = "错误: 图像数据指针为空"
                print(self.last_error)
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                return None

            self.data_buf = (c_ubyte * nPayloadSize)()
            cdll.msvcrt.memcpy(byref(self.data_buf), pData, nPayloadSize)

            end_time = time.time()
            camera_time = round(abs(start_time - end_time) * 1000, 3)
            
            # 仅在耗时较长时打印日志（避免视频流产生大量日志）
            if camera_time > 50:  # 超过50ms才打印
                print(f"⚠️ 图像获取耗时较长: {camera_time}ms ({self.frame_info.nWidth}x{self.frame_info.nHeight})")

            self.frame_acquired = True
            return self.data_buf

        except Exception as e:
            self.last_error = f"获取图像时发生异常: {str(e)}"
            print(self.last_error)
            # 如果发生异常，尝试释放缓冲区
            try:
                if hasattr(self, 'stOutFrame') and self.stOutFrame is not None:
                    self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
            except:
                pass
            return None
        
        finally:
            # 无论如何都要释放锁
            self.lock.release()

# 释放帧缓冲区
    def release_frame_buffer(self):
        with self.lock:  # 保护帧缓冲区释放操作
            if self.stOutFrame is not None and self.device_status:
                self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                self.stOutFrame = None

# 重启取流
    def restart_streaming(self):
        with self.lock:  # 保护重启取流过程
            if self.device_status:
                print("尝试重启取流...")
                ret = self.cam.MV_CC_StopGrabbing()
                if ret != 0:
                    print(f"停止取流失败，错误码: 0x{ret:x}")
                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    print(f"重启取流失败，错误码: 0x{ret:x}")
                else:
                    print("取流重启成功")

#关闭相机
    def off_camera(self):
        with self.lock:  # 保护关闭相机过程
            if not self.device_status:
                print("相机未连接，无需关闭")
                return True
            ret = self.cam.MV_CC_StopGrabbing()
            print(f"停止取流执行码: [0x{ret:x}]")
            ret = self.cam.MV_CC_CloseDevice()
            print(f"关闭设备执行码: [0x{ret:x}]")
            ret = self.cam.MV_CC_DestroyHandle()
            print(f"销毁句柄执行码: [0x{ret:x}]")
            self.device_status = False
            self.streaming_paused = False  # 重置暂停标志
            return True
    
    def pause_streaming(self):
        """暂停视频流（拍照时使用，不关闭相机）"""
        with self.lock:
            if self.device_status and not self.streaming_paused:
                self.streaming_paused = True
                print("[视频流] 已暂停视频流（为拍照让出资源）")
                return True
            return False
    
    def resume_streaming(self):
        """恢复视频流"""
        with self.lock:
            if self.device_status and self.streaming_paused:
                self.streaming_paused = False
                print("[视频流] 已恢复视频流")
                return True
            return False

    def get_status(self):
        return {
            "connected": self.device_status,
            "device_name": self.strModeName,
            "last_error": self.last_error
        }

    def health_check(self):
        """相机健康检查"""
        health_info = {
            "camera_connected": self.device_status,
            "camera_object_valid": self.cam is not None,
            "device_name": self.strModeName,
            "last_error": self.last_error,
            "frame_acquired": getattr(self, 'frame_acquired', False),
            "timestamp": datetime.now().isoformat()
        }

        if self.device_status and self.cam is not None:
            try:
                # 尝试获取一帧图像来测试相机响应
                test_frame = self.get_image()
                if test_frame is not None:
                    health_info["camera_responsive"] = True
                    health_info["last_frame_size"] = f"{self.frame_info.nWidth}x{self.frame_info.nHeight}" if hasattr(
                        self, 'frame_info') and self.frame_info else "Unknown"
                    # 释放测试帧
                    self.release_frame_buffer()
                else:
                    health_info["camera_responsive"] = False
                    health_info["camera_error"] = self.last_error
            except Exception as e:
                health_info["camera_responsive"] = False
                health_info["camera_error"] = str(e)
        else:
            health_info["camera_responsive"] = False

        return health_info

# 获取图像
    def get_frame(self):
        while True:
            # 如果视频流被暂停，生成暂停画面
            if self.streaming_paused:
                placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_img, "Video Paused", (200, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(placeholder_img, "(Capturing Photo...)", (180, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                ret, jpeg = cv2.imencode('.jpg', placeholder_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.5)  # 暂停时降低更新频率
                continue
            
            frame = self.get_image()
            if frame is not None:
                temp = np.frombuffer(frame, dtype=np.uint8)
                width = self.frame_info.nWidth
                height = self.frame_info.nHeight
                pixel_format = self.frame_info.enPixelType
                channels = 3 if pixel_format in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed] else 1
                if channels == 1 and pixel_format == PixelType_Gvsp_BayerGR8:
                    temp = temp.reshape((height, width))
                    temp = cv2.cvtColor(temp, cv2.COLOR_BayerGR2BGR)
                else:
                    temp = temp.reshape((height, width, channels))
                    if pixel_format == PixelType_Gvsp_RGB8_Packed:
                        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
                ret, jpeg = cv2.imencode('.jpg', temp)
                self.release_frame_buffer()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.2)  # 增加延迟到 5 FPS，减轻相机负担
            time.sleep(0.1)  # 额外延迟以稳定流


# 相机线程管理器
class CameraThreadManager:
    def __init__(self):
        self.cam_manager = None
        self.is_running = False
        self.stream_thread = None
        self.detection_thread = None
        self.lock = threading.Lock()
        
    def start_camera(self):
        """启动相机"""
        with self.lock:
            if self.is_running:
                print("[相机管理] 相机已在运行中")
                return {"success": True, "message": "相机已在运行中"}
            
            print("[相机管理] 正在启动相机...")
            self.cam_manager = CameraManager()
            
            # 尝试连接相机
            if not self.cam_manager.data_camera():
                error_msg = self.cam_manager.last_error or "相机初始化失败"
                print(f"[相机管理] 相机启动失败: {error_msg}")
                self.cam_manager = None
                return {"success": False, "error": error_msg}
            
            self.is_running = True
            
            # 确保视频流不是暂停状态
            if self.cam_manager:
                self.cam_manager.streaming_paused = False
            
            print("[相机管理] ✅ 相机启动成功")
            return {"success": True, "message": "相机启动成功"}
    
    def stop_camera(self):
        """停止相机"""
        with self.lock:
            if not self.is_running:
                print("[相机管理] 相机未运行")
                return {"success": True, "message": "相机未运行"}
            
            print("[相机管理] 正在停止相机...")
            
            try:
                if self.cam_manager:
                    self.cam_manager.off_camera()
                    self.cam_manager = None
                
                self.is_running = False
                print("[相机管理] ✅ 相机已停止")
                return {"success": True, "message": "相机已停止"}
                
            except Exception as e:
                print(f"[相机管理] 停止相机时出错: {e}")
                return {"success": False, "error": str(e)}
    
    def get_camera(self):
        """获取相机管理器实例"""
        return self.cam_manager if self.is_running else None
    
    def get_status(self):
        """获取相机状态"""
        if not self.is_running or not self.cam_manager:
            return {
                "running": False,
                "connected": False,
                "device_name": "未连接",
                "last_error": "相机未启动"
            }
        
        cam_status = self.cam_manager.get_status()
        return {
            "running": self.is_running,
            "connected": cam_status["connected"],
            "device_name": cam_status["device_name"],
            "last_error": cam_status["last_error"]
        }


# Blueprint object
re = Blueprint("recognition", __name__)

# 初始化相机线程管理器（不立即启动相机）
camera_thread_manager = CameraThreadManager()

# 程序退出时清理相机资源
def cleanup_camera():
    print("[系统] 程序退出，清理相机资源...")
    camera_thread_manager.stop_camera()

atexit.register(cleanup_camera)


@re.route("/recognition")
def recognition():
    return render_template("on_line.html")


@re.route("/recognition/camera_status")
def camera_status():
    """获取相机状态"""
    try:
        status = camera_thread_manager.get_status()
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/start_camera", methods=["POST"])
def start_camera():
    """启动相机"""
    try:
        result = camera_thread_manager.start_camera()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/stop_camera", methods=["POST"])
def stop_camera():
    """停止相机"""
    try:
        result = camera_thread_manager.stop_camera()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/camera_health")
def camera_health():
    """相机健康检查API"""
    try:
        CAM = camera_thread_manager.get_camera()
        if not CAM:
            return jsonify({"success": False, "error": "相机未启动"})
        health_info = CAM.health_check()
        return jsonify({"success": True, "health": health_info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/capture", methods=["POST"])
def capture_image():
    max_retries = 3
    retry_delay = 0.5  # 重试间隔（秒）
    
    # 获取相机实例
    CAM = camera_thread_manager.get_camera()
    if not CAM:
        return jsonify({"success": False, "error": "相机未启动，请先启动相机"})

    for attempt in range(max_retries):
        try:
            # 检查相机状态
            if not CAM.device_status:
                # 尝试重新连接相机
                print(f"相机未连接，尝试重新连接... (尝试 {attempt + 1}/{max_retries})")
                if not CAM.data_camera():
                    if attempt == max_retries - 1:
                        return jsonify({"success": False, "error": "相机连接失败，请检查设备连接"})
                    time.sleep(retry_delay)
                    continue

            # 获取图像数据（使用阻塞锁，确保拍照时独占相机）
            print(f"[拍照] 尝试获取图像... (尝试 {attempt + 1}/{max_retries})")
            image_data = CAM.get_image(blocking=True)

            if image_data is None:
                error_msg = CAM.last_error or "获取图像失败"
                print(f"[拍照] 获取图像失败: {error_msg}")

                # 如果是特定错误，尝试重启相机
                if "超时" in error_msg or "0x8000000D" in error_msg:
                    print("检测到相机问题，尝试重启...")
                    CAM.restart_streaming()

                if attempt == max_retries - 1:
                    return jsonify({"success": False, "error": f"拍照失败: {error_msg}"})

                time.sleep(retry_delay)
                continue

            # 检查图像数据有效性
            if not hasattr(CAM, 'frame_info') or CAM.frame_info is None:
                print("图像信息无效")
                if attempt == max_retries - 1:
                    return jsonify({"success": False, "error": "图像信息无效"})
                time.sleep(retry_delay)
                continue

            # 处理图像数据
            try:
                temp = np.frombuffer(image_data, dtype=np.uint8)
                width = CAM.frame_info.nWidth
                height = CAM.frame_info.nHeight
                pixel_format = CAM.frame_info.enPixelType

                # 验证图像尺寸
                if width <= 0 or height <= 0:
                    print(f"无效的图像尺寸: {width}x{height}")
                    if attempt == max_retries - 1:
                        return jsonify({"success": False, "error": "无效的图像尺寸"})
                    time.sleep(retry_delay)
                    continue

                channels = 3 if pixel_format in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed] else 1

                # 根据像素格式处理图像
                if channels == 1 and pixel_format == PixelType_Gvsp_BayerGR8:
                    temp = temp.reshape((height, width))
                    temp = cv2.cvtColor(temp, cv2.COLOR_BayerGR2BGR)
                else:
                    temp = temp.reshape((height, width, channels))
                    if pixel_format == PixelType_Gvsp_RGB8_Packed:
                        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

                # 验证处理后的图像
                if temp is None or temp.size == 0:
                    print("图像处理失败")
                    if attempt == max_retries - 1:
                        return jsonify({"success": False, "error": "图像处理失败"})
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                print(f"图像处理异常: {e}")
                if attempt == max_retries - 1:
                    return jsonify({"success": False, "error": f"图像处理异常: {str(e)}"})
                time.sleep(retry_delay)
                continue

            # 生成唯一文件名：uuid + on_line + 日期
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]  # 取UUID的前8位
            filename = f"{unique_id}_on_line_{date_str}.jpg"
            session["img_name"] = filename

            print(f"[拍照] 生成文件名: {filename}")

            # 保存图片到online_cache缓存系统
            try:
                # 确保缓存目录存在
                os.makedirs(online_cache_manager.images_dir, exist_ok=True)
                print(f"[拍照] 缓存目录: {online_cache_manager.images_dir}")

                # 保存图片到 online_cache/on_images/ 目录（主要存储）
                cached_filename = f"{unique_id}_{date_str}_on_line.jpg"
                cached_path = os.path.join(online_cache_manager.images_dir, cached_filename)
                print(f"[拍照] 准备保存到: {cached_path}")

                # 保存图像文件到online_cache
                success = cv2.imwrite(cached_path, temp)
                if not success:
                    raise Exception("cv2.imwrite 保存失败")
                print(f"[拍照] cv2.imwrite 执行成功")

                # 验证文件是否成功保存
                if not os.path.exists(cached_path):
                    raise Exception(f"文件不存在: {cached_path}")
                    
                file_size = os.path.getsize(cached_path)
                if file_size == 0:
                    raise Exception(f"文件大小为0: {cached_path}")
                    
                print(f"[拍照] 文件验证成功，大小: {file_size} 字节")

                # 添加到在线缓存元数据
                online_cache_manager.add_image(
                    filename=filename,
                    cached_filename=cached_filename,
                    file_path=cached_path,
                    file_size=file_size
                )
                print(f"[拍照] 已添加到缓存元数据")

                # 释放相机缓冲区
                CAM.release_frame_buffer()

                # 拍照成功后，暂停视频流（释放资源，避免冲突）
                CAM.pause_streaming()

                # 构造返回路径
                image_path = f"/api/online/image/{cached_filename}"
                print(f"[拍照] ✅ 拍照成功！返回路径: {image_path}")

                return jsonify({
                    "success": True,
                    "image_path": image_path,
                    "image_name": filename,
                    "image_size": f"{width}x{height}",
                    "attempt": attempt + 1
                })

            except Exception as e:
                print(f"保存到本地缓存失败: {e}")
                CAM.release_frame_buffer()
                if attempt == max_retries - 1:
                    return jsonify({"success": False, "error": f"保存图片失败: {str(e)}"})
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"拍照过程异常: {e}")
            CAM.release_frame_buffer()
            if attempt == max_retries - 1:
                return jsonify({"success": False, "error": f"拍照异常: {str(e)}"})
            time.sleep(retry_delay)
            continue

    # 如果所有重试都失败了
    return jsonify({"success": False, "error": f"拍照失败，已重试 {max_retries} 次"})


@re.route("/recognition/image")
def serve_image():
    try:
        image_name = session.get("img_name")
        if not image_name:
            return jsonify({"success": False, "error": "No image name in session"})

        print(f"尝试显示图片: {image_name}")

        if OSS_AVAILABLE:
            # 优先尝试从OSS获取图片
            try:
                # 检查OSS中是否存在图片
                if check_image_exists("root", image_name):
                    image_url = get_image_url("root", image_name)
                    from flask import redirect
                    print(f"从OSS获取图片: {image_url}")
                    return redirect(image_url)
                else:
                    print(f"OSS中未找到图片: {image_name}，尝试本地存储")
            except Exception as e:
                print(f"从OSS获取图片失败: {e}，尝试本地存储")

        # 优先从online_cache查找图片
        print(f"从online_cache查找图片: {image_name}")
        cached_images = online_cache_manager.get_all_images()
        current_image = None
        for img in cached_images:
            if img.get("filename") == image_name:
                current_image = img
                break

        if current_image and os.path.exists(current_image["path"]):
            print(f"从online_cache找到图片: {current_image['path']}")
            return send_file(current_image["path"], mimetype="image/jpeg")

        # 如果online_cache中找不到，尝试从root目录查找（备用存储）
        print(f"online_cache中未找到图片，尝试从root备用目录查找")
        root_dir = r"E:\DengHuiXiong\python\flash\pythonWeb\pythonWeb\static\data\root"
        if os.path.exists(root_dir):
            image_path = os.path.join(root_dir, image_name)
            if os.path.exists(image_path):
                print(f"从root备用目录找到图片: {image_path}")
                return send_file(image_path, mimetype="image/jpeg")

        # 如果都找不到，返回最新的图片（从root备用目录）
        if os.path.exists(root_dir):
            files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
            if files:
                latest_file = max([os.path.join(root_dir, f) for f in files], key=os.path.getmtime)
                print(f"返回最新的备用图片: {latest_file}")
                return send_file(latest_file, mimetype="image/jpeg")

        print("没有找到任何图片")
        return jsonify({"success": False, "error": "No images found"})
    except Exception as e:
        print(f"serve_image异常: {e}")
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/upload_result", methods=["POST"])
def upload_result_image():
    """上传结果图片到云端（OSS） - 使用online_processing模块"""
    try:
        # 获取当前session中的图片名
        current_image_name = session.get("img_name")
        print(f"[上传结果] Session中的图片名: {current_image_name}")

        if not current_image_name:
            print("[上传结果] 错误: Session中没有图片名")
            return jsonify({"success": False, "error": "No image in session"})

        # 从online_cache中查找对应的结果
        results = online_cache_manager.get_all_results()
        print(f"[上传结果] 缓存中共有 {len(results)} 个结果")

        current_result = None
        for result in results:
            # 通过image_id查找对应的结果
            cached_images = online_cache_manager.get_all_images()
            for img in cached_images:
                if img.get("id") == result.get("image_id") and img.get("filename") == current_image_name:
                    current_result = result
                    print(f"[上传结果] 找到匹配的结果: {result}")
                    break
            if current_result:
                break

        if not current_result:
            print(f"[上传结果] 错误: 未找到图片 {current_image_name} 的结果")
            print(f"[上传结果] 可用的结果: {[r.get('id') for r in results]}")
            return jsonify({"success": False, "error": f"Result not found for image: {current_image_name}"})

        # 直接实现上传逻辑
        try:
            print(f"[上传结果] 准备读取结果图片: {current_result['result_path']}")
            print(f"[上传结果] 文件是否存在: {os.path.exists(current_result['result_path'])}")

            # 读取结果图片数据
            img = cv2.imread(current_result["result_path"])
            if img is None:
                print(f"[上传结果] 错误: 无法读取结果图片，路径: {current_result['result_path']}")
                return jsonify(
                    {"success": False, "error": f"无法读取结果图片数据，路径: {current_result['result_path']}"})

            print(f"[上传结果] 成功读取结果图片，形状: {img.shape}")

            # 上传到OSS云端 result/ 文件夹并写入数据库
            if OSS_AVAILABLE:
                try:
                    print(f"[上传结果] 开始上传到云端OSS...")
                    from utils.oss_client import upload_result_image as oss_upload_result
                    image_url = oss_upload_result(img, current_result["result_filename"])
                    print(f"[上传结果] ✅ 云端上传成功! URL: {image_url}")

                    # ⚠️ 重要：上传成功后，写入数据库
                    from utils import db
                    now = datetime.now()
                    create_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 使用云端URL作为数据库存储路径
                    db_image_path = image_url
                    
                    # 获取识别结果类别
                    result_class = current_result.get("result_class", "未识别")
                    
                    print(f"[数据库] 准备写入数据库:")
                    print(f"  - 时间: {create_time}")
                    print(f"  - 类别: {result_class}")
                    print(f"  - 用户: admin")
                    print(f"  - 图片URL: {db_image_path}")
                    
                    try:
                        # 插入在线检测结果
                        sql_online = 'insert into on_line (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        print(f"[数据库] 执行SQL: {sql_online}")
                        
                        db.insert(sql_online, (create_time, result_class, 'admin', db_image_path))
                        print(f"[数据库] ✅ on_line表插入成功")
                        
                        # 插入历史记录
                        sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        print(f"[数据库] 执行SQL: {sql_history}")
                        
                        db.insert(sql_history, (create_time, result_class, 'admin', db_image_path))
                        print(f"[数据库] ✅ history表插入成功")
                        
                        print(f"[上传结果] ✅✅✅ 数据库写入成功! 结果类别: {result_class} ✅✅✅")
                        
                    except Exception as e:
                        print(f"[上传结果] ❌❌❌ 数据库写入失败: {e}")
                        import traceback
                        traceback.print_exc()
                        return jsonify({
                            "success": False,
                            "error": f"数据库写入失败: {str(e)}",
                            "message": "云端上传成功，但数据库写入失败"
                        })
                    
                    # 清空online_cache
                    online_cache_manager.clear_images_cache()
                    online_cache_manager.clear_results_cache()
                    print("[上传结果] ✅ 缓存已清空")
                    
                    # 上传成功后，恢复视频流
                    CAM = camera_thread_manager.get_camera()
                    if CAM:
                        CAM.resume_streaming()

                    return jsonify({
                        "success": True,
                        "image_url": image_url,
                        "message": f"✅ 云端上传成功 → 数据库写入成功 → 缓存已清空 (结果: {result_class})"
                    })

                except Exception as e:
                    print(f"[上传结果] ❌ OSS上传失败: {current_result['result_filename']}, 错误: {e}")

                    # ⚠️ 上传失败时，保存到result目录作为备用并写入数据库
                    result_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'result')
                    os.makedirs(result_dir, exist_ok=True)
                    result_backup_path = os.path.join(result_dir, current_result["result_filename"])
                    cv2.imwrite(result_backup_path, img)
                    print(f"[上传结果] 结果图片已保存到result备用目录: {current_result['result_filename']}")

                    # ⚠️ 重要：即使OSS上传失败，也要写入数据库
                    from utils import db
                    now = datetime.now()
                    create_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 使用本地路径作为数据库存储路径
                    db_image_path = f"/static/data/result/{current_result['result_filename']}"
                    
                    # 获取识别结果类别
                    result_class = current_result.get("result_class", "未识别")
                    
                    print(f"[数据库] 准备写入数据库:")
                    print(f"  - 时间: {create_time}")
                    print(f"  - 类别: {result_class}")
                    print(f"  - 用户: admin")
                    print(f"  - 图片路径: {db_image_path}")
                    
                    try:
                        # 插入在线检测结果
                        sql_online = 'insert into on_line (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        print(f"[数据库] 执行SQL: {sql_online}")
                        
                        db.insert(sql_online, (create_time, result_class, 'admin', db_image_path))
                        print(f"[数据库] ✅ on_line表插入成功")
                        
                        # 插入历史记录
                        sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        print(f"[数据库] 执行SQL: {sql_history}")
                        
                        db.insert(sql_history, (create_time, result_class, 'admin', db_image_path))
                        print(f"[数据库] ✅ history表插入成功")
                        print(f"[上传结果] ✅✅✅ 数据库写入成功! 结果类别: {result_class} ✅✅✅")
                        
                    except Exception as e:
                        print(f"[上传结果] ❌❌❌ 数据库写入失败: {e}")
                        import traceback
                        traceback.print_exc()
                        return jsonify({
                            "success": False,
                            "error": f"数据库写入失败: {str(e)}",
                            "message": "结果图片已保存到本地备用目录，但数据库写入失败"
                        })
                    
                    # 清空online_cache
                    online_cache_manager.clear_images_cache()
                    online_cache_manager.clear_results_cache()
                    print("[上传结果] ✅ 缓存已清空")
                    
                    # 上传成功后，恢复视频流
                    CAM = camera_thread_manager.get_camera()
                    if CAM:
                        CAM.resume_streaming()

                    return jsonify({
                        "success": True,
                        "image_url": db_image_path,
                        "message": f"✅ 本地保存成功 → 数据库写入成功 → 缓存已清空 (结果: {result_class})"
                    })
            else:
                # ⚠️ OSS不可用时，保存到result目录作为备用并写入数据库
                print(f"[上传结果] ❌ OSS不可用")
                result_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'result')
                os.makedirs(result_dir, exist_ok=True)
                result_backup_path = os.path.join(result_dir, current_result["result_filename"])
                cv2.imwrite(result_backup_path, img)
                print(f"[上传结果] 结果图片已保存到result备用目录: {current_result['result_filename']}")

                # ⚠️ 重要：即使OSS不可用，也要写入数据库
                from utils import db
                now = datetime.now()
                create_time = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # 使用本地路径作为数据库存储路径
                db_image_path = f"/static/data/result/{current_result['result_filename']}"
                
                # 获取识别结果类别
                result_class = current_result.get("result_class", "未识别")
                
                print(f"[数据库] 准备写入数据库:")
                print(f"  - 时间: {create_time}")
                print(f"  - 类别: {result_class}")
                print(f"  - 用户: admin")
                print(f"  - 图片路径: {db_image_path}")
                
                try:
                    # 插入在线检测结果
                    sql_online = 'insert into on_line (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                    print(f"[数据库] 执行SQL: {sql_online}")
                    
                    db.insert(sql_online, (create_time, result_class, 'admin', db_image_path))
                    print(f"[数据库] ✅ on_line表插入成功")
                    
                    # 插入历史记录
                    sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                    print(f"[数据库] 执行SQL: {sql_history}")
                    
                    db.insert(sql_history, (create_time, result_class, 'admin', db_image_path))
                    print(f"[数据库] ✅ history表插入成功")
                    
                    print(f"[上传结果] ✅✅✅ 数据库写入成功! 结果类别: {result_class} ✅✅✅")
                    
                except Exception as e:
                    print(f"[上传结果] ❌❌❌ 数据库写入失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({
                        "success": False,
                        "error": f"数据库写入失败: {str(e)}",
                        "message": "结果图片已保存到本地备用目录，但数据库写入失败"
                    })
                
                # 清空online_cache
                online_cache_manager.clear_images_cache()
                online_cache_manager.clear_results_cache()
                print("[上传结果] ✅ 缓存已清空")
                
                # 上传成功后，恢复视频流
                CAM = camera_thread_manager.get_camera()
                if CAM:
                    CAM.resume_streaming()

                return jsonify({
                    "success": True,
                    "image_url": db_image_path,
                    "message": f"✅ 本地保存成功 → 数据库写入成功 → 缓存已清空 (结果: {result_class})"
                })

        except Exception as e:
            print(f"上传结果图片失败 {current_result['result_filename']}: {e}")
            return jsonify({"success": False, "error": f"上传失败: {str(e)}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/recognition/stream")
def stream():
    """视频流端点"""
    CAM = camera_thread_manager.get_camera()
    if not CAM or not CAM.device_status:
        # 返回空视频流或占位图
        def generate_placeholder():
            while True:
                # 生成一个占位图
                placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_img, "Camera Not Started", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(1)  # 1秒更新一次
        return Response(generate_placeholder(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(CAM.get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@re.route("/recognition/result_image")
def serve_result_image():
    """服务识别结果图片"""
    try:
        image_name = session.get("img_name")
        if not image_name:
            return jsonify({"success": False, "error": "No image name in session"})

        print(f"查找结果图片: {image_name}")

        if OSS_AVAILABLE:
            # 检查OSS中是否存在结果图片
            stem, ext = os.path.splitext(image_name)
            result_image_name = f"{stem}_pred.jpg"
            if check_image_exists("result", result_image_name):
                image_url = get_image_url("result", result_image_name)
                from flask import redirect
                return redirect(image_url)

        # 优先从online_cache/on_results/查找结果图片
        print(f"从online_cache/on_results/查找结果图片")
        results = online_cache_manager.get_all_results()
        current_result = None
        for result in results:
            # 检查结果是否对应当前图片
            if result.get("image_id"):
                # 通过image_id查找对应的图片
                cached_images = online_cache_manager.get_all_images()
                for img in cached_images:
                    if img.get("id") == result.get("image_id") and img.get("filename") == image_name:
                        current_result = result
                        break
                if current_result:
                    break

        if current_result and os.path.exists(current_result["result_path"]):
            print(f"从online_cache/on_results/找到结果图片: {current_result['result_path']}")
            return send_file(current_result["result_path"], mimetype="image/jpeg")

        # 如果online_cache中找不到，尝试从result备用目录查找
        print(f"online_cache中未找到结果图片，尝试从result备用目录查找")
        stem, ext = os.path.splitext(image_name)
        result_image_name = f"{stem}_pred.jpg"
        result_dir = r"E:\DengHuiXiong\python\flash\pythonWeb\pythonWeb\static\data\result"
        result_image_path = os.path.join(result_dir, result_image_name)

        if os.path.exists(result_image_path):
            print(f"从result备用目录找到结果图片: {result_image_path}")
            return send_file(result_image_path, mimetype="image/jpeg")

        print(f"未找到结果图片: {result_image_name}")
        return jsonify({"success": False, "error": f"Result image not found: {result_image_name}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# 在线缓存图片服务API
@re.route("/api/online/image/<path:filename>")
def serve_online_image(filename):
    """服务online_cache中的图片"""
    try:
        print(f"请求online_cache图片: {filename}")

        # 从online_cache/on_images/查找图片
        cached_images = online_cache_manager.get_all_images()
        for img in cached_images:
            if img.get("cached_filename") == filename or img.get("filename") == filename:
                if os.path.exists(img["path"]):
                    print(f"从online_cache/on_images/找到图片: {img['path']}")
                    return send_file(img["path"], mimetype="image/jpeg")

        print(f"未找到图片: {filename}")
        return jsonify({"success": False, "error": f"Image not found: {filename}"})
    except Exception as e:
        print(f"serve_online_image异常: {e}")
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/online/result/<path:filename>")
def serve_online_result(filename):
    """服务online_cache中的结果图片"""
    try:
        print(f"请求online_cache结果图片: {filename}")

        # 从online_cache/on_results/查找结果图片
        results = online_cache_manager.get_all_results()
        for result in results:
            if result.get("result_filename") == filename:
                if os.path.exists(result["result_path"]):
                    print(f"从online_cache/on_results/找到结果图片: {result['result_path']}")
                    return send_file(result["result_path"], mimetype="image/jpeg")

        print(f"未找到结果图片: {filename}")
        return jsonify({"success": False, "error": f"Result image not found: {filename}"})
    except Exception as e:
        print(f"serve_online_result异常: {e}")
        return jsonify({"success": False, "error": str(e)})


# 在线缓存管理API
@re.route("/api/online/cache_status")
def online_cache_status():
    """获取在线缓存状态"""
    try:
        stats = online_cache_manager.get_cache_stats()
        images = online_cache_manager.get_all_images()
        return jsonify({
            "success": True,
            "data": {
                "stats": stats,
                "images": images
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/online/results_status")
def online_results_status():
    """获取在线结果缓存状态"""
    try:
        results = online_cache_manager.get_all_results()
        return jsonify({
            "success": True,
            "data": {
                "results": results
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/online/clear_images", methods=["POST"])
def clear_online_images():
    """清空在线捕获图片缓存"""
    try:
        print("[清除捕获图] 收到清空在线捕获图片缓存请求")

        # 直接调用缓存管理器清除图片
        try:
            # 清空online_cache中的图片
            online_cache_manager.clear_images_cache()
            print("[清除捕获图] ✅ 已清空在线捕获图片缓存")
            
            # 同时清空session中的图片名
            if 'img_name' in session:
                del session['img_name']
                print("[清除捕获图] ✅ 已清空session中的图片名")
            
            # 清除成功后，恢复视频流
            CAM = camera_thread_manager.get_camera()
            if CAM:
                CAM.resume_streaming()
            
            return jsonify({
                "success": True,
                "message": "捕获图片缓存已清空"
            })
            
        except Exception as clear_error:
            print(f"[清除捕获图] ❌ 清空缓存失败: {clear_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"清空缓存失败: {str(clear_error)}"
            })

    except Exception as e:
        print(f"[清除捕获图] ❌ API错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/online/clear_results", methods=["POST"])
def clear_online_results():
    """清空在线结果图片缓存"""
    try:
        print("[清除结果图] 收到清空在线结果图片缓存请求")

        # 直接调用缓存管理器清除结果图片
        try:
            # 清空online_cache中的结果图片
            online_cache_manager.clear_results_cache()
            print("[清除结果图] ✅ 已清空在线结果图片缓存")
            
            return jsonify({
                "success": True,
                "message": "结果图片缓存已清空"
            })
            
        except Exception as clear_error:
            print(f"[清除结果图] ❌ 清空缓存失败: {clear_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"清空缓存失败: {str(clear_error)}"
            })

    except Exception as e:
        print(f"[清除结果图] ❌ API错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/recognize", methods=["POST"])
def online_recognize():
    """在线识别API - 简化版本，使用online_processing模块"""
    try:
        # 导入必要的模块
        from utils.model_pool import infer_image
        import shutil

        image_name = session.get("img_name")
        if not image_name:
            return jsonify({"success": False, "error": "No image in session"})

        # 从online_cache中查找图片进行识别
        cached_images = online_cache_manager.get_all_images()
        current_image = None
        for img in cached_images:
            if img.get("filename") == image_name:
                current_image = img
                break

        if not current_image:
            return jsonify({"success": False, "error": "Image not found in online cache"})

        # 设置临时目录
        tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # 更新状态为处理中
            online_cache_manager.update_image_status(current_image["id"], "processing")
            print(f"[识别] 开始识别图片: {current_image['filename']}")

            # 进行识别（保存到tmp目录）
            best_class, best_confidence, vis_img = infer_image(current_image["path"], tmp_dir, tmp_dir)
            confidence = round(float(best_confidence), 2)
            print(f"[识别] 识别完成 - 类别: {best_class}, 置信度: {confidence}")

            # 检查vis_img是否存在
            result_image_saved = False
            result_filename = None
            result_cached_path = None

            if vis_img is not None:
                # 确保结果目录存在
                os.makedirs(online_cache_manager.results_dir, exist_ok=True)

                # 生成结果文件名
                result_filename = f"result_{current_image['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                result_cached_path = os.path.join(online_cache_manager.results_dir, result_filename)

                # 直接保存vis_img到在线缓存
                try:
                    import cv2
                    cv2.imwrite(result_cached_path, vis_img)
                    result_image_saved = True
                    print(f"[识别] 结果图片已保存: {result_cached_path}")
                except Exception as e:
                    print(f"[识别] 保存结果图片失败: {e}")
            else:
                print(f"[识别] 警告: vis_img为None，无法保存结果图片")

            result_data = {
                "id": current_image["id"],
                "filename": current_image["filename"],
                "result_class": best_class,
                "confidence": confidence,
                "annotated_image": None,
                "status": "completed"
            }

            # 如果结果图片保存成功，添加到结果缓存
            if result_image_saved and result_cached_path:
                # 添加到结果缓存
                result_meta = online_cache_manager.add_result(
                    image_id=current_image["id"],
                    result_class=best_class,
                    confidence=confidence,
                    annotated_image_path=result_cached_path,
                    result_filename=result_filename,
                    result_path=result_cached_path
                )

                result_data["annotated_image"] = f"/api/online/result/{result_filename}"
                print(f"[识别] 结果已添加到缓存，ID: {result_meta['id']}")
            else:
                print(f"[识别] 警告: 未生成结果图片，无法添加到缓存")

            # 更新图片状态
            result_update_data = {
                "result_class": best_class,
                "confidence": confidence,
                "annotated_image": result_data["annotated_image"]
            }
            online_cache_manager.update_image_status(current_image["id"], "completed", result_update_data)
            print(f"[识别] 图片状态已更新为completed")

        except Exception as e:
            print(f"识别图片失败 {current_image['filename']}: {e}")
            online_cache_manager.update_image_status(current_image["id"], "error")
            return jsonify({"success": False, "error": f"识别失败: {str(e)}"})

        # 注意：识别完成后不立即写入数据库
        # 只有在成功上传到云端或保存到备用目录后才写入数据库
        print(f"识别完成，结果已保存到缓存，等待上传后写入数据库")

        return jsonify({
            "success": True,
            "data": {
                "result_class": result_data["result_class"],
                "confidence": result_data["confidence"],
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": image_name,
                "has_result_image": result_data.get("annotated_image") is not None,
                "result_image_path": result_data.get("annotated_image")
            }
        })

    except Exception as e:
        print(f"在线识别失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@re.route("/api/online/add_result", methods=["POST"])
def add_online_result():
    """添加识别结果到在线缓存"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        result_class = data.get('result_class')
        confidence = data.get('confidence')
        result_image_path = data.get('result_image_path')

        if not all([image_id, result_class, confidence, result_image_path]):
            return jsonify({"success": False, "error": "缺少必要参数"})

        # 生成结果文件名
        result_filename = f"result_{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        result_cached_path = os.path.join(online_cache_manager.results_dir, result_filename)

        # 复制结果图片到在线缓存
        import shutil
        shutil.copy2(result_image_path, result_cached_path)

        # 添加到结果缓存
        online_cache_manager.add_result(
            image_id=image_id,
            result_class=result_class,
            confidence=confidence,
            result_filename=result_filename,
            result_path=result_cached_path
        )

        return jsonify({"success": True, "message": "结果已添加到在线缓存"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})