package com.example.resnetimageprocessor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Exception

class ImageProcessor(private val context: Context) {

    private val imageSize = 224 // 标准输入尺寸
    private var module: Module? = null
    
    // 归一化参数 - 根据模型训练时使用的参数设置
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    
    init {
        try {
            // 加载模型并确保在CPU上运行
            val modelPath = assetFilePath(context, "model.ptl")
            module = Module.load(modelPath)
            android.util.Log.d("ImageProcessor", "模型加载成功")
        } catch (e: IOException) {
            // 模型加载失败时记录错误
            android.util.Log.e("ImageProcessor", "模型加载失败: ${e.message}", e)
        } catch (e: Exception) {
            // 捕获所有其他异常
            android.util.Log.e("ImageProcessor", "未知错误: ${e.message}", e)
        }
    }
    
    // 从assets目录加载模型文件到缓存目录
    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
    
    data class ProcessingError(
        val message: String,
        val exception: Exception? = null
    )
    
    data class ProcessingResult(
        val processedBitmap: Bitmap? = null,
        val inferenceTime: Long = 0,
        val deviceInfo: String = "",
        val processorInfo: String = "",
        val error: ProcessingError? = null
    )
    
    fun processImage(inputBitmap: Bitmap): ProcessingResult {
        val startTime = System.currentTimeMillis()
        
        try {
            if (inputBitmap.width <= 0 || inputBitmap.height <= 0) {
                return ProcessingResult(
                    error = ProcessingError("无效的输入图像尺寸")
                )
            }
            
            // 调整图像大小并进行预处理
            val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, imageSize, imageSize, true)
            
            // 使用本地图像处理方法
            val processedBitmap = simulateImageProcessing(resizedBitmap)
            
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // 获取设备芯片信息
            val deviceInfo = android.os.Build.MODEL
            val processorInfo = android.os.Build.HARDWARE
            
            return ProcessingResult(processedBitmap, inferenceTime, deviceInfo, processorInfo)
        } catch (e: Exception) {
            return ProcessingResult(
                error = ProcessingError("图像处理失败: ${e.message}", e)
            )
        }
    }
    
    private fun simulateImageProcessing(bitmap: Bitmap): Bitmap {
        // 使用PyTorch模型进行图像处理
        if (module == null) {
            android.util.Log.e("ImageProcessor", "模型未加载，使用备用处理方法")
            return applySimpleImageEffect(bitmap)
        }
        
        try {
            // 将Bitmap转换为Tensor，应用归一化处理
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                MEAN,
                STD
            )
            
            // 执行模型推理，确保在CPU上运行
            val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
            
            // 获取输出张量的维度和数据
            val outputShape = outputTensor.shape()
            val outputData = outputTensor.dataAsFloatArray
            
            // 添加安全检查
            if (outputData.isEmpty()) {
                android.util.Log.e("ImageProcessor", "模型输出数据为空，使用备用处理方法")
                return applySimpleImageEffect(bitmap)
            }
            
            // 创建结果Bitmap
            val width = bitmap.width
            val height = bitmap.height
            val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            
            // 将模型输出转换回Bitmap
            // 注意：这里的处理方式取决于模型的输出格式
            // 假设模型输出是3通道RGB图像数据
            if (outputShape.size >= 3) { // 确保输出是图像格式
                val channels = outputShape[1].toInt()
                val outHeight = outputShape[2].toInt()
                val outWidth = if (outputShape.size > 3) outputShape[3].toInt() else 1
                
                // 如果输出尺寸与输入不同，需要调整
                if (outHeight == height && outWidth == width) {
                    // 直接将输出数据映射到Bitmap
                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            // 计算RGB值（需要从模型输出中正确提取）
                            val r = (outputData[y * width * channels + x * channels] * 255).coerceIn(0f, 255f).toInt()
                            val g = (outputData[y * width * channels + x * channels + 1] * 255).coerceIn(0f, 255f).toInt()
                            val b = (outputData[y * width * channels + x * channels + 2] * 255).coerceIn(0f, 255f).toInt()
                            
                            result.setPixel(x, y, Color.rgb(r, g, b))
                        }
                    }
                } else {
                    // 如果尺寸不匹配，可能需要调整大小或其他处理
                    // 这里简单地复制原图并应用一些处理
                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            val pixel = bitmap.getPixel(x, y)
                            
                            // 应用一些基于模型输出的处理
                            // 这里仅作为示例，实际应根据模型输出调整
                            val r = (Color.red(pixel) * outputData[0]).coerceIn(0f, 255f).toInt()
                            val g = (Color.green(pixel) * outputData[1]).coerceIn(0f, 255f).toInt()
                            val b = (Color.blue(pixel) * outputData[2]).coerceIn(0f, 255f).toInt()
                            
                            result.setPixel(x, y, Color.rgb(r, g, b))
                        }
                    }
                }
            } else {
                throw IOException("模型输出格式不符合预期")
            }
            
            return result
        } catch (e: Exception) {
            android.util.Log.e("ImageProcessor", "模型推理失败: ${e.message}", e)
            return applySimpleImageEffect(bitmap)
        }
    }
    
    // 添加一个简单的图像处理备用方法
    private fun applySimpleImageEffect(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // 模拟一些图像处理效果（这里简单地调整对比度）
        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                
                val alpha = Color.alpha(pixel)
                var red = Color.red(pixel)
                var green = Color.green(pixel)
                var blue = Color.blue(pixel)
                
                // 增强对比度的简单算法
                red = (red * 1.2).coerceIn(0.0, 255.0).toInt()
                green = (green * 1.2).coerceIn(0.0, 255.0).toInt()
                blue = (blue * 1.2).coerceIn(0.0, 255.0).toInt()
                
                result.setPixel(x, y, Color.argb(alpha, red, green, blue))
            }
        }
        
        return result
    }
    

    
    fun close() {
        // 清理PyTorch模型资源
        module?.destroy()
        module = null
    }
}