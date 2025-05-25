package com.example.resnetimageprocessor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Exception

class ImageProcessor(
    private val context: Context,
) {
    private val imageSize = 256 // 标准输入尺寸
    private var module: Module? = null

    // 归一化参数 - 根据模型训练时使用的参数设置
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        try {
            // 加载模型并确保在CPU上运行
            val modelPath = assetFilePath(context, "model_deblur.ptl")
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
    private fun assetFilePath(
        context: Context,
        assetName: String,
    ): String {
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
        val exception: Exception? = null,
    )

    data class ProcessingResult(
        val processedBitmap: Bitmap? = null,
        val inferenceTime: Long = 0,
        val deviceInfo: String = "",
        val processorInfo: String = "",
        val error: ProcessingError? = null,
    )


    fun processImage(inputBitmap: Bitmap): ProcessingResult {
        try {
            if (inputBitmap.width <= 0 || inputBitmap.height <= 0) {
                return ProcessingResult(
                    error = ProcessingError("无效的输入图像尺寸"),
                )
            }
            val startTime = System.currentTimeMillis()

            // 调整图像大小并进行预处理
            val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, imageSize, imageSize, true)

            // 使用模拟处理获取结果
            val processedBitmapResized = simulateImageProcessing(resizedBitmap)

            // 检查处理后的结果大小是否与原始resizedBitmap大小一致
            val isSameSize = (processedBitmapResized.width == resizedBitmap.width &&
                    processedBitmapResized.height == resizedBitmap.height)

            // 日志记录结果大小信息
            android.util.Log.d(
                "ImageProcessor",
                "原始调整大小后: ${resizedBitmap.width}x${resizedBitmap.height}"
            )
            android.util.Log.d(
                "ImageProcessor",
                "处理后大小: ${processedBitmapResized.width}x${processedBitmapResized.height}"
            )

            // 根据大小是否一致决定如何处理
            val finalBitmap = if (!isSameSize) {
                // 如果大小不一致，先调整到与resizedBitmap相同大小
                android.util.Log.d(
                    "ImageProcessor",
                    "处理结果大小与原调整图像不一致，先调整到相同大小"
                )
                val adjustedBitmap = Bitmap.createScaledBitmap(
                    processedBitmapResized,
                    resizedBitmap.width,
                    resizedBitmap.height,
                    true
                )
                // 再调整到原始输入大小
                Bitmap.createScaledBitmap(
                    adjustedBitmap,
                    inputBitmap.width,
                    inputBitmap.height,
                    true
                )
            } else {
                // 如果大小一致，直接调整到原始输入大小
                android.util.Log.d(
                    "ImageProcessor",
                    "处理结果大小与原调整图像一致，直接调整到原始输入大小"
                )
                Bitmap.createScaledBitmap(
                    processedBitmapResized,
                    inputBitmap.width,
                    inputBitmap.height,
                    true
                )
            }

            val inferenceTime = System.currentTimeMillis() - startTime

            // 获取设备芯片信息
            val deviceInfo = android.os.Build.MODEL
            val processorInfo = android.os.Build.HARDWARE

            return ProcessingResult(finalBitmap, inferenceTime, deviceInfo, processorInfo)
        } catch (e: Exception) {
            return ProcessingResult(
                error = ProcessingError("图像处理失败: ${e.message}", e),
            )
        }
    }


    private fun simulateImageProcessing(bitmap: Bitmap): Bitmap {
        if (module == null) {
            android.util.Log.e("ImageProcessor", "模型未加载，使用备用处理方法")
            return applySimpleImageEffect(bitmap)
        }

        try {
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                MEAN,
                STD,
            )

            val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
            val outputShape = outputTensor.shape()
            val outputData = outputTensor.dataAsFloatArray

            if (outputData.isEmpty()) {
                android.util.Log.e("ImageProcessor", "模型输出数据为空，使用备用处理方法")
                return applySimpleImageEffect(bitmap)
            }

            // 从输出张量确定实际尺寸
            if (outputShape.size < 3) {
                android.util.Log.e("ImageProcessor", "模型输出维度不足，使用备用处理方法")
                return applySimpleImageEffect(bitmap)
            }

            val channels = outputShape[1].toInt()
            val outHeight = outputShape[2].toInt()
            val outWidth = if (outputShape.size > 3) outputShape[3].toInt() else 1

            android.util.Log.d("ImageProcessor", "输出张量形状: [${channels}, ${outHeight}, ${outWidth}]")

            // 创建与输出张量尺寸匹配的位图
            val result = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)

            // 将张量数据填充到位图中
            for (y in 0 until outHeight) {
                for (x in 0 until outWidth) {
                    // 计算每个像素在outputData中的索引
                    // 根据PyTorch张量的存储格式：[batch, channel, height, width]
                    // 对于给定位置(y,x)，我们需要计算对应的r,g,b值索引

                    // 索引计算方式：batch_idx * C * H * W + c * H * W + y * W + x
                    // 由于我们处理单个图像，batch_idx = 0，可以简化为：
                    // c * H * W + y * W + x 或 c * (H * W) + y * W + x

                    val rIndex = 0 * outHeight * outWidth + y * outWidth + x
                    val gIndex = 1 * outHeight * outWidth + y * outWidth + x
                    val bIndex = 2 * outHeight * outWidth + y * outWidth + x

                    // 确保索引在有效范围内
                    val r = if (rIndex < outputData.size) (outputData[rIndex] * 255).coerceIn(0f, 255f).toInt() else 0
                    val g = if (gIndex < outputData.size) (outputData[gIndex] * 255).coerceIn(0f, 255f).toInt() else 0
                    val b = if (bIndex < outputData.size) (outputData[bIndex] * 255).coerceIn(0f, 255f).toInt() else 0

                    result.setPixel(x, y, Color.rgb(r, g, b))
                }
            }

            android.util.Log.d("ImageProcessor", "成功将输出张量转换为位图: ${result.width}x${result.height}")
            return result
        } catch (e: Exception) {
            android.util.Log.e("ImageProcessor", "模型推理或张量转换失败: ${e.message}", e)
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
