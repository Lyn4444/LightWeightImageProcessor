package com.example.resnetimageprocessor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.resnetimageprocessor.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var performanceMonitor: PerformanceMonitor
    private var selectedImageBitmap: Bitmap? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        when {
            permissions.getOrDefault(getReadStoragePermissionName(), false) -> {
                // 存储权限已授予，检查相机权限
                if (isCameraPermissionRequired() && !hasCameraPermission()) {
                    requestCameraPermission()
                } else {
                    openGallery()
                }
            }
            permissions.getOrDefault(Manifest.permission.CAMERA, false) -> {
                // 相机权限已授予，但可能仍需要存储权限
                if (!hasReadStoragePermission()) {
                    requestStoragePermission()
                } else {
                    // 可以打开相机
                    // openCamera() // 如果需要相机功能，可以实现此方法
                }
            }
            else -> {
                Toast.makeText(this, "需要相关权限来使用应用功能", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                loadImageFromUri(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // 添加全局异常处理
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            android.util.Log.e("CrashHandler", "应用崩溃: ${throwable.message}", throwable)
            // 可以在这里添加崩溃日志上传或其他处理
        }
        
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imageProcessor = ImageProcessor(this)
        performanceMonitor = PerformanceMonitor()

        binding.btnSelectImage.setOnClickListener {
            checkPermissionAndOpenGallery()
        }

        binding.btnProcessImage.setOnClickListener {
            processSelectedImage()
        }
    }

    private fun checkPermissionAndOpenGallery() {
        when {
            hasReadStoragePermission() -> {
                openGallery()
            }
            else -> {
                requestStoragePermission()
            }
        }
    }
    
    private fun hasReadStoragePermission(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_MEDIA_IMAGES
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun getReadStoragePermissionName(): String {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
    }
    
    private fun requestStoragePermission() {
        val permissions = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        }
        requestPermissionLauncher.launch(permissions)
    }
    
    private fun requestCameraPermission() {
        requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }
    
    private fun isCameraPermissionRequired(): Boolean {
        // 不需要相机权限
        return false
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        pickImage.launch(intent)
    }

    private fun loadImageFromUri(uri: Uri) {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val inputStream = contentResolver.openInputStream(uri)
                    selectedImageBitmap = BitmapFactory.decodeStream(inputStream)
                    inputStream?.close()
                }

                selectedImageBitmap?.let {
                    binding.imageView.setImageBitmap(it)
                    binding.btnProcessImage.visibility = View.VISIBLE
                    binding.resultLayout.visibility = View.GONE
                }
            } catch (e: IOException) {
                Toast.makeText(this@MainActivity, "加载图片失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun processSelectedImage() {
        selectedImageBitmap?.let { bitmap ->
            binding.progressBar.visibility = View.VISIBLE
            binding.btnProcessImage.isEnabled = false

            lifecycleScope.launch {
                try {
                    // 开始性能监测
                    performanceMonitor.startMonitoring()

                    // 处理图像
                    val result = withContext(Dispatchers.Default) {
                        imageProcessor.processImage(bitmap)
                    }

                    // 停止性能监测并获取结果
                    val performanceStats = performanceMonitor.stopMonitoring()

                    if (result.error != null) {
                        // 处理错误情况
                        Toast.makeText(this@MainActivity, "处理失败: ${result.error.message}", Toast.LENGTH_LONG).show()
                        binding.resultLayout.visibility = View.GONE
                    } else {
                        // 显示处理后的图像
                        result.processedBitmap?.let { processedBitmap ->
                            binding.imageViewProcessed.setImageBitmap(processedBitmap)
                            
                            // 显示性能指标
                            binding.tvInferenceTime.text = getString(R.string.inference_time, result.inferenceTime.toString())
                            binding.tvFps.text = getString(R.string.fps_info, performanceStats.fps.toString())
                            binding.tvMemoryUsage.text = getString(R.string.memory_usage, performanceStats.memoryUsageMB.toString())
                            
                            // 显示设备芯片信息
                            binding.tvDeviceInfo.text = getString(R.string.device_model, result.deviceInfo)
                            binding.tvProcessorInfo.text = getString(R.string.processor_info, result.processorInfo)
                            
                            binding.resultLayout.visibility = View.VISIBLE
                        }
                    }
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, "处理图片失败: ${e.message}", Toast.LENGTH_SHORT).show()
                } finally {
                    binding.progressBar.visibility = View.GONE
                    binding.btnProcessImage.isEnabled = true
                }
            }
        } ?: run {
            Toast.makeText(this, "请先选择一张图片", Toast.LENGTH_SHORT).show()
        }
    }
}