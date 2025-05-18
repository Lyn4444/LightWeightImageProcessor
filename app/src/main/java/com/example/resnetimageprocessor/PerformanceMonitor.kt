package com.example.resnetimageprocessor

import android.os.Debug

class PerformanceMonitor {

    private var startTime: Long = 0
    private var frameCount: Int = 0
    
    data class PerformanceStats(
        val fps: Float,
        val memoryUsageMB: Float
    )
    
    fun startMonitoring() {
        startTime = System.currentTimeMillis()
        frameCount = 0
        // 触发垃圾回收以获得更准确的内存使用情况
        Runtime.getRuntime().gc()
    }
    
    fun stopMonitoring(): PerformanceStats {
        frameCount++
        val endTime = System.currentTimeMillis()
        val elapsedTimeSeconds = (endTime - startTime) / 1000f
        
        // 计算FPS
        val fps = if (elapsedTimeSeconds > 0) frameCount / elapsedTimeSeconds else 0f
        
        // 获取内存使用情况
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        val memoryUsageMB = memoryInfo.totalPrivateDirty / 1024f
        
        return PerformanceStats(fps, memoryUsageMB)
    }
    
    fun recordFrame() {
        frameCount++
    }
}