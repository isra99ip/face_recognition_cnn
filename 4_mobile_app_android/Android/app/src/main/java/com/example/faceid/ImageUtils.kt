package com.example.faceid

import android.graphics.*
import androidx.camera.core.ImageProxy

fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
    val yuv = out.toByteArray()
    var bmp = BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    if (imageInfo.rotationDegrees != 0) {
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    }
    return bmp
}

fun Bitmap.centerCropResize(size: Int): Bitmap {
    val s = minOf(width, height)
    val x = (width - s) / 2
    val y = (height - s) / 2
    val square = Bitmap.createBitmap(this, x, y, s, s)
    return Bitmap.createScaledBitmap(square, size, size, true)
}
