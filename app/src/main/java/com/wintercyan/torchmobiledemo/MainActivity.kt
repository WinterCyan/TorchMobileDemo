package com.wintercyan.torchmobiledemo

import android.graphics.Bitmap
import android.graphics.Color.rgb
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.core.graphics.drawable.toBitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.InputStream
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private var estBtn: Button? = null
    private var estModule: Module? = null
    private var fullImgView: ImageView? = null
    private var partialImgView: ImageView? = null
    private var matImgView: ImageView? = null
    private var maskT: Tensor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        estBtn = findViewById(R.id.estimate_button)
        fullImgView = findViewById(R.id.full_img)
        partialImgView = findViewById(R.id.partial_img)
        matImgView = findViewById(R.id.mat_img)

//        val modelName = "traced_mobile.ptl"
        val modelName = "mobilenetv3.pt"
        val maskName = "mask.csv"
        var modelPath = ""
        val file = File("${Environment.getExternalStorageDirectory()}/Model", modelName)
        if (file.exists() && file.length()>0) {
            modelPath = file.absolutePath
            Log.d(TAG, "model path: $modelPath")
        }
//        estModule = Module.load(modelPath)
        estModule = LiteModuleLoader.load(modelPath)
        Log.d(TAG, estModule.toString())

        val reader = File("${Environment.getExternalStorageDirectory()}/Model", maskName).bufferedReader()
        val maskArr = reader.lineSequence()
            .filter { it.isNotBlank() }
            .map {
                it.toFloat()
            }
            .toList()
            .toFloatArray()
        val maskShape = longArrayOf(512,1,1)
        maskT = Tensor.fromBlob(maskArr, maskShape)

        Log.d(TAG, "mask shape: ${maskT!!.shape()}")
        Log.d(TAG, "loaded mask: $maskT")

        estBtn!!.setOnClickListener {
            val fullBitmap = fullImgView!!.drawable.toBitmap();
            val partialBitmap = partialImgView!!.drawable.toBitmap();
            val fullInputT = TensorImageUtils.bitmapToFloat32Tensor(fullBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)
            val partialInputT = TensorImageUtils.bitmapToFloat32Tensor(partialBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)
            val outputT = estModule!!.forward(IValue.from(fullInputT), IValue.from(partialInputT), IValue.from(maskT!!)).toTensor()
            val outputArray = outputT.dataAsFloatArray
            println("output shape: ${outputT.shape()}")
            val outputBitmap = floatArrayToBitmap(outputArray, 256*4, 256)
            matImgView!!.setImageBitmap(outputBitmap)
        }
    }

    private fun floatArrayToBitmap(floatArray: FloatArray, width: Int, height: Int) : Bitmap {

        // Create empty bitmap in ARGB format
        val bmp: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.max() ?: 1.0f
        val minValue = floatArray.min() ?: -1.0f
        val delta = maxValue-minValue

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = { v: Float -> ((v-minValue)/delta*255.0f).roundToInt()}

        // copy each value from float array to RGB channels
        for (i in 0 until width * height) {
            val r = conversion(floatArray[i])
            val g = conversion(floatArray[i+width*height])
            val b = conversion(floatArray[i+2*width*height])
            pixels[i] = rgb(r, g, b) // you might need to import for rgb()
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)

        return bmp
    }

    private fun readCsv(inputStream: InputStream): FloatArray {
        val reader = inputStream.bufferedReader()
        return reader.lineSequence()
            .filter { it.isNotBlank() }
            .map {
                it.toFloat()
            }
            .toList()
            .toFloatArray()
    }

    companion object {
        const val TAG = "WINTERCYAN"
    }
}