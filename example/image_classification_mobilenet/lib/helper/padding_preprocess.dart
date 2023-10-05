import 'package:fraction/fraction.dart';
import 'package:image/image.dart' as image_lib;

//Approximates a function by another with a denominator less than
// maxDenominator.
Fraction limitDenominator(Fraction frac, int maxDenominator) {
  double val = frac.numerator / frac.denominator;
  int newNumerator = (val * maxDenominator).round();
  return Fraction(newNumerator, maxDenominator);
}

// Apply average pooling to an image with strides and padding of  size (i, j).
image_lib.Image averagePooling2D(image_lib.Image image, int i, int j) {
  int inputWidth = image.width;
  int inputHeight = image.height;

  // Calculate output dimensions
  int outputWidth = (inputWidth / i).ceil();
  int outputHeight = (inputHeight / j).ceil();

  // Create an output image with reduced dimensions
  image_lib.Image output = image_lib.Image(width:outputWidth, height:outputHeight);

  for (int y = 0; y < outputHeight; y++) {
    for (int x = 0; x < outputWidth; x++) {
      int rSum = 0, gSum = 0, bSum = 0, count = 0;

      // Iterate over the pooling window in the original image
      for (int py = 0; py < j && y * j + py < inputHeight; py++) {
        for (int px = 0; px < i && x * i + px < inputWidth; px++) {
          // Get pixel color from the original image
          image_lib.Pixel pixelColor = image.getPixel(x * i + px, y * j + py);
          rSum += pixelColor.r.toInt();
          gSum += pixelColor.g.toInt();
          bSum += pixelColor.b.toInt();
          count++;
        }
      }

      // Calculate average color
      int avgR = rSum ~/ count;
      int avgG = gSum ~/ count;
      int avgB = bSum ~/ count;

      // Set the average color to a pixel in the output image
      output.setPixelRgb(x, y, avgR, avgG, avgB);
    }
  }

  return output;
}

// Apply the averagePooling to an image with i, j with i, j selected as the
// aspect ratio values of the input image taking care that the output image is
// not lower resolution than 224,224
image_lib.Image? applyPoolingWithPadding(image_lib.Image? image) {
  // Return null if input is null
  if (image == null) {
    return null;
  }

  // Assuming you have an Image class that has width and height as properties.
  int width = image.width;
  int height = image.height;

  Fraction fracRatio = limitDenominator(Fraction(width, height).reduce(), (height/224).truncate());
  int originalWidthRatio = fracRatio.numerator;
  int originalHeightRatio = fracRatio.denominator;

  // Apply average pooling layer.
  final pooledImage = averagePooling2D(image, originalWidthRatio, originalHeightRatio);

  return pooledImage;
}