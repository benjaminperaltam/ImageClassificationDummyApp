import 'dart:io';
import 'package:image/image.dart';
import 'padding_preprocess.dart';

Image? imagePreprocessing(Image img) {
  return applyPoolingWithPadding(img);
}

void main(List<String> arguments) {
  // Paths
  var inputPath = '4.jpeg';   // Replace with your input path
  var outputPath = 'output.jpg'; // Replace with your output path

  // Read image
  Image? image = decodeImage(File(inputPath).readAsBytesSync());

  if (image != null) {
    // Process image
    Image? processedImage = imagePreprocessing(image);

    // Save processed image
    File(outputPath).writeAsBytesSync(encodeJpg(processedImage!));
    print('Image saved to $outputPath, shape: (${processedImage.width}, ${processedImage.height})');
  } else {
    print('Failed to load image from $inputPath');
  }
}
