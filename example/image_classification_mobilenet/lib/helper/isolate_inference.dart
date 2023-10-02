/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:io';
import 'dart:isolate';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as image_lib;
import 'package:image_classification_mobilenet/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'qa_helper.dart';

class IsolateInference {
  static const String _debugName = "TFLITE_INFERENCE";
  final ReceivePort _receivePort = ReceivePort();
  late Isolate _isolate;
  late SendPort _sendPort;

  SendPort get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(entryPoint, _receivePort.sendPort,
        debugName: _debugName);
    _sendPort = await _receivePort.first;
  }

  Future<void> close() async {
    _isolate.kill();
    _receivePort.close();
  }

  static void entryPoint(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);

    await for (final InferenceModel isolateModel in port) {
      image_lib.Image? img = getPreprocessedImage(isolateModel);

      // First (primary) inference
      var classification1 = runInference(img, isolateModel, modelType: 'primary');

      // Second (binary) inference
      var classification2 = runInference(img, isolateModel, modelType: 'binary');

      // Concatenate and send results (or do any other desired operations with them)
      var concatenatedResults = {...classification1, ...classification2};
      isolateModel.responsePort.send(concatenatedResults);
    }
  }

  static image_lib.Image? getPreprocessedImage(InferenceModel isolateModel) {
    if (isolateModel.isCameraFrame()) {
      return ImageUtils.convertCameraImage(isolateModel.cameraImage!);
    } else {
      return isolateModel.image;
    }
  }

  static Map<String, double> runInference(image_lib.Image? img, InferenceModel model, {required String modelType}) {
    // resize original image to match model shape.
    image_lib.Image imageInput = image_lib.copyResize(
      img!,
      width: model.inputShape[1],
      height: model.inputShape[2],
    );

    if (Platform.isAndroid && model.isCameraFrame()) {
      imageInput = image_lib.copyRotate(imageInput, angle: 0);
    }

    final imageMatrix = List.generate(
      imageInput.height,
          (y) =>
          List.generate(
            imageInput.width,
                (x) {
              final pixel = imageInput.getPixel(x, y);
              return [
                pixel.r.toDouble(),
                pixel.g.toDouble(),
                pixel.b.toDouble()
              ];
            },
          ),
    );

    // Set tensor input [1, 224, 224, 3]
    final outputShape = (modelType == 'binary') ? model.binaryOutputShape : model.outputShape;
    final interpreterAddress = (modelType == 'binary') ? model.binaryInterpreterAddress : model.interpreterAddress;

    final input = [imageMatrix];
    final output = (modelType == 'binary') ? [List<int>.filled(outputShape[1], 0)] : [List<double>.filled(outputShape[1], 0)];

    Interpreter interpreter = Interpreter.fromAddress(interpreterAddress);
    interpreter.run(input, output);

    var outValue = 0.0;
    var outName = "";
    if (modelType == 'binary'){
      outName = 'IsCar';
      outValue = output.first.last.toDouble();
    }else{
      outName = 'Good';
      List<double> doubleList = output.first.map((i) => i.toDouble()).toList();
      outValue = QAHelper.combinedPostProcessing([doubleList]).first.last.toDouble();
    }
    return <String, double>{outName: output.first.last.toDouble()};
    // ... rest of your processing logic, return results as needed.
    //var classification = <String, double>{};
    //if(modelType == 'binary'){
    //  classification['isCar'] = output.first.last.toDouble();
    //}else {
    // var result = output.first;
      //final postprocess = QAHelper.combinedPostProcessing([result]);
    // classification['Good'] = output.first.last.toDouble();
    }
}

class InferenceModel {
  CameraImage? cameraImage;
  image_lib.Image? image;
  int interpreterAddress;
  int binaryInterpreterAddress;
  List<String> labels;
  List<int> inputShape;
  List<int> binaryInputShape;
  List<int> outputShape;
  List<int> binaryOutputShape;
  late SendPort responsePort;

  InferenceModel(
  this.cameraImage,
  this.image,
  this.interpreterAddress,
  this.binaryInterpreterAddress,
  this.labels,
  this.inputShape,
  this.binaryInputShape,
  this.outputShape,
  this.binaryOutputShape
  );

  // check if it is camera frame or still image
  bool isCameraFrame() {
    return cameraImage != null;
  }
}
