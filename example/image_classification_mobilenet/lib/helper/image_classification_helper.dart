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

import 'dart:developer';
import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'isolate_inference.dart';

class ImageClassificationHelper {
  static const modelPath = 'assets/models/61-large_nearest_w_pre_fl16.tflite';
  static const binaryModelPath = 'assets/models/59-binary_large_w_pre_fl16.tflite';
  static const labelsPath = 'assets/models/custom_labels.txt';

  late final Interpreter interpreter;
  late final Interpreter binaryInterpreter;
  late final List<String> labels;
  late final IsolateInference isolateInference;
  late final IsolateInference binaryIsolateInference;
  late Tensor inputTensor;
  late Tensor outputTensor;
  late Tensor binaryInputTensor;
  late Tensor binaryOutputTensor;

  // Load model
  Future<void> _loadModel() async {
    final options = InterpreterOptions();

    // ... (rest of the delegate assignments)

    // Load model from assets
    interpreter = await Interpreter.fromAsset(modelPath, options: options);
    binaryInterpreter = await Interpreter.fromAsset(binaryModelPath, options: options);

    var primaryInputTensors = interpreter.getInputTensors();
    var primaryOutputTensors = interpreter.getOutputTensors();
    var binaryInputTensors = binaryInterpreter.getInputTensors();
    var binaryOutputTensors = binaryInterpreter.getOutputTensors();

    if (primaryInputTensors.isEmpty || primaryOutputTensors.isEmpty ||
        binaryInputTensors.isEmpty || binaryOutputTensors.isEmpty) {
      throw Exception('Tensor lists are empty! Model might be incompatible or not loaded correctly.');
    }

    // Get tensor input shape [1, 224, 224, 3]
    inputTensor = primaryInputTensors.first;
    // Get tensor output shape [1, 10]
    outputTensor = primaryOutputTensors.first;
    // Get tensor output shape [1, 1]
    binaryInputTensor = binaryInputTensors.first;
    binaryOutputTensor = binaryOutputTensors.first;

    log('Interpreter loaded successfully');
  }

  // Load labels from assets
  Future<void> _loadLabels() async {
    final labelTxt = await rootBundle.loadString(labelsPath);
    labels = labelTxt.split('\n');
  }

  Future<void> initHelper() async {
    _loadLabels();
    _loadModel();
    isolateInference = IsolateInference();
    await isolateInference.start();
    binaryIsolateInference = IsolateInference();
    await binaryIsolateInference.start();
  }

  Future<Map<String, double>> _inference(InferenceModel inferenceModel) async {
    ReceivePort responsePort = ReceivePort();
    isolateInference.sendPort
        .send(inferenceModel..responsePort = responsePort.sendPort);
    // get inference result.
    var results = await responsePort.first;
    return results;
  }

  Future<Map<String, double>> _binaryInference(InferenceModel inferenceModel) async {
    ReceivePort responsePort = ReceivePort();
    binaryIsolateInference.sendPort
        .send(inferenceModel..responsePort = responsePort.sendPort);
    // get inference result.
    var results = await responsePort.first;
    return results;
  }

  // inference camera frame
  Future<Map<String, double>> inferenceCameraFrame(CameraImage cameraImage) async {
    var isolateModel = InferenceModel(
        cameraImage,
        null,
        interpreter.address,
        binaryInterpreter.address,
        labels,
        inputTensor.shape,
        binaryInputTensor.shape,
        outputTensor.shape,
        binaryOutputTensor.shape
    );

    var primaryResults = await _inference(isolateModel);
    var binaryResults = await _binaryInference(isolateModel);

    // Combine the two results
    var combinedResults = {...primaryResults, ...binaryResults};

    return combinedResults;
  }

  Future<Map<String, double>> inferenceImage(Image image) async {
    var isolateModel = InferenceModel(
        null,
        image,
        interpreter.address,
        binaryInterpreter.address,
        labels,
        inputTensor.shape,
        binaryInputTensor.shape,
        outputTensor.shape,
        binaryOutputTensor.shape
    );

    var primaryResults = await _inference(isolateModel);
    var binaryResults = await _binaryInference(isolateModel);

    // Combine the two results
    var combinedResults = {...primaryResults, ...binaryResults};

    return combinedResults;
  }

  Future<void> close() async {
    isolateInference.close();
    binaryIsolateInference.close();
  }
}