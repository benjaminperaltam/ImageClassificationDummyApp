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
import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_classification_mobilenet/helper/image_classification_helper.dart';

class ModelInfo {
  final String name;
  final String path;

  ModelInfo(this.name, this.path);
}

double? goodAvg;
double? isCarAvg;
List<ModelInfo> models = [
  ModelInfo('Model 61', 'assets/models/61-large_nearest_w_pre_fl16.tflite'),
  ModelInfo('Model 81', 'assets/models/81-large_nearest_w_pre_11class(datasetV7)_fl16.tflite'),
  ModelInfo('Model 83', 'assets/models/83-large_nearest_w_pre_square_pad_11class(datasetV7)_fl16.tflite'),
  ModelInfo('Model 91', 'assets/models/91-large_nearest_w_pre_11class(datasetV10)_fl16.tflite')
  // ... Add more models as required
];
var selectedModel =  models[0];

class CameraScreen extends StatefulWidget {
  const CameraScreen({
    Key? key,
    required this.camera,
  }) : super(key: key);

  final CameraDescription camera;

  @override
  State<StatefulWidget> createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  late CameraController cameraController;
  late ImageClassificationHelper imageClassificationHelper;
  Map<String, double>? classification;
  bool _isProcessing = false;
  List<double> primaryPredictions = [];
  List<double> binaryPredictions = [];

  // init camera
  initCamera() {
    cameraController = CameraController(widget.camera, ResolutionPreset.medium,
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420);
    cameraController.initialize().then((value) {
      cameraController.startImageStream(imageAnalysis);
      if (mounted) {
        setState(() {});
      }
    });
  }

  Future<void> imageAnalysis(CameraImage cameraImage) async {
    // if image is still analyze, skip this frame
    if (_isProcessing) {
      return;
    }
    _isProcessing = true;
    classification =
        await imageClassificationHelper.inferenceCameraFrame(cameraImage);
    var primaryPrediction = classification?["Good"] ?? 0.0;
    primaryPredictions.add(primaryPrediction);

    var binaryPrediction = classification?["IsCar"] ?? 0.0;
    binaryPredictions.add(binaryPrediction);

    // This is where we ensure the averages are always present in the classification
    if (goodAvg != null) {
      classification?['GoodAvg'] = goodAvg!;
    }
    if (isCarAvg != null) {
      classification?['IsCarAvg'] = isCarAvg!;
    }

    _isProcessing = false;
    if (mounted) {
      setState(() {});
    }
  }

  Timer? periodicTimer;

  @override
  void initState() {
    super.initState();

    periodicTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (primaryPredictions.isNotEmpty) {
        goodAvg = primaryPredictions.reduce((a, b) => a + b) / primaryPredictions.length;
        isCarAvg = binaryPredictions.reduce((a, b) => a + b) / binaryPredictions.length;

        primaryPredictions = [];
        binaryPredictions = [];
        setState(() {});
      }
    });
    WidgetsBinding.instance.addObserver(this);
    initCamera();
    imageClassificationHelper = ImageClassificationHelper(modelPath: selectedModel.path);
    imageClassificationHelper.initHelper();
    super.initState();
  }

  @override
  Future<void> didChangeAppLifecycleState(AppLifecycleState state) async {
    switch (state) {
      case AppLifecycleState.paused:
        cameraController.stopImageStream();
        break;
      case AppLifecycleState.resumed:
        if (!cameraController.value.isStreamingImages) {
          await cameraController.startImageStream(imageAnalysis);
        }
        break;
      default:
    }
  }

  @override
  void dispose() {
    periodicTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    cameraController.dispose();
    imageClassificationHelper.close();
    super.dispose();
  }

  Widget cameraWidget(context) {
    var camera = cameraController.value;
    // fetch screen size
    final size = MediaQuery.of(context).size;

    // calculate scale depending on screen and camera ratios
    // this is actually size.aspectRatio / (1 / camera.aspectRatio)
    // because camera preview size is received as landscape
    // but we're calculating for portrait orientation
    var scale = size.aspectRatio * camera.aspectRatio;

    // to prevent scaling down, invert the value
    if (scale < 1) scale = 1 / scale;

    return Transform.scale(
      scale: scale,
      child: Center(
        child: CameraPreview(cameraController),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Size size = MediaQuery.of(context).size;
    List<Widget> list = [];

    list.add(
      SizedBox(
        child: (!cameraController.value.isInitialized)
            ? Container()
            : cameraWidget(context),
      ),
    );
    list.add(Align(
      alignment: Alignment.bottomCenter,
      child: SingleChildScrollView(
        child: Column(
          children: [
            if (classification != null)
              ...(classification!.entries.toList()
                    ..sort(
                      (a, b) => a.value.compareTo(b.value),
                    ))
                  .reversed
                  .take(8)
                  .map(
                    (e) => Container(
                      padding: const EdgeInsets.all(8),
                      color: Colors.white.withOpacity(0.8),
                      child: Row(
                        children: [
                          Text(e.key),
                          const Spacer(),
                          Text(e.value.toStringAsFixed(2))
                        ],
                      ),
                    ),
                  ),
          ],
        ),
      ),
    ));
    list.add(
        Positioned(  // Using Positioned to place the dropdown at the top
          top: 20,   // Adjust as needed
          left: 20,  // Adjust as needed
          child: ModelSelector(
            models: models,
            onModelSelected: (model) {
              setState(() {
                selectedModel = model;
                imageClassificationHelper = ImageClassificationHelper(modelPath: selectedModel.path);
                imageClassificationHelper.initHelper();
              });
            },
          ),
        )
    );

    return SafeArea(
      child: Stack(
        children: list,
      ),
    );
  }
}

class ModelSelectorController {
  ModelInfo? selectedModel;

  void changeModel(ModelInfo model) {
    selectedModel = model;
  }
}

class ModelSelector extends StatefulWidget {
  final List<ModelInfo> models;
  final ValueChanged<ModelInfo> onModelSelected;
  final ModelSelectorController? controller;

  const ModelSelector({
    Key? key,
    required this.models,
    required this.onModelSelected,
    this.controller,
  }) : super(key: key);

  @override
  ModelSelectorState createState() => ModelSelectorState();
}

class ModelSelectorState extends State<ModelSelector> {
  ModelInfo? selectedModel;

  @override
  void initState() {
    super.initState();
    selectedModel = widget.models.isNotEmpty ? widget.models[0] : null;

    // Check if controller is provided and set the initial value
    if (widget.controller != null && selectedModel != null) {
      widget.controller!.changeModel(selectedModel!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return DropdownButton<ModelInfo>(
      value: selectedModel,
      onChanged: (ModelInfo? newValue) {
        if (newValue != null) {
          setState(() {
            selectedModel = newValue;
          });
          if (widget.controller != null) {
            widget.controller!.changeModel(newValue);
          }
          widget.onModelSelected(newValue);
        }
      },
      items: widget.models.map<DropdownMenuItem<ModelInfo>>((ModelInfo value) {
        return DropdownMenuItem<ModelInfo>(
          value: value,
          child: Text(value.name),
        );
      }).toList(),
    );
  }
}

class ParentWidget extends StatefulWidget {
  const ParentWidget({Key? key}) : super(key: key);
  @override
  ParentWidgetState createState() => ParentWidgetState();
}


class ParentWidgetState extends State<ParentWidget> {
  final controller = ModelSelectorController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Model Selector Example')),
      body: Column(
        children: [
          ModelSelector(
            models: models,
            onModelSelected: (model) {
              // This is called whenever a new model is selected
            },
            controller: controller,
          ),
          ElevatedButton(
            onPressed: () {
              // Access the selected model from the controller
            },
            child: const Text('Check Selected Model'),
          ),
        ],
      ),
    );
  }
}
