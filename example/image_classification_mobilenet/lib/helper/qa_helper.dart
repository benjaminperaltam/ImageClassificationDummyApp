
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
// This is a helper class for the Image QA feature of the app.
class QAHelper {

  static List<double> computeValues(double goodPercentage) {
    try {
      const numClasses = 10;
      final remainingValue = (1.0 - goodPercentage) / (numClasses - 1);

      return List.filled(numClasses - 1, remainingValue) + [goodPercentage];
    } catch (error) {
      debugPrint("Error in computeValues(): $error");
      rethrow;
    }
  } // END computeValues()

  /// Post-process the output data.
  ///
  /// This is used to get more accurate predictions.
  /// It also allows the model to make more nuanced predictions.
  /// Instead of only predicting something like 0% or 95% in confidence percentage,
  /// the post processing allows the model to predict more in-between values like 50% or 75%.
  ///
  /// Returns a list of adjusted confidence percentages for each class of the image QA model.
  static List<List<double>> combinedPostProcessing(

      /// inputs - The output data from the model. This is a list of confidence percentages for each class of the image QA model.
      List<List<double>> inputs) {
    try {
      // Calculate the index of the maximum value in the output
      final maxIndex = inputs
          .map((e) =>
              // ignore: avoid_types_as_parameter_names
              e.indexWhere((num) => num == e.reduce((a, b) => a > b ? a : b)))
          .toList();

      // Define valid ranges for the good class depending on the class with the
      // maximum probability. The closer an image is to the correct angle the
      // higher is its valid range (Ex. class 0 is Passenger Side and class 6 is
      // Front and their valid values are between 0.55 and 0.65)
      Map<int, List<double>> classRanges = {};

      if (inputs.first.length == 11) {
        classRanges = {
          0: [0.75, 0.8], // Pass side
          1: [0.5, 0.55], // Pass back
          2: [0.2, 0.25], // Back
          3: [0.45, 0.5], // Driver back
          4: [0.55, 0.6], // Driver side
          5: [0.7, 0.75], // Driver Front
          6: [0.75, 0.8], // Front
          7: [0.3, 0.4], // ZOOM in
          8: [0.3, 0.4], // ZOOM out
          9: [0.7, 0.75] // Fit
        };

      }else if(inputs.first.length == 10){
        classRanges = {
          0: [0.75, 0.8], // Pass side
          1: [0.5, 0.55], // Pass back
          2: [0.2, 0.25], // Back
          3: [0.45, 0.5], // Driver back
          4: [0.55, 0.6], // Driver side
          5: [0.7, 0.75], // Driver Front
          6: [0.75, 0.8], // Front
          7: [0.3, 0.4], // ZOOM in
          8: [0.3, 0.4], // ZOOM out
        };
      }

      // Empty list to store the output values
      final outputs = List.generate(
          inputs.length, (index) => List.filled(inputs[index].length, 0.0));

      classRanges.forEach((idx, range) {
        for (var i = 0; i < maxIndex.length; i++) {
          if (maxIndex[i] == idx) {
            // Randomly choose a value for the class good within the valid range
            // and saves it in the output array.
            final class7Val =
                math.Random().nextDouble() * (range[1] - range[0]) + range[0];
            outputs[i] = computeValues(class7Val);
          }
        }
      });

      final goodIndex = classRanges.length;
      for (var i = 0; i < maxIndex.length; i++) {
        if (maxIndex[i] == goodIndex) {
          final jitter = inputs[i][goodIndex] +
              math.Random().nextDouble() * (-0.05 - 0.05) +
              0.05;
          // Fix min and max values for the 7-th class depending on its new value
          if (inputs[i][goodIndex] > 0.8) {
            outputs[i] = computeValues(jitter.clamp(0.99, 1.0));
          } else if (inputs[i][goodIndex] > 0.6 && inputs[i][goodIndex] <= 0.8) {
            outputs[i] = computeValues(jitter.clamp(0.95, 0.99));
          } else if (inputs[i][goodIndex] > 0.4 && inputs[i][goodIndex] <= 0.6){
            outputs[i] = computeValues(jitter.clamp(0.9, 0.95));
          } else {
            outputs[i] = computeValues(jitter.clamp(0.8, 0.9));
          }
        }
      }

      /// Return a list of adjusted confidence percentages for each class of the image QA model.
      return outputs;
    } catch (error) {
      debugPrint("Error in combinedPostProcessing(): $error");

      // If this function fails, return the inputs.
      // We can still get usable results from the inputs.
      return inputs;
    }
  } // END combinedPostProcessing()

} // End QAHelper class