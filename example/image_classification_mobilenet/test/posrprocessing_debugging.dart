import 'dart:math';

const Map<List<double>, List<double>> maxUpscaleRanges = {
[0, 0.64102462]: [0.5, 0.6],
[0.64102462, 0.65128077]: [0.6, 0.7],
[0.65128077, 0.67179308]: [0.7, 0.8],
[0.67179308, 0.69230538]: [0.8, 0.85],
[0.69230538, 0.73333]: [0.85, 0.9],
[0.73333, 0.79486692]: [0.9, 0.95],
[0.79486692, 0.97947769]: [0.95, 0.98],
[0.97947769, 0.98973385]: [0.98, 0.99],
[0.98973385, 1]: [0.99, 1],
};

List<double> postProcess(List<double> yPred, Map<List<double>, List<double>> maxUpscaleRanges) {
  const pMin = 1e-4;

  // Replacing values in yPred if they're less than pMin
  for (int i = 0; i < yPred.length; i++) {
    if (yPred[i] < pMin) yPred[i] = 0;
  }

  final dp = (1 - yPred.reduce((curr, next) => curr + next)) / yPred.length;
  for (int i = 0; i < yPred.length; i++) {
    yPred[i] += dp;
  }

  final maxYPred = yPred.reduce((curr, next) => curr > next ? curr : next);
  final yClass = yPred.indexOf(maxYPred);

  final postY = List<double>.filled(yPred.length, 0.0);

  for (final key in maxUpscaleRanges.keys) {
    final lbound = key[0];
    final ubound = key[1];
    if (maxYPred > lbound && maxYPred <= ubound) {
      final newMax = (maxUpscaleRanges[key]![1] - maxUpscaleRanges[key]![0]) * Random().nextDouble() + maxUpscaleRanges[key]![0];
      postY[yClass] = newMax;

      final remainingP = 1 - newMax;
      List<double> pDist = List.from(yPred)
        ..removeAt(yClass);

      final total = pDist.reduce((a, b) => a + b);

      if (total == 0) {
        for (int i = 0; i < yClass; i++) {
          postY[i] = remainingP / (yPred.length - 1);
        }
        for (int i = yClass + 1; i < yPred.length; i++) {
          postY[i] = remainingP / (yPred.length - 1);
        }
      } else {
        for (int i = 0; i < yClass; i++) {
          postY[i] = pDist[i] * remainingP / total;
        }
        for (int i = yClass + 1, j = yClass; j < yPred.length - 1; i++, j++) {
          postY[i] = pDist[j] * remainingP / total;
        }
      }
    }
  }

  return postY;
}

void main() {
  List<double> sampleList = [0.1, 0.1, 0, 0, 0.2, 0, 0, 0.5, 0, 0.1];
  List<double> result = postProcess(sampleList, maxUpscaleRanges);
  print(result);
}