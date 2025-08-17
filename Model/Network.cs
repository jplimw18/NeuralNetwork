using System;

namespace NeuralNetwork.Model
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public double Eps { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public double[][] TrainingData { get; set; }

        public Network(int ephocs, double[][] trainingData, double learningRate = 1e-4, double eps = 1e-4)
        {
            Epochs = ephocs;
            TrainingData = trainingData;
            LearningRate = learningRate;
            Eps = eps;
            Layers = new List<Layer>();
        }

        private double Cost(double output, double expected) => (expected * Math.Log(output) + (1 - expected) * Math.Log(1 - output)) / -TrainingData[0][..^1].Length;

        // TODO: Ajustar o BackPropagation para lidar com múltiplas camadas e pesos corretamente
        private void BackPropagation(double expected, double[] inputs)
        {
            List<double> deltas = new List<double>();

            foreach (var n in Layers.Last().Neurons)
                deltas.Add((n.Output - expected) * n.Output);

            for (int i = Layers.Count - 2; i >= 0; ++i)
            {

            }
        }

        private void ForwardPass(double[] inputs)
        {
            if (Layers.Count == 0)
                throw new InvalidOperationException("Network must have at least one layer.");

            double[] outputs = inputs;
            foreach (var layer in Layers)
            {
                layer.Activate(inputs);
                outputs = layer.Outputs;
            }
        }

        public void AddLayer(int layerSize)
        {
            var inputSize = Layers.Count == 0 ? TrainingData[0][..^1].Length : Layers.Last().LayerSize;
            Layers.Add(new Layer(layerSize, inputSize));
        }


    }
}
