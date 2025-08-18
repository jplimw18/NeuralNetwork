using System;

namespace NeuralNetwork.Model
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        //public double Eps { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public double[][] TrainingData { get; set; }

        public Network(int ephocs, double[][] trainingData, double learningRate = 1e-4)
        {
            Epochs = ephocs;
            TrainingData = trainingData;
            LearningRate = learningRate;
            //Eps = eps;
            Layers = new List<Layer>();
        }

        // TODO: Ajustar o BackPropagation para lidar com múltiplas camadas e pesos corretamente
        private void BackPropagation(double expected)
        {
            List<(double, double[])> deltas = new();

            foreach (var n in Layers.Last().Neurons)
            {
                double delta = (n.Output - expected) * n.Output * (1 - n.Output);
                deltas.Add((delta, n.Weights));
                n.Delta = delta;
            }

            for (int i = Layers.Count - 2; i >= 0; --i)
            {
                var layer = Layers[i];

                for (uint j = 0; j < layer.LayerSize; ++j)
                {
                    var neuron = layer.Neurons[j];
                    double[] d = deltas.Select(x => x.Item1).ToArray();
                    double[] w = deltas.Select(x => x.Item2[j]).ToArray();
                    neuron.CalculateDelta(w, d);
                }

                deltas.Clear();
                foreach (var n in layer.Neurons)
                    deltas.Add((n.Delta, n.Weights));
            }
        }

        private void UpdateWeights(double[] inputs)
        {
            foreach (var layer in Layers)
            {

                foreach (var n in layer.Neurons)
                    n.UpdateWeights(LearningRate, inputs);
                
                inputs = layer.Outputs;
            }
        }

        private void ForwardPass(double[] inputs)
        {
            if (Layers.Count == 0)
                throw new InvalidOperationException("Network must have at least one layer.");

            double[] outputs = inputs;
            foreach (var layer in Layers)
            {
                layer.Activate(outputs);
                outputs = layer.Outputs;
            }
        }

        public void AddLayer(int layerSize)
        {
            var inputSize = Layers.Count == 0 ? TrainingData[0][..^1].Length : Layers.Last().LayerSize;
            Layers.Add(new Layer(layerSize, inputSize));
        }

        public void Train()
        {
            for (uint e = 0; e < Epochs; ++e)
            {
                for (uint i = 0; i < TrainingData.Length; ++i)
                {
                    double[] inputs = TrainingData[i][..^1];
                    double expected = TrainingData[i][^1];

                    ForwardPass(inputs);
                    BackPropagation(expected);
                    UpdateWeights(inputs);
                }
            }
        }


        public double[] Predict(double[] inputs)
        {
            ForwardPass(inputs);
            return Layers.Last().Outputs;
        }
    }
}
