using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public double Output { get; set; }
        public double Delta { get; set; }
        public double PonderedSum { get; set; }


        public Neuron(int inputSize)
        {
            Random rand = new Random();
            InitializeWeights(inputSize, rand);
            PonderedSum = 0.0;
            Output = 0.0;
            Delta = 0.0;
        }

        private void InitializeWeights(int inputSize, Random random)
        {
            Weights = new double[inputSize];
            for (uint i = 0; i < inputSize; ++i)
            {
                Weights[i] = random.NextDouble();
            }

            Bias = random.NextDouble();
        }

        public void CalculateDelta(double[] w, double[] d)
        {
            for (uint i = 0; i < w.Length; ++i)
            {
                Delta += d[i] * w[i];
            }

            Delta *= Output;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        public double Activate(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("Input size must match weights size.");

            for (uint i = 0; i < inputs.Length; ++i)
            {
                PonderedSum += inputs[i] * Weights[i];
            }
            Output = Sigmoid(PonderedSum);

            return Output;
        }

        public void UpdateWeights(double learningRate, double[] inputs)
        {
            for (uint i = 0; i < Weights.Length; ++i)
            {
                Weights[i] -= learningRate * Delta * inputs[i];
            }

            Bias -= learningRate * Delta;
        }
    }
}
