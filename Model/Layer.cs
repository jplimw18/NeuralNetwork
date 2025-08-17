using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    public class Layer
    {
        public int LayerSize { get; set; }
        public Neuron[] Neurons { get; set; }
        public double[] Outputs { get; set; }

        public Layer(int layerSize, int inputSize)
        {
            LayerSize = layerSize;
            Neurons = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++)
            {
                Neurons[i] = new Neuron(inputSize);
            }

            Outputs = new double[layerSize];
        }

        public double[] Activate(double[] inputs)
        {
            for (uint i = 0; i < LayerSize; ++i)
            {
                Outputs[i] = Neurons[i].Activate(inputs);
            }

            return Outputs;
        }
    }
}
