using NeuralNetwork.Model;

namespace NeuralNetwork
{
    class Program
    {
        static double[][] GetAndSample()
        {
            return new double[][]
            {
                new double[] { 0, 0, 0 },
                new double[] { 0, 1, 0 },
                new double[] { 1, 0, 0 },
                new double[] { 1, 1, 1 },
            };
        }
        static double[][] GetOrSample()
        {
            return new double[][]
            {
                new double[] { 0, 0, 0 },
                new double[] { 0, 1, 1 },
                new double[] { 1, 0, 1 },
                new double[] { 1, 1, 1 },
            };
        }
        static double[][] GetNandSample()
        {
            return new double[][]
            {
                new double[] { 0, 0, 1 },
                new double[] { 0, 1, 1 },
                new double[] { 1, 0, 1 },
                new double[] { 1, 1, 0 },
            };
        }
        static double[][] GetXorSample()
        {
            return new double[][]
            {
                new double[] { 0, 0, 0 },
                new double[] { 0, 1, 1 },
                new double[] { 1, 0, 1 },
                new double[] { 1, 1, 0 },
            };
        }

        static void Main(string[] args)
        {
            double[][] data = null!;

            //data = GetAndSample();
            //data = GetOrSample();
            //data = GetNandSample();
            data = GetXorSample();


            var network = new Network(1000 * 1000, data, 1e-3);
            //network.AddLayer(2);
            //network.AddLayer(3);
            network.AddLayer(2);
            network.AddLayer(1);

            network.Train();

            Console.WriteLine("Training complete. Testing the network...");
            foreach (var sample in data)
            {
                var output = network.Predict(sample.Take(sample.Length - 1).ToArray());
                foreach (var o in output)
                {
                    Console.WriteLine($"O/Neuron {output.ToList().IndexOf(o)}: inputs-> {sample[..^1][0]}, {sample[..^1][1]} => (Expected: {sample[2]}), out => {(o > 0.5 ? 1: 0)} (raw: {o})");
                }
            }
        }
    }
}

