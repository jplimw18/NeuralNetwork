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
            var data = GetAndSample();
            
            var network = new Network(100, data, 1e-4);
            network.AddLayer(2);
            network.AddLayer(3);
            network.AddLayer(1);

            network.Train();

            Console.WriteLine("Training complete. Testing the network...");
            foreach (var sample in data)
            {
                var output = network.Predict(sample.Take(sample.Length - 1).ToArray());
                Console.WriteLine($"Input: {string.Join(", ", sample.Take(sample.Length - 1))} => Output: {(output[0] > 0.5 ? 1 : 0)} (Expected: {sample.Last()}), Raw: {output[0]}");
            }
        }
    }
}

