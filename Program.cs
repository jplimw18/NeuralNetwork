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
           
        }
    }
}

