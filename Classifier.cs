namespace RNA
{
    internal class Classifier
    {
        private readonly NeuralNetwork _neuralNetwork;
        private readonly Dataset _trainingDataset;

        public Classifier(Dataset trainingDataset, Dataset validationDataset, int hiddenNodes)
        {
            _trainingDataset = trainingDataset;

            int numInputs = trainingDataset.Inputs[0].Length;
            int numOutputs = trainingDataset.Outputs[0].Length;
            _neuralNetwork = new NeuralNetwork(numInputs, hiddenNodes, numOutputs);
        }

        public void Fit()
        {
            for (int epoch = 0; epoch < 10000; epoch++)
            {
                var index = Random.Shared.Next(_trainingDataset.Inputs.Length);
                _neuralNetwork.Train(_trainingDataset.Inputs[index], _trainingDataset.Outputs[index]);
            }
            Console.WriteLine("Training complete!");
        }

        public double[] Predict(double[] input)
        {
            var result = Matrix.MatrixToArray(_neuralNetwork.Predict(input));
            Console.WriteLine("Prediction: " + result[0]);
            return result;
        }
    }
}
