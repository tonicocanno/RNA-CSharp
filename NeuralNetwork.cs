namespace RNA
{
    internal class NeuralNetwork
    {
        int I_nodes { get; set; }
        int H_nodes { get; set; }
        int O_nodes { get; set; }
        Matrix Bias_ih { get; set; }
        Matrix Bias_ho { get; set; }
        Matrix Weights_ih { get; set; }
        Matrix Weights_ho { get; set; }
        float Learning_rate { get; set; }

        public NeuralNetwork(int i_nodes, int h_nodes, int o_nodes, float learning_rate = 0.1f)
        {
            I_nodes = i_nodes;
            H_nodes = h_nodes;
            O_nodes = o_nodes;
            Learning_rate = learning_rate;

            Weights_ih = new Matrix(H_nodes, I_nodes);
            Weights_ih.XavierInitializer(H_nodes, I_nodes);
            Weights_ho = new Matrix(O_nodes, H_nodes);
            Weights_ho.XavierInitializer(O_nodes, H_nodes);

            Bias_ih = new Matrix(H_nodes, 1);
            Bias_ih.Randomize(-0.01f, 0.01f);
            Bias_ho = new Matrix(O_nodes, 1);
            Bias_ho.Randomize(-0.01f, 0.01f);
        }

        public void Train(double[] array, double[] target)
        {
            var input = Matrix.ArrayToMatrix(array);
            var hidden = Matrix.Activate(Matrix.Add(Matrix.Multiply(Weights_ih, input), Bias_ih));

            var output = Matrix.Activate(Matrix.Add(Matrix.Multiply(Weights_ho, hidden), Bias_ho));

            var expected = Matrix.ArrayToMatrix(target);
            var outputError = Matrix.Subtract(expected, output);

            var outputGradient = Matrix.Hadamard(outputError, Matrix.ActivateDerivative(output));
            Bias_ho = Matrix.Add(Bias_ho, Matrix.EscalarMultiply(outputGradient, Learning_rate));
            Weights_ho = Matrix.Add(Weights_ho, Matrix.Multiply(outputGradient, Matrix.Transpose(hidden)));

            var hiddenError = Matrix.Multiply(Matrix.Transpose(Weights_ho), outputError);
            var hiddenGradient = Matrix.Hadamard(hiddenError, Matrix.ActivateDerivative(hidden));

            Bias_ih = Matrix.Add(Bias_ih, Matrix.EscalarMultiply(hiddenGradient, Learning_rate));
            Weights_ih = Matrix.Add(Weights_ih, Matrix.Multiply(hiddenGradient, Matrix.Transpose(input)));
        }

        public Matrix Predict(double[] array)
        {
            var input = Matrix.ArrayToMatrix(array);

            var hidden = Matrix.Activate(Matrix.Add(Matrix.Multiply(Weights_ih, input), Bias_ih));
            var output = Matrix.Activate(Matrix.Add(Matrix.Multiply(Weights_ho, hidden), Bias_ho));

            return output;
        }

        public Matrix PredictBatch(double[][] inputs)
        {
            return new Matrix(inputs.Length, 1).Map((element, i, j) =>
            {
                return Predict(inputs[i])[0, 0];
            });
        }
    }
}
