namespace RNA
{
    internal static class Function
    {
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double DSigmoid(double x)
        {
            return x * (1 - x);
        }
    }
}
