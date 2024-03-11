namespace RNA
{
    internal class Matrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; }
        private double[]? _data;

        public double this[int i, int j]
        {
            get => _data![i * Columns + j];
            set => _data![i * Columns + j] = value;
        }

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Columns = cols;
            _data = new double[rows * cols];
        }

        public Matrix(double[][] data)
        {
            ValidateArrayInput(data);

            Rows = data.Length;
            Columns = data[0].Length;
            _data = new double[Rows * Columns];

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    _data[i * Columns + j] = data[i][j];
                }
            }
        }

        public Matrix(double[,] data)
        {
            ValidateArrayInput(data);

            Rows = data.GetLength(0);
            Columns = data.GetLength(1);
            _data = new double[Rows * Columns];

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = data[i, j];
                }
            }
        }

        public static Matrix ArrayToMatrix(double[] array) => new Matrix(array.Length, 1) { _data = array };

        public static double[] MatrixToArray(Matrix matrix) => matrix._data!;

        public void Print()
        {
            Console.WriteLine($"Matrix:");
            for (int i = 0; i < Rows; i++)
            {
                Console.Write($"[");
                for (int j = 0; j < Columns; j++)
                {
                    Console.Write($"{this[i, j]} ");
                }
                Console.WriteLine("]");
            }
        }

        public void Randomize(float min = 0, float max = 1f)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = (float)Random.Shared.NextDouble() * (max - min) + min;
                }
            }
        }

        public void XavierInitializer(int output_size, int input_size)
        {
            float standard_deviation = MathF.Sqrt(2.0f / (input_size + output_size));

            Matrix weights = new Matrix(output_size, input_size);
            weights.Randomize(-standard_deviation, standard_deviation);

            _data = weights._data;
        }

        public static Matrix Map(Matrix matrix, Func<double, int, int, double> func)
        {
            var result = new Matrix(matrix.Rows, matrix.Columns);

            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    result[i, j] = func(matrix[i, j], i, j);
                }
            }

            return result;
        }

        public Matrix Map(Func<double, int, int, double> func)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = func(this[i, j], i, j);
                }
            }

            return this;
        }

        public void Fill(double value)
        {
            for (int i = 0; i < _data!.Length; i++)
            {
                _data[i] = value;
            }
        }

        public static Matrix Transpose(Matrix m1)
        {
            var result = new Matrix(m1.Columns, m1.Rows);

            result.Map((element, i, j) =>
            {
                return m1[j, i];
            });

            return result;
        }

        public Matrix Transpose()
        {
            var result = new Matrix(Columns, Rows);

            result.Map((element, i, j) =>
            {
                return this[j, i];
            });

            return result;
        }

        public static Matrix Add(Matrix m1, Matrix m2)
        {
            ValidateMatrixDimensions(m1, m2);

            var result = new Matrix(m1.Rows, m1.Columns);

            result.Map((element, i, j) =>
            {
                return m1[i, j] + m2[i, j];
            });

            return result;
        }

        public static Matrix Subtract(Matrix m1, Matrix m2)
        {
            ValidateMatrixDimensions(m1, m2);

            var result = new Matrix(m1.Rows, m1.Columns);

            result.Map((element, i, j) =>
            {
                return m1[i, j] - m2[i, j];
            });

            return result;
        }

        public static Matrix Multiply(Matrix m1, Matrix m2)
        {
            ValidateMultiplicationDimensions(m1, m2);

            var result = new Matrix(m1.Rows, m2.Columns);

            result.Map((element, i, j) =>
            {
                double sum = 0;
                for (int k = 0; k < m1.Columns; k++)
                {
                    sum += m1[i, k] * m2[k, j];
                }

                return sum;
            });

            return result;
        }

        public static Matrix EscalarMultiply(Matrix m1, double escalar)
        {
            var result = new Matrix(m1.Rows, m1.Columns);

            result.Map((element, i, j) =>
            {
                return m1[i, j] * escalar;
            });

            return result;
        }

        public static Matrix Hadamard(Matrix m1, Matrix m2)
        {
            var result = new Matrix(m1.Rows, m1.Columns);

            result.Map((element, i, j) =>
            {
                return m1[i, j] * m2[i, j];
            });

            return result;
        }

        public static Matrix Activate(Matrix matrix)
        {
            return matrix.Map((element, i, j) =>
            {
                return Function.Sigmoid(element);
            });
        }

        public static Matrix ActivateDerivative(Matrix matrix)
        {
            return matrix.Map((element, i, j) =>
            {
                return Function.DSigmoid(element);
            });
        }

        public static Matrix Sum(Matrix matrix)
        {
            var result = new Matrix(matrix.Rows, matrix.Columns);

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = matrix[i, j];
                }
            }

            return result;
        }

        public static Matrix Convolution(Matrix input, Matrix kernel)
        {
            // Verificações de dimensões
            if (input.Rows < kernel.Rows || input.Columns < kernel.Columns)
            {
                throw new ArgumentException("Input matrix must be larger than kernel matrix");
            }

            int outputRows = input.Rows - kernel.Rows + 1;
            int outputCols = input.Columns - kernel.Columns + 1;

            var output = new Matrix(outputRows, outputCols);

            for (int i = 0; i < outputRows; i++)
            {
                for (int j = 0; j < outputCols; j++)
                {
                    double sum = 0;

                    for (int k = 0; k < kernel.Rows; k++)
                    {
                        for (int l = 0; l < kernel.Columns; l++)
                        {
                            sum += input[i + k, j + l] * kernel[k, l];
                        }
                    }

                    output[i, j] = sum;
                }
            }

            return output;
        }

        private static void ValidateArrayInput(Array array)
        {
            if (array.Length == 0)
            {
                throw new ArgumentException("Matrix must have at least one row and one column");
            }
        }

        private static void ValidateMatrixDimensions(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Columns != m2.Columns)
            {                
                throw new ArgumentException("Matrices must have the same dimensions for the operation");
            }
        }

        private static void ValidateMultiplicationDimensions(Matrix m1, Matrix m2)
        {
            if (m1.Columns != m2.Rows)
            {
                throw new ArgumentException("Matrices must have compatible dimensions for multiplication");
            }
        }
    }
}
