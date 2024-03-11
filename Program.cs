using RNA;

var dataset = new Dataset {
    Inputs = new double[][] {
        new double[] {1, 1},
        new double[] {1, 0},
        new double[] {0, 1},
        new double[] {0, 0}
    },
    Outputs = new double[][] {
        new double[] {0},
        new double[] {1},
        new double[] {1},
        new double[] {0}
    }
};

var classifier = new Classifier(dataset, dataset, 5);
classifier.Fit();

classifier.Predict(new double[] {1, 0});

class Dataset
{
    public double[][] Inputs { get; set; }
    public double[][] Outputs { get; set; }
}