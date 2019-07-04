public class Program {

    public static final int epoch = 15000;

    public static final double[][] signal = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    public static final double[][] target = {{0.0}, {1.0}, {1.0}, {0.0}};

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
        double[] testSignal = {1.0, 0.0};

        for (int i = 0; i < epoch; i++){
            for (int j = 0; j < signal.length; j++){
                nn.trainIteration(signal[j], target[j]);
            }
        }

        nn.feedForward(testSignal);
        nn.displayNetwork();
    }
}
