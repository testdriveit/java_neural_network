public class DigitRecognition {

    public static final int epoch = 15000;

    public static final double[][] digits = {
            // 0
            {1, 1, 1,
             1, 0, 1,
             1, 0, 1,
             1, 0, 1,
             1, 0, 1,
             1, 1, 1},
            // 1
            {0, 0, 1,
             0, 1, 1,
             1, 0, 1,
             0, 0, 1,
             0, 0, 1,
             0, 0, 1},
            // 2
            {1, 1, 1,
             1, 0, 1,
             0, 0, 1,
             0, 1, 0,
             1, 0, 0,
             1, 1, 1},
            // 3
            {1, 1, 1,
             0, 1, 0,
             1, 1, 1,
             0, 0, 1,
             0, 0, 1,
             1, 1, 1},
            // 4
            {0, 0, 1,
             0, 1, 1,
             1, 0, 1,
             1, 1, 1,
             0, 0, 1,
             0, 0, 1}
/*
            {1.0, 1.0},

            {1.0, 1.0},

            {1.0, 1.0},

            {1.0, 1.0},

            {1.0, 1.0},*/
    };

    public static final double[][] target =
            {{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}};

    public static void main(String[] args) {
        double[] testSignal =
                // 3
                {1, 1, 1,
                 0, 1, 0,
                 1, 1, 1,
                 0, 0, 1,
                 1, 0, 1,
                 1, 1, 1};

        NeuralNetwork nn = new NeuralNetwork(18, 10, 5);

        for (int i = 0; i < epoch; i++){
            for (int j = 0; j < digits.length; j++){
                nn.trainIteration(digits[j], target[j]);
            }
        }

        nn.feedForward(testSignal);
        nn.displayNetwork();

    }
}
