public class DigitAddition {
    public static final int epoch = 15000;
    
    public static void main(String[] args) {
        final double[][] digits = new double[100][2];
        for (int i = 0; i < 100; i++) {
            digits[i][0] = i / 10;
            digits[i][1] = i % 10;
        }

        final double[][] target = new double[100][19];
        for (int i = 0; i < 100; i++){
            target[i][i/10 + i%10] = 1;
        }



        double[] testData = {5, 6};
        NeuralNetwork nn = new NeuralNetwork(2, 30, 19);

        nn.displayNetwork();

        for (int i = 0; i < epoch; i++){
            for (int j = 0; j < digits.length; j++){
                nn.trainIteration(digits[j], target[j]);
            }
        }

        nn.feedForward(testData);
        nn.displayNetwork();

    }
}
