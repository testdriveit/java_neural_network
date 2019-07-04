import java.util.Random;

public class NeuralNetwork {
    private double[][] wih;
    private double[][] who;
    private double[] hid_layer;
    private double[] outs;
    private double[] errh;
    private double[] erro;

    private int inp_length;
    private int hid_length;
    private int out_length;

    private final double LEARN_RATE = 0.2;

    public NeuralNetwork(int inp, int hid, int out){

        inp_length = inp;
        hid_length = hid;
        out_length = out;

        wih = new double[inp + 1][hid];
        who = new double[hid + 1][out];
        hid_layer = new double[hid];
        outs = new double[out];
        erro = new double[out];
        errh = new double[hid];

        for (int i = 0; i < hid; i++){
            hid_layer[i] = 0.0;
        }

        for (int i = 0; i < out; i++){
            outs[i] = 0.0;
        }

        randNetwork();
    }

    private void randNetwork(){
        for (int i = 0; i < inp_length+1; i++){
            for (int j = 0; j < hid_length; j++){
                wih[i][j] = Math.random();
            }
        }

        for (int i = 0; i < hid_length+1; i++){
            for (int j = 0; j < out_length; j++){
                who[i][j] = Math.random();
            }
        }
    }

    public void displayNetwork(){
        System.out.println("Input-hidden struscture: ");
        for (int i = 0; i < inp_length+1; i++){
            for (int j = 0; j < hid_length; j++){
                System.out.printf("%.2f ", wih[i][j]);
            }
            System.out.println();
        }

        System.out.println("Hidden layer output: ");
        for (int i = 0; i < hid_length; i++){
            System.out.printf("%.2f ", hid_layer[i]);
        }

        System.out.println();
        System.out.println("Hidden-output structure: ");
        for (int i = 0; i < hid_length+1; i++){
            for (int j = 0; j < out_length; j++){
                System.out.printf("%.2f ", who[i][j]);
            }
            System.out.println();
        }

        System.out.println("Out layer output: ");
        for (int i = 0; i < out_length; i++){
            System.out.printf("%.2f ", outs[i]);
        }

    }

    public void feedForward(double[] signal){
        double sum;
        for (int i = 0; i < hid_length; i++){
            sum = 0.0;
            for (int j = 0; j < inp_length; j++){
                sum += signal[j]*wih[j][i];
            }
            sum += wih[inp_length][i];
            hid_layer[i] = MathHelper.sigmoid(sum);
        }

        for (int i = 0; i < out_length; i++){
            sum = 0.0;
            for (int j = 0; j < hid_length; j++){
                sum +=  hid_layer[j]*who[j][i];
            }
            sum += who[hid_length][i];
            outs[i] = MathHelper.sigmoid(sum);
        }
    }

    public void backPropagation(double[] signal, double[] target){
        for (int i = 0; i < out_length; i++) {
            erro[i] = (target[i] - this.outs[i])*MathHelper.sigmoidDerivative(this.outs[i]);
        }

        for (int i = 0; i < hid_length; i++) {
            errh[i] = 0.0;
            for (int j = 0; j < out_length; j++){
                errh[i] += erro[j]*who[i][j];
            }
            errh[i] *= MathHelper.sigmoidDerivative(hid_layer[i]);
        }

        for (int i = 0; i < out_length; i++){
            for (int j = 0; j < hid_length; j++){
               who[j][i] += (LEARN_RATE * erro[i] * hid_layer[j]);
            }
            who[hid_length][i] += (LEARN_RATE * erro[i]);
        }

        for (int i = 0; i < hid_length; i++){
            for (int j = 0; j < inp_length; j++){
                wih[j][i] += (LEARN_RATE * errh[i] * signal[j]);
            }
            wih[inp_length][i] += (LEARN_RATE * errh[i]);
        }
    }

    public void trainIteration(double[] signal, double[] target){
        feedForward(signal);
        backPropagation(signal, target);
    }
}
