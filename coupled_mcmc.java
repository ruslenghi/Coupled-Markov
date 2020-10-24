package coupledmcmc;

import beast.core.Distribution;
import beast.core.Input;
import beast.core.Runnable;
import beast.core.parameter.RealParameter;
import beast.util.Randomizer;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static coupledmcmc.CoupledMCMCBasic.Markov_Step;

public class CoupledMCMC extends Runnable {

    public Input<Integer> numberOfChainsInput = new Input<>("numberOfChains",
            "Number of of coupled mcmc chains.", Input.Validate.REQUIRED);

    public Input<RealParameter> xInput = new Input<>("x",
            "Parameter to sample using coupled MCMC.",
            Input.Validate.REQUIRED);

    public Input<Distribution> distributionInput = new Input<>("distribution",
            "Distribution to sample from",
            Input.Validate.REQUIRED);

    public Input<String> outputFileInput = new Input<>("outputFile",
            "Name of output file.",
            Input.Validate.REQUIRED);

    int numberOfChains;
    RealParameter x;
    Distribution distribution;

    @Override
    public void initAndValidate() {
        numberOfChains = numberOfChainsInput.get();
        x = xInput.get();
        distribution = distributionInput.get();

        // Example usages follow:

        // x.getValue(); // Get _current_ value of parameter.
        // x.setValue(3.0);  // Set _new_ value of parameter.
        // double logP = distribution.calculateLogP(); // Compute distribution given _current_ value of parameter.

    }


    @Override
    public void run() throws Exception {

        double sgm_q = 1; // sd for the normal proposal distribution in the Markov_Step method
        int k = 50; // k parameter in the H_mk method
        int m = 500; // m parameter in the H_mk method

        // In case of bad initialization this "n_iter" parameter should be increased
        // to compensate for the poor choice of the initialization distribution
        // Here the initialization is deliberately bad, and as a consequence "n_iter" has to be large


        // I prepare a matrix "M" to store the values of the 64 indicator function observables that i am interested in
        // I am interested in these 64 indicator functions because I want to use them to make a histogram of the
        // distribution obtained by using the Markov Chains
        double[][] M = new double[numberOfChains][64];
        double avg = 0; // I use this variable to count the average number of steps taken for two chains to meet

        for (int i = 0; i < numberOfChains; i++) {
            // I perform the first steps of the two chains. Two for X and one for Y.
            List<Double> X = new ArrayList<>();
            List<Double> Y = new ArrayList<>();
            double X_0 = 10 + Randomizer.nextGaussian();// Poor proposal distribution
            X.add(X_0);
            double X_1 = Markov_Step(X_0, sgm_q);
            X.add(X_1);
            double Y_0 = 10 + Randomizer.nextGaussian(); // Poor proposal distribution
            Y.add(Y_0);
            double X_t = X_1;
            double Y_t_prev = Y_0;
            int count = 0;

            int steps_b4_meet = 1;
            while (count < m - 1) {
                double[] v;
                v = Max_Coup(X_t, sgm_q, Y_t_prev, sgm_q);
                double X_star = v[0];
                double Y_star = v[1];
                double U = Randomizer.nextDouble();
                if (U <= Math.min(1, Probability(X_star, X_t))) { X_t = X_star; }
                if (U <= Math.min(1, Probability(Y_star, Y_t_prev))) { Y_t_prev = Y_star; }
                X.add(X_t);
                Y.add(Y_t_prev);
                if (Y_t_prev != X_t) { steps_b4_meet += 1; }
                count += 1; }

            while (Y_t_prev != X_t) {
                steps_b4_meet += 1;
                double[] v;
                v = Max_Coup(X_t, sgm_q, Y_t_prev, sgm_q);
                double X_star = v[0];
                double Y_star = v[1];
                double U = Randomizer.nextDouble();
                if (U <= Math.min(1, Probability(X_star, X_t))) { X_t = X_star; }
                if (U <= Math.min(1, Probability(Y_star, Y_t_prev))) { Y_t_prev = Y_star; }
                X.add(X_t);
                Y.add(Y_t_prev);
            }


            for (int j = 0; j < 64; j++) {
                M[i][j] = H_mk(m, k, X, Y, -8 + j * 0.25, -8 + (j + 1) * 0.25);
            }

            avg += steps_b4_meet / (double) numberOfChains;
            if (X.size() > 501) {
                System.out.println("Outlier!");
                System.out.println("Outlier contribution" + H_mk(m, k, X, Y, 2.25, 2.5));
            }
        }

        // I print the results for the observables values to a text file
        double[] values;
        values = new double[64];

        FileOutputStream fout=new FileOutputStream("mfile.txt");
        PrintStream pout=new PrintStream(fout);

        for (int j = 0; j < 64; j++) {
            double temp = 0;
            for (int i = 0; i < numberOfChains; i++) {
                temp += M[i][j];
            }
            temp = temp/(double) numberOfChains;
            values[j] = temp;
            pout.println(temp);
        }

        pout.close();
        fout.close();

    }

    public double[] Max_Coup(double m_1, double sgm_1, double m_2, double sgm_2) {
        double[] v;
        v = new double[2];
        double X = m_1 + sgm_1 * Randomizer.nextGaussian();
        double a = 1 / (sgm_1 * Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * Math.pow((X - m_1) / sgm_1, 2));
        double U = Randomizer.nextDouble() * a;
        double b = 1 / (sgm_2 * Math.sqrt(2 * Math.PI)) * Math.exp((-0.5 * Math.pow((X - m_2) / sgm_2, 2)));
        if (U < b) {
            v[0] = X;
            v[1] = X;
        } else {
            boolean check = false;
            while (!check) {
                double Y = m_2 + sgm_2 * Randomizer.nextGaussian();
                double c = 1 / (sgm_2 * Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * Math.pow((Y - m_2) / sgm_2, 2));
                double d = 1 / (sgm_1 * Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * Math.pow((Y - m_1) / sgm_1, 2));
                double V = Randomizer.nextDouble() * c;
                if (V > d) {
                    check = true;
                    v[0] = X;
                    v[1] = Y;
                }
            }
        }
        return v;
    }

    // This method returns the result of a Markov step starting in X_t
    // With a normal distribution function (center = 0, sd=1) as proposal distribution
    // Be careful! This method has to be modified in case you want to use it for a different probability distribution
    // As it is, it only works for the linear combination of two normal distributions centered in +/-4 with sd=1
    public double Markov_Step(double X_t, double sgm_q) {
        double prop = X_t + sgm_q * Randomizer.nextGaussian();
        x.setValue(prop);
        double num = distribution.calculateLogP();
        x.setValue(X_t);
        double den = distribution.calculateLogP();
        double log_alpha = num - den;
        double U = Randomizer.nextDouble();
        if (Math.exp(log_alpha) >= U) {
            return prop; }
        else { return X_t; }
    }

    // I need to use this method in the main. Again be careful! It needs to be changed
    // if you intend to sample probability distributions that are not
    // the linear combinations of normal distributions mentioned above
    public double Probability(double z, double w) {
        x.setValue(z);
        double a = distribution.calculateLogP();
        x.setValue(w);
        double b = distribution.calculateLogP();
        double c = a - b;
        // Here you previously used an if to prevent the division by zero
        // However the usage of Log should prevent it as well, even better hopefully.
        c = Math.exp(c);
        return c;
     }

    // This is method that is used to evaluate observables on the samples produced
    // by the Markov chains without wasting useful information.
    // The implementation of this method here only accounts observables that are indicator functions
    // the "start" and "finish" parameters set the boundaries of the indicator function
    // Be careful! If you want to consider different observables you have to modify
    // this method accordingly to your needs
    public double H_mk(int m, int k, List<java.lang.Double> X, List<java.lang.Double> Y, double start, double finish) {
        double term_1 = 0;
        for (int i = 0; i < (m - k); i++) {
            if (X.get(k + i) <= finish & X.get(k + i) >= start) { term_1 += 1; }
        }
        term_1 = term_1 * 1 / (m - k - 1);
        int meet = 1;
        while (!X.get(meet).equals(Y.get(meet - 1))) { meet += 1; }
        double term_2 = 0;
        for (int l = 0; l < (meet - k - 1); l++) {
            int a = 0;
            int b = 0;
            if (X.get(l + 1 + k) <= finish & X.get(l + 1 + k) >= start) { a += 1; }
            if (Y.get(l + k) <= finish & Y.get(l + k) >= start) { b += 1; }
            term_2 += Math.min(1, (l + 1) / (m - k + 1)) * (a - b);
        }
        return term_1 + term_2;
    }

    public static void main(String[] args) throws Exception {

        CoupledMCMC coupledMCMC = new CoupledMCMC();
        coupledMCMC.initByName("maxChainLength", "1000");
        coupledMCMC.run();

    }

}
