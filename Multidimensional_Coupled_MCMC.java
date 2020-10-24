package com.company;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {

        double sgm_q = 1; // sd for the normal proposal distribution in the Markov_Step method
        int k = 50; // k parameter in the H_mk method
        int m = 500; // m parameter in the H_mk method
        int n_iter = 1000; // number of couples of chains that we want to draw
        int int_len = 16; // length of the interval that I want to explore
        int points = 64;// number of points in the (1d projection of the) interval that i explore
        double incr = (double) int_len/(double) points; // increment that I use in the indicator function evaluation
        double[][][] M = new double[n_iter][points][points];

        FileOutputStream fout = new FileOutputStream("mfile.txt");
        PrintStream pout=new PrintStream(fout);

        // In case of bad initialization this "n_iter" parameter should be increased
        // to compensate for the poor choice of the initialization distribution
        // Here the initialization is deliberately bad, and as a consequence "n_iter" has to be large

        double avg = 0; // I use this variable to count the average number of steps taken for two chains to meet

        for (int i = 0; i < n_iter; i++) {
            System.out.println(i);
            // I perform the first steps of the two chains. Two for X and one for Y.
            List<double[]> X = new ArrayList<>();
            List<double[]> Y = new ArrayList<>();
            Random r = new Random();
            double[] X_0 = { r.nextGaussian(), r.nextGaussian() +10}; //Bad initialization is chosen
            X.add(X_0);
            double[] X_1 = Markov_Step(X_0, sgm_q);
            X.add(X_1);
            double[] Y_0 = { r.nextGaussian(), r.nextGaussian() +10}; //Bad initialization is chosen
            Y.add(Y_0);
            double[] X_t = X_1;
            double[] Y_t_prev = Y_0;
            int count = 0;

            int steps_b4_meet = 1;
            while (count < m - 1) {
                double[][] v;
                // Sometimes it enters Max_coup and never leaves it, figure out why
                v = Max_Coup(X_t, Y_t_prev, sgm_q);
                double[] X_star = v[0];
                double[] Y_star = v[1];
                double U = r.nextDouble();
                if (U <= Math.min(1, Probability(X_star, X_t))) {
                    X_t = X_star;
                }
                if (U <= Math.min(1, Probability(Y_star, Y_t_prev))) {
                    Y_t_prev = Y_star;
                }
                X.add(X_t);
                Y.add(Y_t_prev);
                if ((Y_t_prev[0] != X_t[0]) && (Y_t_prev[1] != X_t[1])) {
                    steps_b4_meet += 1;
                }
                count += 1;
            }
            if ((Y_t_prev[0] != X_t[0]) && (Y_t_prev[1] != X_t[1])) {
                System.out.println("Hello!");
            }
            while ((Y_t_prev[0] != X_t[0]) && (Y_t_prev[1] != X_t[1])) {
                steps_b4_meet += 1;
                double[][] v;
                v = Max_Coup(X_t, Y_t_prev, sgm_q);
                double[] X_star = v[0];
                double[] Y_star = v[1];
                double U = r.nextDouble();
                if (U <= Math.min(1, Probability(X_star, X_t))) {
                    X_t = X_star;
                }
                if (U <= Math.min(1, Probability(Y_star, Y_t_prev))) {
                    Y_t_prev = Y_star;
                }
                X.add(X_t);
                Y.add(Y_t_prev);
            }

            avg += steps_b4_meet / (double) n_iter;

            for (int j = 0; j < points; j++) {
                for (int l = 0; l < points; l++) {
                    double[] start = {-8 + j * incr, -8 + l * incr};
                    double[] finish = {-8 + (j + 1) * incr, -8 + (l + 1) * incr};
                    M[i][j][l] = H_mk(m, k, X, Y, start, finish);
                }
            }
        }

        double[][] result = new double[points][points];
        for (int j = 0; j < points; j++) {
            for (int l = 0; l < points; l++) {
                double a = 0;
                for (int i = 0; i < n_iter; i++) {
                    a += M[i][j][l];
                }
                a = a / (double) n_iter;
                result[j][l] = a;
                System.out.println((-8 + j * incr) + " " + (-8 + l * incr) + " " + "---->" + " " + result[j][l]);
                pout.println( (float) result[j][l]);
            }
            pout.println( " " );
        }
        pout.close();
        fout.close();

        System.out.println("The average length of the chain is ---> " + avg);

    }

    public static double[][] Max_Coup(double[] m_1, double[] m_2, double sgm) {
        // You might want to set two distinct values for the variances along the 2 dimensions
        // Since, from tracer, you can see that the two distributions have very different variances
        double[][] v = new double[2][2];
        double[] X;
        double[] Y;
        X = new double[2];
        Y = new double[2];
        Random r = new Random();
        X[0] = m_1[0] + sgm * r.nextGaussian();
        X[1] = m_1[1] + sgm * r.nextGaussian();
        double a = 1 / (sgm * (2 * Math.PI)) * Math.exp(-0.5 * (Math.pow((X[0] - m_1[0]) / sgm, 2) + Math.pow((X[1] - m_1[1]) / sgm, 2)));
        double U = r.nextDouble() * a;
        double b = 1 / (sgm * (2 * Math.PI)) * Math.exp(-0.5 * (Math.pow((X[0] - m_2[0]) / sgm, 2) + Math.pow((X[1] - m_2[1]) / sgm, 2)));
        if (U < b) {
            v[0] = X;
            v[1] = X;
        } else {
            boolean check = false;
            while (!check) {
                Y[0] = m_2[0] + sgm * r.nextGaussian();
                Y[1] = m_2[1] + sgm * r.nextGaussian();
                double c = 1 / (sgm * (2 * Math.PI)) * Math.exp(-0.5 * (Math.pow((Y[0] - m_2[0]) / sgm, 2) + Math.pow((Y[1] - m_2[1]) / sgm, 2)));
                double d = 1 / (sgm * (2 * Math.PI)) * Math.exp(-0.5 * (Math.pow((Y[0] - m_1[0]) / sgm, 2) + Math.pow((Y[1] - m_1[1]) / sgm, 2)));
                double V = r.nextDouble() * c;
                if (V > d) {
                    check = true;
                    v[0] = X;
                    v[1] = Y;
                }
            }
        }
        return v;
    }

    public static double[] Markov_Step(double[] X_t, double sgm_q) {
        Random r = new Random();
        double[] prop = new double[2];
        prop[0] = X_t[0] + sgm_q * r.nextGaussian();
        prop[1] = X_t[1] + sgm_q * r.nextGaussian();
        // You might want to use the logarithm here
        double num = Math.exp(-0.5 * ((Math.pow((prop[0] - 4), 2) + Math.pow(prop[1], 2)))) + Math.exp(-0.5 * ((Math.pow((prop[0] + 4), 2) + Math.pow(prop[1], 2))));
        double den = Math.exp(-0.5 * ((Math.pow((X_t[0] - 4), 2) + Math.pow(X_t[1], 2)))) + Math.exp(-0.5 * ((Math.pow((X_t[0] + 4), 2) + Math.pow(X_t[1], 2))));
        double alpha = num / den;
        double U = r.nextDouble();
        if (alpha >= U) {
            return prop;
        } else {
            return X_t;
        }
    }

    public static double Probability(double[] z, double[] w) {
        // You might want to use the logarithm here
        double a = Math.exp(-0.5 * ((Math.pow((z[0] - 4), 2) + Math.pow(z[1], 2)))) + Math.exp(-0.5 * ((Math.pow((z[0] + 4), 2) + Math.pow(z[1], 2))));
        double b = Math.exp(-0.5 * ((Math.pow((w[0] - 4), 2) + Math.pow(w[1], 2)))) + Math.exp(-0.5 * ((Math.pow((w[0] + 4), 2) + Math.pow(w[1], 2))));
        double c = a / b;
        return c;
    }

    // This is method that is used to evaluate observables on the samples produced
    // by the Markov chains without wasting useful information.
    // The implementation of this method here only accounts observables that are indicator functions
    // the "start" and "finish" parameters set the boundaries of the indicator function
    // Be careful! If you want to consider different observables you have to modify
    // this method accordingly to your needs
    public static double H_mk(int m, int k, List<double[]> X, List<double[]> Y, double[] start, double[] finish) {
        double term_1 = 0;
        for (int i = 0; i < (m - k); i++) {
            if (X.get(k + i)[0] <= finish[0] & X.get(k + i)[0] >= start[0]) {
                if (X.get(k + i)[1] <= finish[1] & X.get(k + i)[1] >= start[1]) {
                    term_1 += 1;
                }
            }
        }
        term_1 = term_1 * 1 / (m - k - 1);
        int meet = 1;
        while (!X.get(meet).equals(Y.get(meet - 1))) {
            meet += 1;
        }
        double term_2 = 0;
        for (int l = 0; l < (meet - k - 1); l++) {
            int a = 0;
            int b = 0;
            if (X.get(l + 1 + k)[0] <= finish[0] & X.get(l + 1 + k)[0] >= start[0]) {
                if (X.get(l + 1 + k)[1] <= finish[1] & X.get(l + 1 + k)[1] >= start[1]) {
                    a += 1;
                }
                if (Y.get(l + k)[0] <= finish[0] & Y.get(l + k)[0] >= start[0]) {
                    if (Y.get(l + k)[1] <= finish[1] & Y.get(l + k)[1] >= start[1]) {
                        b += 1;
                    }
                }
            }
            term_2 += Math.min(1, (l + 1) / (m - k + 1)) * (a - b);
        }
        return term_1 + term_2;
    }

}
