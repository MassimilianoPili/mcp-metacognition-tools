package io.github.massimilianopili.mcp.metacognition;

/**
 * Gaussian Process regression with RBF (squared exponential) kernel.
 * Pure Java implementation using Cholesky decomposition.
 * Designed for n less than 1000 training points (Cholesky is O(n^3)).
 *
 * <p>Adams and MacKay (2007) showed that Bayesian methods with proper
 * uncertainty quantification outperform point estimates for sequential
 * decision making — the sigma2 output is as important as mu.</p>
 */
public class GaussianProcessEngine {

    private final double lengthScale;
    private final double noiseVariance;
    private final double minSigma2;

    public GaussianProcessEngine(double lengthScale, double noiseVariance, double minSigma2) {
        this.lengthScale = lengthScale;
        this.noiseVariance = noiseVariance;
        this.minSigma2 = minSigma2;
    }

    /**
     * RBF kernel: k(x,y) = exp(-||x-y||^2 / (2 * lengthScale^2))
     */
    public double rbfKernel(double[] x, double[] y) {
        double sqDist = 0.0;
        int len = Math.min(x.length, y.length);
        for (int i = 0; i < len; i++) {
            double d = x[i] - y[i];
            sqDist += d * d;
        }
        return Math.exp(-sqDist / (2.0 * lengthScale * lengthScale));
    }

    /**
     * GP posterior prediction for a test point given training data.
     *
     * @param trainX training embeddings, shape [n][dim]
     * @param trainY training outcomes (quality scores in [0,1]), shape [n]
     * @param testX  test embedding, shape [dim]
     * @return double[2]: {mu, sigma2} — posterior mean and variance
     */
    public double[] predict(double[][] trainX, double[] trainY, double[] testX) {
        int n = trainX.length;

        if (n == 0) {
            return new double[]{0.5, 1.0}; // uninformative prior
        }

        // K = kernel matrix [n x n] + noise on diagonal
        double[][] K = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double k = rbfKernel(trainX[i], trainX[j]);
                K[i][j] = k;
                K[j][i] = k;
            }
            K[i][i] += noiseVariance;
        }

        // k* = kernel vector between test point and training points
        double[] kStar = new double[n];
        for (int i = 0; i < n; i++) {
            kStar[i] = rbfKernel(trainX[i], testX);
        }

        // k** = self-kernel of test point
        double kStarStar = 1.0 + noiseVariance; // rbfKernel(testX, testX) = 1.0

        // Cholesky with progressive jitter for numerical stability
        double[][] L = choleskyWithJitter(K);
        if (L == null) {
            return new double[]{0.5, 1.0}; // fallback if decomposition fails
        }

        // mu* = k*^T K^{-1} y  (via forward+backward solve)
        double[] alpha = choleskyBackwardSolve(L, choleskyForwardSolve(L, trainY));
        double mu = 0.0;
        for (int i = 0; i < n; i++) {
            mu += kStar[i] * alpha[i];
        }

        // sigma2* = k** - k*^T K^{-1} k*  (via forward solve)
        double[] v = choleskyForwardSolve(L, kStar);
        double sigma2 = kStarStar;
        for (int i = 0; i < n; i++) {
            sigma2 -= v[i] * v[i];
        }
        sigma2 = Math.max(sigma2, minSigma2);

        // Clamp mu to [0, 1] (quality scores are bounded)
        mu = Math.max(0.0, Math.min(1.0, mu));

        return new double[]{mu, sigma2};
    }

    /**
     * Cholesky decomposition with progressive jitter.
     * Tries jitter levels: 0, 1e-8, 1e-6, 1e-4, 1e-2.
     * Returns L where K = L * L^T, or null if all fail.
     */
    private double[][] choleskyWithJitter(double[][] A) {
        double[] jitters = {0.0, 1e-8, 1e-6, 1e-4, 1e-2};
        int n = A.length;

        for (double jitter : jitters) {
            double[][] M = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(A[i], 0, M[i], 0, n);
                M[i][i] += jitter;
            }
            double[][] L = choleskyDecomposition(M);
            if (L != null) return L;
        }
        return null;
    }

    /**
     * Standard Cholesky decomposition: A = L * L^T.
     * Returns null if matrix is not positive definite.
     */
    private double[][] choleskyDecomposition(double[][] A) {
        int n = A.length;
        double[][] L = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) {
                    double val = A[i][i] - sum;
                    if (val <= 0) return null; // not positive definite
                    L[i][j] = Math.sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return L;
    }

    /**
     * Forward substitution: solve L * x = b.
     */
    private double[] choleskyForwardSolve(double[][] L, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / L[i][i];
        }
        return x;
    }

    /**
     * Backward substitution: solve L^T * x = b.
     */
    private double[] choleskyBackwardSolve(double[][] L, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += L[j][i] * x[j]; // L^T[i][j] = L[j][i]
            }
            x[i] = (b[i] - sum) / L[i][i];
        }
        return x;
    }
}
