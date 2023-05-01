
import java.util.Arrays;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.inference.TestUtils;

import org.testng.annotations.Test;
import smile.math.Random;

import static org.testng.Assert.assertEquals;
/**
 * Test for class.
 */
public class PolynomialCurveFitterTest {
    @Test
    public void testFit() {
        final RealDistribution rng = new UniformRealDistribution(-100, 100);
        rng.reseedRandomGenerator(123L);

        final double[] coeff = { 12.9, -3.4, 2.1 }; // 12.9 - 3.4 x + 2.1 x^2
        final PolynomialFunction f = new PolynomialFunction(coeff);

        // Collect data from a known polynomial.
        final WeightedObservedPoints obs = new WeightedObservedPoints();
        for (int i = 0; i < 100; i++) {
            final double x = rng.sample();
            obs.add(x, f.value(x));
        }

        // Start fit from initial guesses that are far from the optimal values.
        final PolynomialCurveFitter fitter
                = PolynomialCurveFitter.create(0).withStartPoint(new double[] { -1e-20, 3e15, -5e25 });
        final double[] best = fitter.fit(obs.toList());

        //TestUtils. assertEquals("best != coeff", coeff, best, 1e-
        System.out.println("coeff: " + Arrays.toString(coeff));
        System.out.println("best: " + Arrays.toString(best));
    }

    @Test
    public void testFit2() {
        final RealDistribution rng = new UniformRealDistribution(-100, 100);
        rng.reseedRandomGenerator(641223L);

        final double[] coeff = { 0. };
        final PolynomialFunction f = new PolynomialFunction(coeff);

        // Collect data from a known polynomial.
        final WeightedObservedPoints obs = new WeightedObservedPoints();
        for (int i = 0; i < 100; i++) {
            final double x = rng.sample();
            System.out.println("x: " + x + " f(x): " + f.value(x));
            obs.add(x, f.value(x));
        }
        smile.math.Random rnd = new Random();
        rnd.setSeed(1);
        double [] init = new double[1];
        for(int i = 0; i < init.length; i++)
            init[i] = rnd.nextDouble();

        System.out.println("init: " + Arrays.toString(init));
        // Start fit from initial guesses that are far from the optimal values.
        final PolynomialCurveFitter fitter
                = PolynomialCurveFitter.create(0).withStartPoint(init);
        final double[] best = fitter.fit(obs.toList());

        //TestUtils. assertEquals("best != coeff", coeff, best, 1e-
        System.out.println("coeff: " + Arrays.toString(coeff));
        System.out.println("best: " + Arrays.toString(best));
    }
/*
    @Test
    public void testNoError() {
        final Random randomizer = new Random(64925784252l);
        for (int degree = 1; degree < 10; ++degree) {
            final PolynomialFunction p = buildRandomPolynomial(degree, randomizer);
            final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);

            final WeightedObservedPoints obs = new WeightedObservedPoints();
            for (int i = 0; i <= degree; ++i) {
                obs.add(1.0, i, p.value(i));
            }

            final PolynomialFunction fitted = new PolynomialFunction(fitter.fit(obs.toList()));

            for (double x = -1.0; x < 1.0; x += 0.01) {
                final double error = FastMath.abs(p.value(x) - fitted.value(x)) /
                        (1.0 + FastMath.abs(p.value(x)));
                //assertEquals.assertEquals(0.0, error);
                System.out.printf("%04f", error);
            }
        }
    }
/*
    @Test
    public void testSmallError() {
        final Random randomizer = new Random(53882150042l);
        double maxError = 0;
        for (int degree = 0; degree < 10; ++degree) {
            final PolynomialFunction p = buildRandomPolynomial(degree, randomizer);
            final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);

            final WeightedObservedPoints obs = new WeightedObservedPoints();
            for (double x = -1.0; x < 1.0; x += 0.01) {
                obs.add(1.0, x, p.value(x) + 0.1 * randomizer.nextGaussian());
            }

            final PolynomialFunction fitted = new PolynomialFunction(fitter.fit(obs.toList()));

            for (double x = -1.0; x < 1.0; x += 0.01) {
                final double error = FastMath.abs(p.value(x) - fitted.value(x)) /
                        (1.0 + FastMath.abs(p.value(x)));
                maxError = FastMath.max(maxError, error);
                Assert.assertTrue(FastMath.abs(error) < 0.1);
            }
        }
        Assert.assertTrue(maxError > 0.01);
    }

    @Test
    public void testRedundantSolvable() {
        // Levenberg-Marquardt should handle redundant information gracefully
        checkUnsolvableProblem(true);
    }

    @Test
    public void testLargeSample() {
        final Random randomizer = new Random(0x5551480dca5b369bl);
        double maxError = 0;
        for (int degree = 0; degree < 10; ++degree) {
            final PolynomialFunction p = buildRandomPolynomial(degree, randomizer);
            final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);

            final WeightedObservedPoints obs = new WeightedObservedPoints();
            for (int i = 0; i < 40000; ++i) {
                final double x = -1.0 + i / 20000.0;
                obs.add(1.0, x, p.value(x) + 0.1 * randomizer.nextGaussian());
            }

            final PolynomialFunction fitted = new PolynomialFunction(fitter.fit(obs.toList()));
            for (double x = -1.0; x < 1.0; x += 0.01) {
                final double error = FastMath.abs(p.value(x) - fitted.value(x)) /
                        (1.0 + FastMath.abs(p.value(x)));
                maxError = FastMath.max(maxError, error);
                Assert.assertTrue(FastMath.abs(error) < 0.01);
            }
        }
        Assert.assertTrue(maxError > 0.001);
    }

    private void checkUnsolvableProblem(boolean solvable) {
        final Random randomizer = new Random(1248788532l);

        for (int degree = 0; degree < 10; ++degree) {
            final PolynomialFunction p = buildRandomPolynomial(degree, randomizer);
            final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);
            final WeightedObservedPoints obs = new WeightedObservedPoints();
            // reusing the same point over and over again does not bring
            // information, the problem cannot be solved in this case for
            // degrees greater than 1 (but one point is sufficient for
            // degree 0)
            for (double x = -1.0; x < 1.0; x += 0.01) {
                obs.add(1.0, 0.0, p.value(0.0));
            }

            try {
                fitter.fit(obs.toList());
                Assert.assertTrue(solvable || (degree == 0));
            } catch(ConvergenceException e) {
                Assert.assertTrue((! solvable) && (degree > 0));
            }
        }
    }

    private PolynomialFunction buildRandomPolynomial(int degree, Random randomizer) {
        final double[] coefficients = new double[degree + 1];
        for (int i = 0; i <= degree; ++i) {
            coefficients[i] = randomizer.nextGaussian();
        }
        return new PolynomialFunction(coefficients);
    }

 */
}
