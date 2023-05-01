
import com.github.jiamny.ml.utils.MathExHelper;

import org.testng.annotations.Test;
import org.testng.Assert;

import smile.data.DataFrame;

public class MathExHelperTest {

    @Test
    public void testArrayBindByCol() {
        double[] x1 = {1., 2., 3.};
        double[] x2 = {4., 5., 6.};

        double [][] x = MathExHelper.arrayBindByCol(x1, x2);

        String [] names = {"x", "y"};
        DataFrame xx = DataFrame.of(x, names);
        System.out.println(xx);

        System.exit(0);
    }
}
