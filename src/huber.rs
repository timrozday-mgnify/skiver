pub fn huber_ridge_fit_1d(
    x: &[f32],
    y: &[f32],
    delta: f32,
    lambda: f32,
    max_iter: usize,
    tol: f32,
) -> (f32, f32) {
    let n = x.len();
    if n < 2 || n != y.len() || !(delta > 0.0) || lambda < 0.0 || max_iter == 0 || tol <= 0.0 {
        return (0.0, 0.0);
    }

    //println!("x: {:?}", x);
    //println!("y: {:?}", y);

    // Solve weighted ridge regression (slope regularized, intercept not)
    #[inline]
    fn solve_weighted_ridge(x: &[f32], y: &[f32], w: &[f32], lambda: f32) -> Option<(f32, f32)> {
        let mut sw = 0.0f32;
        let mut sx = 0.0f32;
        let mut sxx = 0.0f32;
        let mut sy = 0.0f32;
        let mut sxy = 0.0f32;

        for i in 0..x.len() {
            let wi = w[i];
            if !wi.is_finite() || wi < 0.0 {
                return None;
            }
            let xi = x[i];
            let yi = y[i];
            sw += wi;
            sx += wi * xi;
            sxx += wi * xi * xi;
            sy += wi * yi;
            sxy += wi * xi * yi;
        }

        // Normal equations:
        // (sxx + lambda) a + sx b = sxy
        // sx a + sw b = sy
        let a11 = sxx + lambda;
        let a12 = sx;
        let a21 = sx;
        let a22 = sw;

        let det = a11 * a22 - a12 * a21;
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }

        let slope = (sxy * a22 - sy * a12) / det;
        let intercept = (a11 * sy - a21 * sxy) / det;

        if slope.is_finite() && intercept.is_finite() {
            Some((slope, intercept))
        } else {
            None
        }
    }

    // --- Initialize with ordinary ridge ---
    let mut w = vec![1.0f32; n];
    let (mut a, mut b) = match solve_weighted_ridge(x, y, &w, lambda) {
        Some(v) => v,
        None => return (0.0, 0.0),
    };

    // --- IRLS loop ---
    for _ in 0..max_iter {
        // Update Huber weights
        for i in 0..n {
            let r = y[i] - (a * x[i] + b);
            let ar = r.abs();
            w[i] = if ar <= delta { 1.0 } else { delta / ar };
        }

        let (new_a, new_b) = match solve_weighted_ridge(x, y, &w, lambda) {
            Some(v) => v,
            None => return (0.0, 0.0),
        };

        let da = (new_a - a).abs();
        let db = (new_b - b).abs();

        a = new_a;
        b = new_b;

        if da.max(db) < tol {
            break;
        }
    }
    //println!("Weights: {:?}", w);

    (a, b)
}
